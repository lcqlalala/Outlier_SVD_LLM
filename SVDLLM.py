#coding:utf8
import os
import sys
import argparse
import itertools
import inspect
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils.data_utils import *
from component.svd_llama import SVD_LlamaAttention, SVD_LlamaMLP
from component.svd_mistral import SVD_MistralAttention, SVD_MistralMLP
from component.svd_opt import SVDOPTDecoderLayer
from component.stable_svd_linear import StableSVDLinear
from utils.model_utils import *
from evaluater import * 

current_path = os.path.dirname(os.path.abspath(__file__))
parent_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(current_path)


def _move_llm_shared_modules(model_name, model, dev):
    # Llama-3.x keeps RoPE (rotary_emb) state at model level and may carry
    # non-buffer tensor attrs (e.g., original_inv_freq) that .to(dev) won't move.
    # In low-resource/sequential flows this can cause cpu/cuda mismatch.
    if "opt" in model_name:
        return

    def _move_rotary_module(m, device):
        if not isinstance(m, nn.Module):
            return
        m.to(device)
        # Move plain tensor attrs that are not registered buffers/parameters.
        param_names = set(getattr(m, "_parameters", {}).keys())
        buffer_names = set(getattr(m, "_buffers", {}).keys())
        for k, v in vars(m).items():
            if k in param_names or k in buffer_names:
                continue
            if torch.is_tensor(v):
                try:
                    setattr(m, k, v.to(device))
                except Exception:
                    pass

    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        _move_rotary_module(model.model.rotary_emb, dev)

    # Compatibility fallback for implementations that keep per-layer rotary modules.
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb"):
                _move_rotary_module(layer.self_attn.rotary_emb, dev)


def _register_rotary_runtime_device_guard(model_name, model):
    # Guard RoPE modules right before forward runs, so any lazily-updated
    # tensor state is always on the same device as runtime inputs.
    if "opt" in model_name:
        return []

    def _first_tensor_device(x):
        if torch.is_tensor(x):
            return x.device
        if isinstance(x, (list, tuple)):
            for y in x:
                d = _first_tensor_device(y)
                if d is not None:
                    return d
        if isinstance(x, dict):
            for y in x.values():
                d = _first_tensor_device(y)
                if d is not None:
                    return d
        return None

    def _sync_module_state(module, device):
        module.to(device)
        param_names = set(getattr(module, "_parameters", {}).keys())
        buffer_names = set(getattr(module, "_buffers", {}).keys())
        for k, v in vars(module).items():
            if k in param_names or k in buffer_names:
                continue
            if torch.is_tensor(v):
                try:
                    setattr(module, k, v.to(device))
                except Exception:
                    pass

    def _pre_hook(module, args):
        device = _first_tensor_device(args)
        if device is None:
            return
        _sync_module_state(module, device)

    handles = []
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb") and isinstance(model.model.rotary_emb, nn.Module):
        handles.append(model.model.rotary_emb.register_forward_pre_hook(_pre_hook))
    if hasattr(model, "model") and hasattr(model.model, "layers"):
        for layer in model.model.layers:
            if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "rotary_emb") and isinstance(layer.self_attn.rotary_emb, nn.Module):
                handles.append(layer.self_attn.rotary_emb.register_forward_pre_hook(_pre_hook))
    return handles



@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev, return_outlier_stats=False):
    _enforce_model_runtime_compat(name, model)
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    cov_dtype = torch.float64 if _use_high_precision_cov(name) else torch.float32
    def hook(module, input, output):
        inp = input[0].detach()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        inp_cov = inp.to(dtype=cov_dtype)
        adds = torch.matmul(inp_cov.transpose(1,2), inp_cov)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        channel_max_abs = inp.abs().amax(dim=(0, 1)).float()
        module.channel_max_abs = torch.maximum(module.channel_max_abs, channel_max_abs)
        del inp, inp_cov, adds, adds_sum
        torch.cuda.empty_cache()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module.raw_scaling_diag_matrix = 0
            module.channel_max_abs = torch.zeros(module.in_features, device=dev, dtype=torch.float32)
            module.register_forward_hook(hook)
    for batch in tqdm(calib_loader):
        batch = {k: v.to(dev) for k, v in batch.items()}
        model(**batch)
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            module._forward_hooks.clear()
    torch.cuda.empty_cache()
    model = model.cpu()
    outlier_stats = {}
    for i in range(len(layers)):
        subset = find_layers(layers[i])
        layer_outlier = {}
        for name in subset:
            subset[name].raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.cpu()
            layer_outlier[name] = subset[name].channel_max_abs.cpu()
        outlier_stats[i] = layer_outlier
    profiling_mat = {}
    print("Start Cholesky Decomposition...")
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        subset = find_layers(layers[i])
        for name in subset:
            raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix.double().to(dev)
            scaling_diag_matrix = _safe_cholesky(raw_scaling_diag_matrix, dev)
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        profiling_mat[i] = layer_profile
    if return_outlier_stats:
        return profiling_mat, outlier_stats
    return profiling_mat
        

@torch.no_grad()
def profle_svdllm_low_resource(model_name, model, calib_loader, dev, return_outlier_stats=False):
    _enforce_model_runtime_compat(model_name, model)
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    _move_llm_shared_modules(model_name, model, dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    rotary_guard_handles = _register_rotary_runtime_device_guard(model_name, model)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            cache_position = kwargs.get("cache_position", None)
            if position_ids is None and cache_position is not None:
                position_ids = cache_position.unsqueeze(0)
            if attention_mask is not None:
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = attention_mask.cpu()
                else:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], attention_mask.cpu()), dim=0)
            if "opt" not in model_name and position_ids is not None:
                if cache['position_ids'] is None:
                    cache['position_ids'] = position_ids.cpu()
                else:
                    cache['position_ids'] = torch.cat((cache['position_ids'], position_ids.cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    for h in rotary_guard_handles:
        h.remove()
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:  
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    profiling_mat = {}
    outlier_stats = {}
    cov_dtype = torch.float64 if _use_high_precision_cov(model_name) else torch.float32
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer_outlier = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            inp_cov = inp.to(dtype=cov_dtype)
            adds = torch.matmul(inp_cov.transpose(1,2), inp_cov)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            channel_max_abs = inp.abs().amax(dim=(0, 1)).float()
            module.channel_max_abs = torch.maximum(module.channel_max_abs, channel_max_abs)
            del inp, inp_cov, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            subset[name].channel_max_abs = torch.zeros(subset[name].in_features, device=dev, dtype=torch.float32)
            handles.append(subset[name].register_forward_hook(hook))
        _layer_forward_pass(
            model=model,
            model_name=model_name,
            layer=layer,
            inps=inps,
            attention_masks=attention_masks,
            position_ids=position_ids if "opt" not in model_name else None,
            dev=dev,
            outs=outs,
            max_batches=None,
        )
        for h in handles:
            h.remove()
        layer = layer.cpu()
        for name in subset:
            subset[name].scaling_diag_matrix = subset[name].scaling_diag_matrix.cpu()
            subset[name].channel_max_abs = subset[name].channel_max_abs.cpu()
            layer_outlier[name] = subset[name].channel_max_abs
        torch.cuda.empty_cache()
        for name in subset:
            raw_scaling_diag_matrix = subset[name].scaling_diag_matrix.double().to(dev)
            scaling_diag_matrix = _safe_cholesky(raw_scaling_diag_matrix, dev)
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        outlier_stats[i] = layer_outlier
        inps = outs
        torch.cuda.empty_cache()
    _move_llm_shared_modules(model_name, model, "cpu")
    if return_outlier_stats:
        return profiling_mat, outlier_stats
    return profiling_mat
     
 
def _safe_cholesky(raw_scaling_diag_matrix, dev):
    # Numerical-stable Cholesky for large covariance matrices.
    # We factor a correlation-preconditioned matrix to avoid adding one large
    # uniform jitter that would wash out low-variance LLaMA-3.x channels.
    if raw_scaling_diag_matrix.dtype in (torch.float64, torch.double):
        mat = raw_scaling_diag_matrix.to(device=dev, dtype=torch.float64)
    else:
        mat = raw_scaling_diag_matrix.to(device=dev, dtype=torch.float32)
    mat = 0.5 * (mat + mat.transpose(0, 1))

    n = mat.shape[0]
    finfo = torch.finfo(mat.dtype)
    diag = torch.diagonal(mat)
    diag_floor = torch.clamp(torch.mean(torch.abs(diag)) * 1e-12, min=finfo.eps)
    scale = torch.sqrt(torch.clamp(diag, min=diag_floor))

    corr = mat / torch.clamp(scale[:, None] * scale[None, :], min=diag_floor)
    corr = 0.5 * (corr + corr.transpose(0, 1))
    corr.diagonal().copy_(torch.ones(n, device=dev, dtype=mat.dtype))

    eye = torch.eye(n, device=dev, dtype=mat.dtype)
    base_jitter = 1e-8 if mat.dtype == torch.float64 else 1e-5
    jitter = base_jitter
    for _ in range(8):
        chol_corr, info = torch.linalg.cholesky_ex(corr + jitter * eye)
        if int(info.max().item()) == 0:
            return scale.unsqueeze(1) * chol_corr
        jitter *= 10.0

    print("Warning: correlation matrix is not positive definite, fallback to eigenvalue clipping.")
    try:
        evals, evecs = torch.linalg.eigh(corr)
        eval_floor = torch.tensor(base_jitter, device=dev, dtype=mat.dtype)
        evals = torch.clamp(evals, min=eval_floor)
        corr = (evecs * evals.unsqueeze(0)) @ evecs.transpose(0, 1)
        corr = 0.5 * (corr + corr.transpose(0, 1))
        chol_corr = torch.linalg.cholesky(corr + base_jitter * eye)
        return scale.unsqueeze(1) * chol_corr
    except RuntimeError as exc:
        print(f"Warning: GPU eigenvalue clipping failed ({exc}); retrying on CPU.")
        corr_cpu = corr.detach().to(device="cpu", dtype=torch.float64)
        evals, evecs = torch.linalg.eigh(corr_cpu)
        eval_floor = torch.tensor(base_jitter, device="cpu", dtype=torch.float64)
        evals = torch.clamp(evals, min=eval_floor)
        corr_cpu = (evecs * evals.unsqueeze(0)) @ evecs.transpose(0, 1)
        corr_cpu = 0.5 * (corr_cpu + corr_cpu.transpose(0, 1))
        eye_cpu = torch.eye(n, device="cpu", dtype=torch.float64)
        chol_corr = torch.linalg.cholesky(corr_cpu + base_jitter * eye_cpu)
        return scale.unsqueeze(1) * chol_corr.to(device=dev, dtype=mat.dtype)


def _right_project_from_cholesky(VT, scaling_diag_matrix):
    """Compute VT @ L^{-1} without explicitly forming inv(L)."""
    # If L is lower triangular, (VT @ L^{-1})^T solves L^T X = VT^T.
    lhs = scaling_diag_matrix.transpose(0, 1)
    rhs = VT.transpose(0, 1)
    try:
        solved_t = torch.linalg.solve_triangular(lhs, rhs, upper=True)
    except Exception:
        # Fallback is still a solve, not an explicit inverse.
        solved_t = torch.linalg.solve(lhs, rhs)
    return solved_t.transpose(0, 1)


def _safe_svd(matrix, name=""):
    try:
        return torch.linalg.svd(matrix, full_matrices=False)
    except Exception:
        if matrix.is_cuda:
            try:
                print(f"Warning: SVD failed to converge on {name}, falling back to gesvd driver.")
                return torch.linalg.svd(matrix, full_matrices=False, driver="gesvd")
            except Exception:
                pass
        # Last resort fallback: CPU double SVD, then cast back.
        print(f"Warning: SVD fallback to CPU on {name}.")
        cpu_matrix = matrix.detach().to(device="cpu", dtype=torch.float64)
        U, singular_values, VT = torch.linalg.svd(cpu_matrix, full_matrices=False)
        out_dtype = matrix.dtype
        out_dev = matrix.device
        return (
            U.to(device=out_dev, dtype=out_dtype),
            singular_values.to(device=out_dev, dtype=out_dtype),
            VT.to(device=out_dev, dtype=out_dtype),
        )


def _target_rank(rows, cols, ratio, max_rank):
    if max_rank <= 0:
        return 0
    rank = int(rows * cols * ratio / (rows + cols))
    rank = max(1, rank)
    return min(rank, max_rank)


def _is_llama3_model(model_name_or_path):
    name = str(model_name_or_path).lower()
    return "llama-3" in name or "llama3" in name


def _effective_rank_ratio_for_module(model_name_or_path, model_config, module_name, ratio):
    """Relax K/V rank for LLaMA-3.x GQA without touching non-GQA paths."""
    if not _is_llama3_model(model_name_or_path):
        return ratio
    if not _is_gqa_kv_module_name(module_name):
        return ratio

    num_heads = getattr(model_config, "num_attention_heads", None)
    num_kv_heads = getattr(model_config, "num_key_value_heads", None)
    if num_heads is None or num_kv_heads is None or num_kv_heads <= 0:
        return ratio

    group_size = max(1, int(num_heads) // int(num_kv_heads))
    return min(1.0, ratio * group_size)


def _is_gqa_kv_module_name(module_name):
    return module_name.endswith("k_proj") or module_name.endswith("v_proj")


def _should_keep_gqa_kv_uncompressed(model_name_or_path, model_config, module_name):
    # Diagnostic LLaMA-3.x adaptation: K/V are the shared GQA bottleneck.
    # Keep them as native nn.Linear to verify whether K/V compression is the
    # source of the high-PPL collapse before trying block-wise KV SVD.
    if not _is_llama3_model(model_name_or_path):
        return False
    if not _is_gqa_kv_module_name(module_name):
        return False
    num_heads = getattr(model_config, "num_attention_heads", None)
    num_kv_heads = getattr(model_config, "num_key_value_heads", None)
    return num_heads is not None and num_kv_heads is not None and int(num_heads) > int(num_kv_heads)


def _gqa_kv_rank_multiplier(model_name_or_path, model_config):
    if not _is_llama3_model(model_name_or_path):
        return 1
    num_heads = getattr(model_config, "num_attention_heads", None)
    num_kv_heads = getattr(model_config, "num_key_value_heads", None)
    if num_heads is None or num_kv_heads is None or num_kv_heads <= 0:
        return 1
    return max(1, int(num_heads) // int(num_kv_heads))


def _target_rank_for_module(model_name_or_path, model_config, module_name, rows, cols, ratio, max_rank):
    effective_ratio = _effective_rank_ratio_for_module(model_name_or_path, model_config, module_name, ratio)
    return _target_rank(rows, cols, effective_ratio, max_rank)


def _energy_conserving_recalibrate(full_singular_values, selected_idx, max_scale=None, eps=1e-12):
    selected_s = full_singular_values[selected_idx].float()
    if selected_s.numel() == 0:
        return selected_s, 1.0

    total_energy = torch.sum(full_singular_values.float() * full_singular_values.float())
    kept_energy = torch.sum(selected_s * selected_s)
    if kept_energy <= eps or total_energy <= eps:
        return selected_s, 1.0

    gamma = torch.sqrt(total_energy / torch.clamp(kept_energy, min=eps))
    if max_scale is not None and max_scale > 0:
        gamma = torch.clamp(gamma, max=max_scale)
    selected_s = selected_s * gamma
    return selected_s, float(gamma.item())


def _select_channel_partitions(scaling_diag_matrix, outlier_ratio, criterion="infinity_norm", channel_max_abs=None):
    in_features = scaling_diag_matrix.shape[0]
    all_indices = torch.arange(in_features, device=scaling_diag_matrix.device, dtype=torch.long)
    if outlier_ratio <= 0:
        return all_indices, torch.empty(0, device=scaling_diag_matrix.device, dtype=torch.long)

    num_outliers = int(in_features * outlier_ratio)
    num_outliers = max(0, min(num_outliers, in_features - 1))
    if num_outliers == 0:
        return all_indices, torch.empty(0, device=scaling_diag_matrix.device, dtype=torch.long)

    if criterion == "infinity_norm":
        if channel_max_abs is None:
            print("Warning: channel max abs stats are not available, fallback to energy criterion.")
            channel_score = torch.sum(scaling_diag_matrix.float() * scaling_diag_matrix.float(), dim=1)
        else:
            channel_score = channel_max_abs.to(scaling_diag_matrix.device).float()
    elif criterion == "energy":
        # diag(G) where G = S S^T, and S is Cholesky factor.
        channel_score = torch.sum(scaling_diag_matrix.float() * scaling_diag_matrix.float(), dim=1)
    else:
        raise ValueError(f"Unsupported outlier criterion: {criterion}")

    outlier_indices = torch.topk(channel_score, k=num_outliers, largest=True).indices
    outlier_indices, _ = torch.sort(outlier_indices)
    normal_mask = torch.ones(in_features, device=scaling_diag_matrix.device, dtype=torch.bool)
    normal_mask[outlier_indices] = False
    normal_indices = all_indices[normal_mask]
    return normal_indices, outlier_indices


def _project_box_with_fixed_mean(values, lower, upper, target_mean, max_iter=80):
    target_mean = float(target_mean)
    lower = float(lower)
    upper = float(upper)
    if lower > upper:
        lower, upper = upper, lower
    target_mean = min(max(target_mean, lower), upper)

    lo = -1e6
    hi = 1e6
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        cur_mean = torch.clamp(values + mid, min=lower, max=upper).mean().item()
        if cur_mean < target_mean:
            lo = mid
        else:
            hi = mid
    shift = 0.5 * (lo + hi)
    return torch.clamp(values + shift, min=lower, max=upper)


def _compute_laoa_layer_ratios(
    outlier_channel_stats,
    global_ratio,
    min_ratio=0.005,
    max_ratio=0.06,
    temperature=None,
    eps=1e-8,
):
    if outlier_channel_stats is None or global_ratio <= 0:
        return None

    if min_ratio > max_ratio:
        min_ratio, max_ratio = max_ratio, min_ratio
    if not (min_ratio <= global_ratio <= max_ratio):
        print(
            f"Warning: global Stage1 ratio={global_ratio:.6f} is outside "
            f"[{min_ratio:.6f}, {max_ratio:.6f}], LAOA falls back to uniform ratio."
        )
        return {i: float(global_ratio) for i in outlier_channel_stats}

    layer_ids = sorted(outlier_channel_stats.keys())
    score_list = []
    valid_layer_ids = []
    for i in layer_ids:
        channel_values = []
        for _, v in outlier_channel_stats[i].items():
            if v is not None and v.numel() > 0:
                channel_values.append(v.float().reshape(-1))
        if len(channel_values) == 0:
            continue
        c = torch.cat(channel_values, dim=0)
        # Absolute-Std scoring: emphasize absolute fluctuation energy
        # without dividing by mean (unlike CV).
        intensity = c.std(unbiased=False)
        score_list.append(float(intensity.item()))
        valid_layer_ids.append(i)

    if len(valid_layer_ids) == 0:
        return None

    score_tensor = torch.tensor(score_list, dtype=torch.float32)
    if temperature is None or temperature <= 0:
        temperature = float(torch.clamp(score_tensor.mean(), min=1e-6).item())
    logits = score_tensor / max(float(temperature), 1e-6)
    logits = logits - logits.max()
    weight = torch.softmax(logits, dim=0)

    raw_ratio = weight * (len(valid_layer_ids) * float(global_ratio))
    clipped_ratio = torch.clamp(raw_ratio, min=min_ratio, max=max_ratio)
    final_ratio = _project_box_with_fixed_mean(
        clipped_ratio,
        lower=min_ratio,
        upper=max_ratio,
        target_mean=float(global_ratio),
    )

    layer_ratio_map = {i: float(global_ratio) for i in layer_ids}
    for idx, layer_id in enumerate(valid_layer_ids):
        layer_ratio_map[layer_id] = float(final_ratio[idx].item())

    ratio_tensor = torch.tensor([layer_ratio_map[i] for i in layer_ids], dtype=torch.float32)
    print(
        "LAOA layer ratio stats: "
        f"mean={ratio_tensor.mean().item():.6f}, "
        f"min={ratio_tensor.min().item():.6f}, "
        f"max={ratio_tensor.max().item():.6f}, "
        f"temperature={float(temperature):.6f}"
    )
    return layer_ratio_map


def _get_parent_module(root_module, module_name):
    attrs = module_name.split(".")
    parent = root_module
    for attr in attrs[:-1]:
        parent = getattr(parent, attr)
    return parent, attrs[-1]


def _iter_batches(calib_loader, max_batches=None):
    if max_batches is None:
        return calib_loader
    if isinstance(calib_loader, list):
        return calib_loader[:max_batches]
    return itertools.islice(calib_loader, max_batches)


def _select_model_load_dtype(model_name_or_path):
    # Keep existing behavior for LLaMA-1/2 and others.
    # LLaMA-3.x official checkpoints are BF16-first.
    if _is_llama3_model(model_name_or_path) and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return torch.bfloat16
    return torch.float16


def _should_force_eager_attn(model_name_or_path):
    return _is_llama3_model(model_name_or_path)


def _select_model_load_kwargs(model_name_or_path):
    kwargs = {}
    if _should_force_eager_attn(model_name_or_path):
        kwargs["attn_implementation"] = "eager"
    return kwargs


def _use_high_precision_cov(model_name_or_path):
    # LLaMA-3.x has stronger activation spikes; use float64 covariance accumulation.
    return _should_force_eager_attn(model_name_or_path)


def _enforce_model_runtime_compat(model_name_or_path, model):
    if _should_force_eager_attn(model_name_or_path) and hasattr(model, "config"):
        if hasattr(model.config, "_attn_implementation"):
            model.config._attn_implementation = "eager"


def _hf_update_causal_mask(model, attention_mask, input_tensor, cache_position):
    if not (hasattr(model, "model") and hasattr(model.model, "_update_causal_mask")):
        return attention_mask
    fn = model.model._update_causal_mask
    try:
        sig = inspect.signature(fn).parameters
        kwargs = {}
        if "attention_mask" in sig:
            kwargs["attention_mask"] = attention_mask
        if "input_tensor" in sig:
            kwargs["input_tensor"] = input_tensor
        elif "hidden_states" in sig:
            kwargs["hidden_states"] = input_tensor
        if "cache_position" in sig:
            kwargs["cache_position"] = cache_position
        if "past_key_values" in sig:
            kwargs["past_key_values"] = None
        if "past_seen_tokens" in sig:
            kwargs["past_seen_tokens"] = 0
        if "use_cache" in sig:
            kwargs["use_cache"] = False
        if "output_attentions" in sig:
            kwargs["output_attentions"] = False
        return fn(**kwargs)
    except Exception:
        # Backward-compatible fallback for older signatures.
        try:
            return fn(attention_mask, input_tensor, cache_position, None)
        except Exception:
            try:
                return fn(attention_mask, input_tensor, cache_position, past_seen_tokens=0)
            except Exception:
                return attention_mask


def _build_decoder_layer_kwargs(
    model_name,
    model,
    layer,
    hidden_states,
    attention_mask=None,
    position_ids=None,
):
    if "opt" in model_name:
        kwargs = {}
        if attention_mask is not None:
            kwargs["attention_mask"] = attention_mask
        return kwargs

    x = hidden_states
    dev = x.device
    seq_len = x.shape[1]

    if position_ids is None:
        position_ids = torch.arange(seq_len, device=dev, dtype=torch.long).unsqueeze(0)
    else:
        position_ids = position_ids.to(dev)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0)

    if attention_mask is not None:
        attention_mask = attention_mask.to(dev)

    cache_position = position_ids[0]

    if _should_force_eager_attn(model_name):
        # For LLaMA-3.x eager path, build explicit 4D causal mask to avoid
        # private API behavior drift across HF versions.
        min_val = torch.finfo(x.dtype).min
        base_causal = torch.full(
            (seq_len, seq_len),
            fill_value=min_val,
            dtype=x.dtype,
            device=dev,
        )
        base_causal = torch.triu(base_causal, diagonal=1)
        causal_4d = base_causal[None, None, :, :].expand(x.shape[0], 1, seq_len, seq_len)

        if attention_mask is None:
            attention_mask = causal_4d
        elif torch.is_tensor(attention_mask) and attention_mask.dim() == 2:
            # Avoid min_val + min_val overflow in fp16/bf16 by masked_fill.
            pad_mask = attention_mask.to(device=dev, dtype=torch.bool)
            attention_mask = causal_4d.masked_fill(~pad_mask[:, None, None, :], min_val)
        elif torch.is_tensor(attention_mask) and attention_mask.dim() == 3:
            attention_mask = attention_mask.to(device=dev, dtype=x.dtype).unsqueeze(1)
        elif torch.is_tensor(attention_mask) and attention_mask.dim() == 4:
            attention_mask = attention_mask.to(device=dev, dtype=x.dtype)
    else:
        # Reuse HF internal causal-mask builder for missing/2D masks.
        need_update_causal_mask = (
            attention_mask is None
            or (torch.is_tensor(attention_mask) and attention_mask.dim() <= 2)
        )
        if need_update_causal_mask:
            attention_mask = _hf_update_causal_mask(
                model=model,
                attention_mask=attention_mask,
                input_tensor=x,
                cache_position=cache_position,
            )

    position_embeddings = None
    if hasattr(model, "model") and hasattr(model.model, "rotary_emb"):
        try:
            position_embeddings = model.model.rotary_emb(x, position_ids)
        except Exception:
            position_embeddings = None

    sig = inspect.signature(layer.forward).parameters
    kwargs = {}
    if "attention_mask" in sig:
        kwargs["attention_mask"] = attention_mask
    if "position_ids" in sig:
        kwargs["position_ids"] = position_ids
    if "output_attentions" in sig:
        kwargs["output_attentions"] = False
    if "use_cache" in sig:
        kwargs["use_cache"] = False
    if "cache_position" in sig:
        kwargs["cache_position"] = cache_position
    if "position_embeddings" in sig and position_embeddings is not None:
        kwargs["position_embeddings"] = position_embeddings
    return kwargs


@torch.no_grad()
def _layer_forward_pass(
    model,
    model_name,
    layer,
    inps,
    attention_masks,
    position_ids,
    dev,
    outs=None,
    max_batches=None,
):
    total = inps.shape[0]
    if max_batches is not None:
        total = min(total, int(max_batches))
    for j in range(total):
        if "opt" not in model_name:
            x = inps[j].unsqueeze(0)
            mask_j = attention_masks[j].unsqueeze(0).to(dev) if attention_masks is not None else None
            pos_j = position_ids[j].unsqueeze(0).to(dev) if position_ids is not None else None
            kwargs = _build_decoder_layer_kwargs(
                model_name=model_name,
                model=model,
                layer=layer,
                hidden_states=x,
                attention_mask=mask_j,
                position_ids=pos_j,
            )
            out = layer(x, **kwargs)[0]
        else:
            if attention_masks is not None:
                out = layer(
                    inps[j].unsqueeze(0),
                    attention_mask=attention_masks[j].unsqueeze(0).to(dev),
                )[0]
            else:
                out = layer(inps[j].unsqueeze(0))[0]
        if outs is not None:
            outs[j] = out
    return total


@torch.no_grad()
def _check_layer_replay_equivalence(
    model_name,
    model,
    layer,
    calib_loader,
    inps,
    attention_masks,
    position_ids,
    dev,
    max_batches=2,
):
    if "opt" in model_name or calib_loader is None:
        return

    total = min(int(max_batches), inps.shape[0])
    if total <= 0:
        return

    class _ReplayCheckStop(Exception):
        pass

    official_outs = []

    def _capture_layer_output(_module, _input, output):
        hidden = output[0] if isinstance(output, (tuple, list)) else output
        official_outs.append(hidden.detach().float().cpu())
        raise _ReplayCheckStop

    print(f"CCSR replay equivalence check: layer=0, batches={total}")
    handle = layer.register_forward_hook(_capture_layer_output)
    guard_handles = _register_rotary_runtime_device_guard(model_name, model)
    try:
        for batch in _iter_batches(calib_loader, total):
            try:
                batch = {k: v.to(dev) for k, v in batch.items()}
                model(**batch)
            except _ReplayCheckStop:
                pass
    finally:
        handle.remove()
        for h in guard_handles:
            h.remove()

    total = min(total, len(official_outs))
    if total <= 0:
        print("Warning: CCSR replay check captured no official outputs.")
        return

    replay_outs = torch.zeros_like(inps[:total])
    _layer_forward_pass(
        model=model,
        model_name=model_name,
        layer=layer,
        inps=inps[:total],
        attention_masks=attention_masks[:total] if attention_masks is not None else None,
        position_ids=position_ids[:total] if position_ids is not None else None,
        dev=dev,
        outs=replay_outs,
        max_batches=None,
    )

    official = torch.cat(official_outs[:total], dim=0)
    replay = replay_outs.detach().float().cpu()
    diff = replay - official
    official_norm = torch.norm(official)
    rel_l2 = torch.norm(diff) / torch.clamp(official_norm, min=1e-12)
    max_abs = diff.abs().max()
    mean_abs = diff.abs().mean()
    print(
        "CCSR replay check layer0: "
        f"rel_l2={rel_l2.item():.6e}, "
        f"max_abs={max_abs.item():.6e}, "
        f"mean_abs={mean_abs.item():.6e}"
    )
    if rel_l2.item() > 1e-3 or max_abs.item() > 1e-2:
        print("Warning: CCSR replay is not numerically equivalent to HF layer forward for LLaMA-3.x.")


@torch.no_grad()
def _apply_sam(
    model_name,
    model,
    decomposition_book,
    calib_loader,
    dev,
    sam_damp=1e-4,
    sam_max_batches=None,
    enable_ecsvr=False,
    ecsvr_max_scale=None,
):
    """
    Decoupled EC-SAM (Energy-Conserving SAM):
    safely refit low-rank up-projection with fixed v_proj, keep outlier branch untouched,
    and apply post-SAM energy compensation.
    """
    if calib_loader is None:
        print("Warning: SAM is enabled, but no calibration loader is provided. Skipping SAM.")
        return

    model = model.to(dev)
    model.eval()
    if "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print("Start Decoupled EC-SAM: Safely recalibrating low-rank subspace...")
    sam_rel_updates = []
    sam_total_modules = 0

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        runtime = {}
        handles = []

        if i not in decomposition_book:
            continue

        for name in decomposition_book[i]:
            try:
                parent_module, leaf_name = _get_parent_module(layer, name)
                module = getattr(parent_module, leaf_name)
            except Exception:
                continue
            if not isinstance(module, StableSVDLinear) or (not module.has_low_rank):
                continue

            info = decomposition_book[i][name]
            runtime[name] = {
                "module": module,
                "normal_idx": info["normal_idx"].to(dev),
                "U": info["U"].to(dev).float(),
                "S": info["singular_values"].to(dev).float(),
                "right_proj": info["right_proj"].to(dev).float(),
                "gram": None,
                "rhs": None,
                "target_energy": 0.0,
            }
            sam_total_modules += 1

        if len(runtime) == 0:
            continue

        def _make_hook(module_name):
            def _hook(module, input, _output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                inp_2d = inp.reshape(-1, inp.shape[-1])

                info = runtime[module_name]
                normal_idx = info["normal_idx"]
                x_normal = inp_2d if normal_idx.numel() == inp_2d.shape[-1] else inp_2d.index_select(-1, normal_idx)

                # Fixed low-rank features under current compressed branch.
                z = torch.matmul(x_normal, module.v_proj.weight.detach().float().transpose(0, 1))

                # Golden target: ideal normal-branch output only.
                z_full = torch.matmul(x_normal, info["right_proj"].transpose(0, 1))
                z_full = z_full * info["S"].view(1, -1)
                y_ideal = torch.matmul(z_full, info["U"].transpose(0, 1))

                if info["gram"] is None:
                    k = z.shape[-1]
                    info["gram"] = torch.zeros((k, k), device=dev, dtype=torch.float32)
                    info["rhs"] = torch.zeros((y_ideal.shape[-1], k), device=dev, dtype=torch.float32)

                info["gram"] += torch.matmul(z.transpose(0, 1), z)
                info["rhs"] += torch.matmul(y_ideal.transpose(0, 1), z)
                info["target_energy"] += float((y_ideal * y_ideal).sum().item())

            return _hook

        for name in runtime:
            handles.append(runtime[name]["module"].register_forward_hook(_make_hook(name)))

        for batch in _iter_batches(calib_loader, sam_max_batches):
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)

        for handle in handles:
            handle.remove()

        for name in runtime:
            info = runtime[name]
            module = info["module"]
            gram = info["gram"]
            rhs = info["rhs"]
            if gram is None or rhs is None or gram.shape[0] <= 0:
                continue

            k = gram.shape[0]
            trace = torch.trace(gram)
            damp = sam_damp * (trace / max(k, 1))
            gram_reg = gram + damp * torch.eye(k, device=dev, dtype=gram.dtype)

            try:
                solved = torch.linalg.solve(gram_reg, rhs.transpose(0, 1))
            except Exception:
                solved = torch.matmul(torch.linalg.pinv(gram_reg), rhs.transpose(0, 1))
            w_u = solved.transpose(0, 1)  # [d_out, rank]

            if enable_ecsvr:
                target_e = info["target_energy"]
                pred_e = float((torch.matmul(w_u, gram) * w_u).sum().item())
                if target_e > 1e-12 and pred_e > 1e-12:
                    gamma = (target_e / pred_e) ** 0.5
                    if ecsvr_max_scale is not None and ecsvr_max_scale > 0:
                        gamma = min(gamma, float(ecsvr_max_scale))
                    w_u = w_u * gamma

            old_u = module.u_proj.weight.data.detach().float()
            module.u_proj.weight.data = w_u.to(
                dtype=module.u_proj.weight.dtype,
                device=module.u_proj.weight.device,
            )
            rel_update = torch.norm(w_u - old_u.to(w_u.device)) / (torch.norm(old_u) + 1e-12)
            sam_rel_updates.append(float(rel_update.item()))

            runtime[name] = None
            torch.cuda.empty_cache()

    if sam_total_modules == 0:
        print("Warning: SAM did not find any StableSVDLinear modules to update.")
    if len(sam_rel_updates) > 0:
        rel_tensor = torch.tensor(sam_rel_updates, dtype=torch.float32)
        print(
            "SAM relative update stats: "
            f"mean={rel_tensor.mean().item():.6f}, "
            f"max={rel_tensor.max().item():.6f}"
        )

    model = model.cpu()


@torch.no_grad()
def whitening_sequential(
    model_name,
    model,
    ratio,
    dev,
    calib_loader,
    stage1_outlier_ratio=0.0,
    stage1_outlier_criterion="infinity_norm",
    stage1_layer_ratio_map=None,
    enable_stage3=False,
    stage3_lambda=1.0,
    stage3_max_batches=None,
    enable_ecsvr=False,
    ecsvr_max_scale=None,
    enable_sam=False,
    sam_damp=1e-4,
    sam_max_batches=None,
):
    if calib_loader is None:
        raise ValueError("CCSR requires calibration data. Please provide calib_loader.")

    _enforce_model_runtime_compat(model_name, model)
    model.eval()
    use_cache = model.config.use_cache
    model.config.use_cache = False
    gqa_multiplier = _gqa_kv_rank_multiplier(model_name, model.config)
    if gqa_multiplier > 1:
        kv_effective_ratio = min(1.0, ratio * gqa_multiplier)
        print(
            "LLaMA-3 GQA rank adaptation: "
            f"base_ratio={ratio:.6f}, "
            f"k_proj/v_proj effective_ratio={kv_effective_ratio:.6f}, "
            f"multiplier={gqa_multiplier}, cap=1.0"
        )

    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    _move_llm_shared_modules(model_name, model, dev)

    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size),
        dtype=dtype,
        device=dev,
    )
    cache = {"i": 0, "attention_mask": None, "position_ids": None}
    rotary_guard_handles = _register_rotary_runtime_device_guard(model_name, model)

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp.detach().to(dtype=dtype, device=dev)
            cache["i"] += 1
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            cache_position = kwargs.get("cache_position", None)
            if position_ids is None and cache_position is not None:
                position_ids = cache_position.unsqueeze(0)
            if attention_mask is not None:
                if cache["attention_mask"] is None:
                    cache["attention_mask"] = attention_mask.detach().cpu()
                else:
                    cache["attention_mask"] = torch.cat(
                        (cache["attention_mask"], attention_mask.detach().cpu()),
                        dim=0,
                    )
            if "opt" not in model_name and position_ids is not None:
                if cache["position_ids"] is None:
                    cache["position_ids"] = position_ids.detach().cpu()
                else:
                    cache["position_ids"] = torch.cat(
                        (cache["position_ids"], position_ids.detach().cpu()),
                        dim=0,
                    )
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
    for h in rotary_guard_handles:
        h.remove()
    layers[0] = layers[0].module

    if _is_llama3_model(model_name):
        _check_layer_replay_equivalence(
            model_name=model_name,
            model=model,
            layer=layers[0],
            calib_loader=calib_loader,
            inps=inps,
            attention_masks=cache["attention_mask"],
            position_ids=cache["position_ids"] if "opt" not in model_name else None,
            dev=dev,
            max_batches=2,
        )

    layers[0] = layers[0].cpu()

    if "opt" in model_name:
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.cpu()
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    else:
        model.model.embed_tokens = model.model.embed_tokens.cpu()
        model.model.norm = model.model.norm.cpu()

    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache["attention_mask"]
    position_ids = cache["position_ids"] if "opt" not in model_name else None

    decomposition_book = {}
    ecsvr_scales = []
    cov_dtype = torch.float64 if _use_high_precision_cov(model_name) else torch.float32

    print("Start CCSR: compression-consistent sequential reprofiling...")
    kept_gqa_kv_modules = 0
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        layer_book = {}

        # Stage 1/2 profiling under compressed-prefix inputs.
        def _profile_hook(module, input, _output):
            inp = input[0].detach()
            if inp.dim() == 2:
                inp = inp.unsqueeze(0)
            inp_cov = inp.to(dtype=cov_dtype)
            adds_sum = torch.sum(torch.matmul(inp_cov.transpose(1, 2), inp_cov), dim=0)
            if getattr(module, "raw_scaling_diag_matrix", None) is None:
                module.raw_scaling_diag_matrix = adds_sum
            else:
                module.raw_scaling_diag_matrix += adds_sum
            channel_max_abs = inp.abs().amax(dim=(0, 1)).float()
            module.channel_max_abs = torch.maximum(module.channel_max_abs, channel_max_abs)

        profile_handles = []
        for name in subset:
            subset[name].raw_scaling_diag_matrix = None
            subset[name].channel_max_abs = torch.zeros(
                subset[name].in_features, device=dev, dtype=torch.float32
            )
            profile_handles.append(subset[name].register_forward_hook(_profile_hook))

        _layer_forward_pass(
            model=model,
            model_name=model_name,
            layer=layer,
            inps=inps,
            attention_masks=attention_masks,
            position_ids=position_ids,
            dev=dev,
            outs=None,
            max_batches=None,
        )

        for h in profile_handles:
            h.remove()

        layer_outlier_ratio = stage1_outlier_ratio
        if stage1_layer_ratio_map is not None and i in stage1_layer_ratio_map:
            layer_outlier_ratio = float(stage1_layer_ratio_map[i])

        # Stage 1 + Stage 2 decomposition for current layer only.
        for name in subset:
            if _should_keep_gqa_kv_uncompressed(model_name, model.config, name):
                subset[name].raw_scaling_diag_matrix = None
                subset[name].channel_max_abs = None
                kept_gqa_kv_modules += 1
                continue

            module = subset[name]
            W = module.weight.data.to(dev)
            raw_scaling_diag_matrix = module.raw_scaling_diag_matrix
            scaling_diag_matrix = _safe_cholesky(raw_scaling_diag_matrix, dev)

            normal_idx, outlier_idx = _select_channel_partitions(
                scaling_diag_matrix,
                layer_outlier_ratio,
                criterion=stage1_outlier_criterion,
                channel_max_abs=module.channel_max_abs,
            )

            if normal_idx.numel() == scaling_diag_matrix.shape[0]:
                scaling_diag_matrix_normal = scaling_diag_matrix
            else:
                cov = torch.matmul(scaling_diag_matrix, scaling_diag_matrix.transpose(0, 1))
                cov_normal = cov.index_select(0, normal_idx).index_select(1, normal_idx)
                scaling_diag_matrix_normal = _safe_cholesky(cov_normal, dev)
                cov = cov_normal = None
                del cov, cov_normal

            compute_dtype = scaling_diag_matrix_normal.dtype
            W_normal = W.index_select(1, normal_idx).to(dtype=compute_dtype)
            W_scale = torch.matmul(W_normal, scaling_diag_matrix_normal)
            U, singular_values, VT = _safe_svd(W_scale, name=f"{i}:{name}")
            right_proj = _right_project_from_cholesky(VT, scaling_diag_matrix_normal)
            proj_matrix = singular_values.unsqueeze(1) * right_proj
            target_rank = _target_rank_for_module(
                model_name,
                model.config,
                name,
                W_normal.shape[0],
                W_normal.shape[1],
                ratio,
                singular_values.numel(),
            )

            if outlier_idx.numel() > 0:
                outlier_weight = W.index_select(1, outlier_idx).cpu()
            else:
                outlier_weight = torch.zeros((W.shape[0], 0), dtype=W.dtype)

            layer_book[name] = {
                "U": U.cpu(),
                "singular_values": singular_values.cpu(),
                "right_proj": right_proj.cpu(),
                "proj_matrix": proj_matrix.cpu(),
                "normal_idx": normal_idx.cpu(),
                "outlier_idx": outlier_idx.cpu(),
                "outlier_weight": outlier_weight,
                "target_rank": target_rank,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
                "bias": module.bias.data.cpu() if module.bias is not None else None,
                "selected_idx": torch.arange(target_rank, dtype=torch.long),
            }

            module.raw_scaling_diag_matrix = None
            module.channel_max_abs = None
            W = W_normal = W_scale = raw_scaling_diag_matrix = scaling_diag_matrix = None
            scaling_diag_matrix_normal = None
            U = singular_values = VT = right_proj = proj_matrix = None
            del W, W_normal, W_scale, raw_scaling_diag_matrix, scaling_diag_matrix, scaling_diag_matrix_normal
            del U, singular_values, VT, right_proj, proj_matrix
            torch.cuda.empty_cache()

        # Stage 3 stability selection under compressed-prefix inputs.
        if enable_stage3:
            stage3_runtime = {}
            for name in subset:
                if name not in layer_book:
                    continue
                info = layer_book[name]
                if info["singular_values"].numel() == 0:
                    info["selected_idx"] = torch.empty(0, dtype=torch.long)
                    continue
                stage3_runtime[name] = {
                    "normal_idx": info["normal_idx"].to(dev),
                    "proj_matrix": info["proj_matrix"].to(dev),
                    "energies": [],
                    "rank_size": info["singular_values"].numel(),
                }

            def _make_stage3_hook(module_name):
                def _hook(_module, input, _output):
                    inp = input[0].detach().float()
                    if inp.dim() == 2:
                        inp = inp.unsqueeze(0)
                    info = stage3_runtime[module_name]
                    normal_idx = info["normal_idx"]
                    if normal_idx.numel() != inp.shape[-1]:
                        inp = inp.index_select(-1, normal_idx)
                    inp = inp.to(dtype=info["proj_matrix"].dtype)

                    response_tensor = torch.matmul(inp, info["proj_matrix"].transpose(0, 1))
                    seq_energy = torch.sum(response_tensor * response_tensor, dim=1)
                    info["energies"].append(seq_energy.detach().to(device="cpu", dtype=torch.float64))

                return _hook

            stage3_handles = []
            for name in stage3_runtime:
                stage3_handles.append(subset[name].register_forward_hook(_make_stage3_hook(name)))

            _layer_forward_pass(
                model=model,
                model_name=model_name,
                layer=layer,
                inps=inps,
                attention_masks=attention_masks,
                position_ids=position_ids,
                dev=dev,
                outs=None,
                max_batches=stage3_max_batches,
            )

            for h in stage3_handles:
                h.remove()

            for name in stage3_runtime:
                info = stage3_runtime[name]
                if len(info["energies"]) == 0:
                    score = torch.full((info["rank_size"],), -1e10, device=dev, dtype=torch.float64)
                else:
                    all_energies = torch.cat(info["energies"], dim=0).to(dev)
                    mean_energy = all_energies.mean(dim=0)
                    var_energy = all_energies.var(dim=0, unbiased=False)
                    score = mean_energy - stage3_lambda * torch.sqrt(var_energy)
                target_rank = layer_book[name]["target_rank"]
                target_rank = min(target_rank, score.numel())
                selected_idx = torch.topk(score, k=target_rank, largest=True).indices
                layer_book[name]["selected_idx"] = selected_idx.cpu()

                info["normal_idx"] = info["proj_matrix"] = None
                info["energies"] = None
                torch.cuda.empty_cache()

        # Reconstruct current layer immediately.
        for name in subset:
            if name not in layer_book:
                continue
            module = subset[name]
            info = layer_book[name]

            selected_idx = info["selected_idx"]
            if selected_idx.numel() == 0 and info["normal_idx"].numel() > 0:
                selected_idx = torch.arange(min(1, info["singular_values"].numel()), dtype=torch.long)

            U_sel = info["U"][:, selected_idx].float()
            if enable_ecsvr:
                S_sel, gamma = _energy_conserving_recalibrate(
                    info["singular_values"],
                    selected_idx,
                    max_scale=ecsvr_max_scale,
                )
                ecsvr_scales.append(gamma)
            else:
                S_sel = info["singular_values"][selected_idx].float()

            right_proj_sel = info["right_proj"][selected_idx, :].float()
            sqrt_s = torch.sqrt(torch.clamp(S_sel, min=0))
            svd_u = U_sel * sqrt_s.unsqueeze(0)
            svd_v = right_proj_sel * sqrt_s.unsqueeze(1)

            new_linear = StableSVDLinear(
                in_features=info["in_features"],
                out_features=info["out_features"],
                rank=svd_v.shape[0],
                normal_indices=info["normal_idx"],
                outlier_indices=info["outlier_idx"],
                bias=info["has_bias"],
            ).to(dtype=module.weight.dtype, device=module.weight.device)

            if new_linear.has_low_rank:
                new_linear.u_proj.weight.data = svd_u.to(
                    dtype=module.weight.dtype, device=new_linear.u_proj.weight.device
                )
                new_linear.v_proj.weight.data = svd_v.to(
                    dtype=module.weight.dtype, device=new_linear.v_proj.weight.device
                )
                if info["has_bias"]:
                    new_linear.u_proj.bias.data = info["bias"].to(
                        dtype=module.weight.dtype, device=new_linear.u_proj.bias.device
                    )
            elif info["has_bias"] and getattr(new_linear, "bias", None) is not None:
                new_linear.bias.data = info["bias"].to(
                    dtype=module.weight.dtype, device=new_linear.bias.device
                )

            if new_linear.has_outlier:
                new_linear.outlier_proj.weight.data = info["outlier_weight"].to(
                    dtype=module.weight.dtype,
                    device=new_linear.outlier_proj.weight.device,
                )

            parent_module, leaf_name = _get_parent_module(layer, name)
            setattr(parent_module, leaf_name, new_linear)

            U_sel = S_sel = right_proj_sel = sqrt_s = svd_u = svd_v = None
            del U_sel, S_sel, right_proj_sel, sqrt_s, svd_u, svd_v
            torch.cuda.empty_cache()

        decomposition_book[i] = layer_book

        # Propagate compressed outputs to next layer.
        _layer_forward_pass(
            model=model,
            model_name=model_name,
            layer=layer,
            inps=inps,
            attention_masks=attention_masks,
            position_ids=position_ids,
            dev=dev,
            outs=outs,
            max_batches=None,
        )

        layers[i] = layer.cpu()
        inps, outs = outs, inps
        torch.cuda.empty_cache()

    if kept_gqa_kv_modules > 0:
        print(f"LLaMA-3 GQA K/V protection: kept {kept_gqa_kv_modules} k_proj/v_proj modules uncompressed.")

    if enable_ecsvr and len(ecsvr_scales) > 0:
        gamma_tensor = torch.tensor(ecsvr_scales, dtype=torch.float32)
        print(
            "EC-SVR scale stats: "
            f"mean={gamma_tensor.mean().item():.4f}, "
            f"min={gamma_tensor.min().item():.4f}, "
            f"max={gamma_tensor.max().item():.4f}"
        )

    if enable_sam:
        _apply_sam(
            model_name=model_name,
            model=model,
            decomposition_book=decomposition_book,
            calib_loader=calib_loader,
            dev=dev,
            sam_damp=sam_damp,
            sam_max_batches=sam_max_batches,
            enable_ecsvr=enable_ecsvr,
            ecsvr_max_scale=ecsvr_max_scale,
        )

    _move_llm_shared_modules(model_name, model, "cpu")
    model.config.use_cache = use_cache


@torch.no_grad()
def _collect_stage3_scores(model_name, model, decomposition_book, calib_loader, stability_lambda, dev, stage3_max_batches=None):
    if calib_loader is None:
        print("Warning: Stage 3 is enabled, but no calibration loader is provided. Falling back to Stage 2 truncation.")
        return

    model = model.to(dev)
    model.eval()
    if "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print("Start Stage 3: Cross-sample stability scoring...")
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        runtime = {}
        handles = []

        for name in subset:
            if name not in decomposition_book[i]:
                continue
            info = decomposition_book[i][name]
            if info["singular_values"].numel() == 0:
                info["selected_idx"] = torch.empty(0, dtype=torch.long)
                continue
            runtime[name] = {
                "normal_idx": info["normal_idx"].to(dev),
                "proj_matrix": info["proj_matrix"].to(dev),
                "energies": [],
                "rank_size": info["singular_values"].numel(),
            }

        if len(runtime) == 0:
            continue

        def _make_hook(name):
            def _hook(_module, input, _output):
                inp = input[0].detach().float()
                if inp.dim() == 2:
                    inp = inp.unsqueeze(0)
                info = runtime[name]
                normal_idx = info["normal_idx"]
                if normal_idx.numel() != inp.shape[-1]:
                    inp = inp.index_select(-1, normal_idx)
                inp = inp.to(dtype=info["proj_matrix"].dtype)

                # R_tensor: [batch, seq_len, rank]
                response_tensor = torch.matmul(inp, info["proj_matrix"].transpose(0, 1))
                # seq_energy: [batch, rank]
                seq_energy = torch.sum(response_tensor * response_tensor, dim=1)
                info["energies"].append(seq_energy.detach().to(device="cpu", dtype=torch.float64))
            return _hook

        for name in runtime:
            handles.append(subset[name].register_forward_hook(_make_hook(name)))

        for batch in _iter_batches(calib_loader, stage3_max_batches):
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)

        for handle in handles:
            handle.remove()

        for name in runtime:
            info = runtime[name]
            if len(info["energies"]) == 0:
                score = torch.full((info["rank_size"],), -1e10, device=dev, dtype=torch.float64)
            else:
                all_energies = torch.cat(info["energies"], dim=0).to(dev)
                mean_energy = all_energies.mean(dim=0)
                var_energy = all_energies.var(dim=0, unbiased=False)
                score = mean_energy - stability_lambda * torch.sqrt(var_energy)
            target_rank = decomposition_book[i][name]["target_rank"]
            target_rank = min(target_rank, score.numel())
            selected_idx = torch.topk(score, k=target_rank, largest=True).indices
            decomposition_book[i][name]["selected_idx"] = selected_idx.cpu()

            info["normal_idx"] = info["proj_matrix"] = None
            info["energies"] = None
            torch.cuda.empty_cache()

    model = model.cpu()




@torch.no_grad()
def whitening(
    model_name,
    model,
    profiling_mat,
    ratio,
    dev,
    calib_loader=None,
    stage1_outlier_ratio=0.0,
    stage1_outlier_criterion="infinity_norm",
    stage1_layer_ratio_map=None,
    outlier_channel_stats=None,
    enable_stage3=False,
    stage3_lambda=1.0,
    stage3_max_batches=None,
    enable_ecsvr=False,
    ecsvr_max_scale=None,
    enable_sam=False,
    sam_damp=1e-4,
    sam_max_batches=None,
):
    model.eval()
    gqa_multiplier = _gqa_kv_rank_multiplier(model_name, model.config)
    if gqa_multiplier > 1:
        kv_effective_ratio = min(1.0, ratio * gqa_multiplier)
        print(
            "LLaMA-3 GQA rank adaptation: "
            f"base_ratio={ratio:.6f}, "
            f"k_proj/v_proj effective_ratio={kv_effective_ratio:.6f}, "
            f"multiplier={gqa_multiplier}, cap=1.0"
        )
    if "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print("Start Stage 1 + Stage 2 decomposition...")
    decomposition_book = {}
    kept_gqa_kv_modules = 0

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        layer_book = {}
        for name in subset:
            if _should_keep_gqa_kv_uncompressed(model_name, model.config, name):
                kept_gqa_kv_modules += 1
                continue

            module = subset[name]
            W = module.weight.data.to(dev)
            scaling_diag_matrix = profiling_mat[i][name].to(dev)
            layer_outlier_ratio = stage1_outlier_ratio
            if stage1_layer_ratio_map is not None and i in stage1_layer_ratio_map:
                layer_outlier_ratio = float(stage1_layer_ratio_map[i])
            channel_max_abs = None
            if outlier_channel_stats is not None and i in outlier_channel_stats and name in outlier_channel_stats[i]:
                channel_max_abs = outlier_channel_stats[i][name]
            normal_idx, outlier_idx = _select_channel_partitions(
                scaling_diag_matrix,
                layer_outlier_ratio,
                criterion=stage1_outlier_criterion,
                channel_max_abs=channel_max_abs,
            )

            if normal_idx.numel() == scaling_diag_matrix.shape[0]:
                scaling_diag_matrix_normal = scaling_diag_matrix
            else:
                cov = torch.matmul(scaling_diag_matrix, scaling_diag_matrix.transpose(0, 1))
                cov_normal = cov.index_select(0, normal_idx).index_select(1, normal_idx)
                scaling_diag_matrix_normal = _safe_cholesky(cov_normal, dev)
                cov = cov_normal = None
                del cov, cov_normal

            compute_dtype = scaling_diag_matrix_normal.dtype
            W_normal = W.index_select(1, normal_idx).to(dtype=compute_dtype)
            W_scale = torch.matmul(W_normal, scaling_diag_matrix_normal)
            U, singular_values, VT = _safe_svd(W_scale, name=f"{i}:{name}")
            right_proj = _right_project_from_cholesky(VT, scaling_diag_matrix_normal)
            proj_matrix = singular_values.unsqueeze(1) * right_proj
            target_rank = _target_rank_for_module(
                model_name,
                model.config,
                name,
                W_normal.shape[0],
                W_normal.shape[1],
                ratio,
                singular_values.numel(),
            )

            if outlier_idx.numel() > 0:
                outlier_weight = W.index_select(1, outlier_idx).cpu()
            else:
                outlier_weight = torch.zeros((W.shape[0], 0), dtype=W.dtype)

            layer_book[name] = {
                "U": U.cpu(),
                "singular_values": singular_values.cpu(),
                "right_proj": right_proj.cpu(),
                "proj_matrix": proj_matrix.cpu(),
                "normal_idx": normal_idx.cpu(),
                "outlier_idx": outlier_idx.cpu(),
                "outlier_weight": outlier_weight,
                "target_rank": target_rank,
                "in_features": module.in_features,
                "out_features": module.out_features,
                "has_bias": module.bias is not None,
                "bias": module.bias.data.cpu() if module.bias is not None else None,
            }

            W = W_normal = W_scale = scaling_diag_matrix = scaling_diag_matrix_normal = None
            U = singular_values = VT = right_proj = proj_matrix = None
            del W, W_normal, W_scale, scaling_diag_matrix, scaling_diag_matrix_normal, U, singular_values, VT, right_proj, proj_matrix
            torch.cuda.empty_cache()
        decomposition_book[i] = layer_book

    if enable_stage3:
        _collect_stage3_scores(
            model_name=model_name,
            model=model,
            decomposition_book=decomposition_book,
            calib_loader=calib_loader,
            stability_lambda=stage3_lambda,
            dev=dev,
            stage3_max_batches=stage3_max_batches,
        )
    else:
        for i in decomposition_book:
            for name in decomposition_book[i]:
                target_rank = decomposition_book[i][name]["target_rank"]
                decomposition_book[i][name]["selected_idx"] = torch.arange(target_rank, dtype=torch.long)

    print("Start reconstruction with selected stable directions...")
    ecsvr_scales = []
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            if name not in decomposition_book[i]:
                continue
            module = subset[name]
            info = decomposition_book[i][name]

            selected_idx = info["selected_idx"]
            if selected_idx.numel() == 0 and info["normal_idx"].numel() > 0:
                selected_idx = torch.arange(min(1, info["singular_values"].numel()), dtype=torch.long)

            U_sel = info["U"][:, selected_idx].float()
            if enable_ecsvr:
                S_sel, gamma = _energy_conserving_recalibrate(
                    info["singular_values"],
                    selected_idx,
                    max_scale=ecsvr_max_scale,
                )
                ecsvr_scales.append(gamma)
            else:
                S_sel = info["singular_values"][selected_idx].float()
            right_proj_sel = info["right_proj"][selected_idx, :].float()

            sqrt_s = torch.sqrt(torch.clamp(S_sel, min=0))
            svd_u = U_sel * sqrt_s.unsqueeze(0)
            svd_v = right_proj_sel * sqrt_s.unsqueeze(1)

            new_linear = StableSVDLinear(
                in_features=info["in_features"],
                out_features=info["out_features"],
                rank=svd_v.shape[0],
                normal_indices=info["normal_idx"],
                outlier_indices=info["outlier_idx"],
                bias=info["has_bias"],
            ).to(dtype=module.weight.dtype, device=module.weight.device)

            if new_linear.has_low_rank:
                new_linear.u_proj.weight.data = svd_u.to(dtype=module.weight.dtype, device=new_linear.u_proj.weight.device)
                new_linear.v_proj.weight.data = svd_v.to(dtype=module.weight.dtype, device=new_linear.v_proj.weight.device)
                if info["has_bias"]:
                    new_linear.u_proj.bias.data = info["bias"].to(dtype=module.weight.dtype, device=new_linear.u_proj.bias.device)
            elif info["has_bias"] and getattr(new_linear, "bias", None) is not None:
                new_linear.bias.data = info["bias"].to(dtype=module.weight.dtype, device=new_linear.bias.device)

            if new_linear.has_outlier:
                new_linear.outlier_proj.weight.data = info["outlier_weight"].to(
                    dtype=module.weight.dtype,
                    device=new_linear.outlier_proj.weight.device,
                )

            parent_module, leaf_name = _get_parent_module(layer, name)
            setattr(parent_module, leaf_name, new_linear)

            U_sel = S_sel = right_proj_sel = sqrt_s = svd_u = svd_v = None
            del U_sel, S_sel, right_proj_sel, sqrt_s, svd_u, svd_v
            torch.cuda.empty_cache()

    if kept_gqa_kv_modules > 0:
        print(f"LLaMA-3 GQA K/V protection: kept {kept_gqa_kv_modules} k_proj/v_proj modules uncompressed.")

    if enable_ecsvr and len(ecsvr_scales) > 0:
        gamma_tensor = torch.tensor(ecsvr_scales, dtype=torch.float32)
        print(
            "EC-SVR scale stats: "
            f"mean={gamma_tensor.mean().item():.4f}, "
            f"min={gamma_tensor.min().item():.4f}, "
            f"max={gamma_tensor.max().item():.4f}"
        )
    if enable_sam:
        _apply_sam(
            model_name=model_name,
            model=model,
            decomposition_book=decomposition_book,
            calib_loader=calib_loader,
            dev=dev,
            sam_damp=sam_damp,
            sam_max_batches=sam_max_batches,
            enable_ecsvr=enable_ecsvr,
            ecsvr_max_scale=ecsvr_max_scale,
        )


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
    _enforce_model_runtime_compat(model_name, model)
    use_cache = model.config.use_cache
    model.config.use_cache = False
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    _move_llm_shared_modules(model_name, model, dev)
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    rotary_guard_handles = _register_rotary_runtime_device_guard(model_name, model)
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            attention_mask = kwargs.get("attention_mask", None)
            position_ids = kwargs.get("position_ids", None)
            cache_position = kwargs.get("cache_position", None)
            if position_ids is None and cache_position is not None:
                position_ids = cache_position.unsqueeze(0)
            if attention_mask is not None:
                if cache['attention_mask'] is None:
                    cache['attention_mask'] = attention_mask
                else:
                    cache['attention_mask'] = torch.cat((cache['attention_mask'], attention_mask), dim=0)
            if "opt" not in model_name and position_ids is not None:
                if cache['position_ids'] is None:
                    cache['position_ids'] = position_ids
                else:
                    cache['position_ids'] = torch.cat((cache['position_ids'], position_ids), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
    for h in rotary_guard_handles:
        h.remove()
    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.embed_tokens = model.model.embed_tokens.cpu()
    model.model.norm = model.model.norm.cpu()
    torch.cuda.empty_cache()
    outs = torch.zeros_like(inps)
    attention_masks = cache['attention_mask']
    if "opt" not in model_name:
        position_ids = cache['position_ids']
    for i in tqdm(range(len(layers))):
        layer = layers[i].to(dev)
        subset = find_layers(layer)
        gpts = {}
        if "llama" in model_name or "vicuna" in model_name:
            svd_attn = SVD_LlamaAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_LlamaMLP(hidden_size=layer.hidden_size, intermediate_size=model.config.intermediate_size, hidden_act=model.config.hidden_act, ratio=ratio)
        elif "mistral" in model_name:
            svd_attn = SVD_MistralAttention(config=model.config, ratio=ratio)
            svd_mlp = SVD_MistralMLP(config=model.config, ratio=ratio)
        elif 'opt' in model_name:
            svd_decoder = SVDOPTDecoderLayer(model.config, ratio=ratio)
        for name in subset:
            if profiling_mat is not None:
                scaling_diag_matrix = profiling_mat[i][name].to(dev)
            else: 
                scaling_diag_matrix = None
            gpts[name] = local_update(subset[name], scaling_diag_matrix = scaling_diag_matrix, ratio=ratio, name=name, direct_update=direct_update)
        
        def add_batch(name):
            def tmp(_, inp, out):
                gpts[name].add_batch_update_u(inp[0].data, out.data)
            return tmp
        handles = []
        for name in gpts:
            handles.append(subset[name].register_forward_hook(add_batch(name)))
        _layer_forward_pass(
            model=model,
            model_name=model_name,
            layer=layer,
            inps=inps,
            attention_masks=attention_masks,
            position_ids=position_ids if "opt" not in model_name else None,
            dev=dev,
            outs=outs,
            max_batches=None,
        )
        for h in handles:
            h.remove()
        for name in gpts:
            svd_u, svd_v = gpts[name].fasterprune()
            svd_u, svd_v = svd_u.to(dtype), svd_v.to(dtype)
            if 'opt' in model_name:
                if "q_proj" in name:
                    svd_decoder.self_attn.q_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.q_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.q_u_proj.bias.data = layer.self_attn.q_proj.bias.data  # the linear layer in OPT has bias, which is different from LLaMA and Mistral
                elif "k_proj" in name:
                    svd_decoder.self_attn.k_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.k_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.k_u_proj.bias.data = layer.self_attn.k_proj.bias.data
                elif "v_proj" in name:
                    svd_decoder.self_attn.v_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.v_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.v_u_proj.bias.data = layer.self_attn.v_proj.bias.data
                elif "out_proj" in name:
                    svd_decoder.self_attn.out_u_proj.weight.data = svd_u
                    svd_decoder.self_attn.out_v_proj.weight.data = svd_v
                    svd_decoder.self_attn.out_u_proj.bias.data = layer.self_attn.out_proj.bias.data
                elif "fc1" in name:
                    svd_decoder.fc1_u_proj.weight.data = svd_u
                    svd_decoder.fc1_v_proj.weight.data = svd_v
                    svd_decoder.fc1_u_proj.bias.data = layer.fc1.bias.data
                elif "fc2" in name:
                    svd_decoder.fc2_u_proj.weight.data = svd_u
                    svd_decoder.fc2_v_proj.weight.data = svd_v
                    svd_decoder.fc2_u_proj.bias.data = layer.fc2.bias.data
                    svd_decoder.self_attn_layer_norm = layer.self_attn_layer_norm
                    svd_decoder.final_layer_norm = layer.final_layer_norm
                    layers[i] = svd_decoder
            else:
                if "q_proj" in name:
                    svd_attn.q_u_proj.weight.data = svd_u
                    svd_attn.q_v_proj.weight.data = svd_v
                elif "k_proj" in name:
                    svd_attn.k_u_proj.weight.data = svd_u
                    svd_attn.k_v_proj.weight.data = svd_v
                elif "v_proj" in name:
                    svd_attn.v_u_proj.weight.data = svd_u
                    svd_attn.v_v_proj.weight.data = svd_v
                elif "o_proj" in name:
                    svd_attn.o_u_proj.weight.data = svd_u
                    svd_attn.o_v_proj.weight.data = svd_v
                    layer.self_attn =  svd_attn
                elif "gate_proj" in name:
                    svd_mlp.gate_u_proj.weight.data = svd_u
                    svd_mlp.gate_v_proj.weight.data = svd_v
                elif "down_proj" in name:
                    svd_mlp.down_u_proj.weight.data = svd_u
                    svd_mlp.down_v_proj.weight.data = svd_v
                elif "up_proj" in name:
                    svd_mlp.up_u_proj.weight.data = svd_u
                    svd_mlp.up_v_proj.weight.data = svd_v
                    layer.mlp = svd_mlp
        layer = layer.to(dev)
        _layer_forward_pass(
            model=model,
            model_name=model_name,
            layer=layer,
            inps=inps,
            attention_masks=attention_masks,
            position_ids=position_ids if "opt" not in model_name else None,
            dev=dev,
            outs=outs,
            max_batches=None,
        )
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
    _move_llm_shared_modules(model_name, model, "cpu")
    model.config.use_cache = use_cache


class local_update:
    def __init__(self, layer, scaling_diag_matrix, ratio, name, direct_update=False):
        self.layer = layer
        self.name = name
        self.dev = self.layer.weight.device
        # W = layer.weight.data.clone()
        W = layer.weight.data.clone()
        self.rows = W.shape[0]
        self.columns = W.shape[1]
        if direct_update:
            self.U, self.S, self.VT = _safe_svd(W.data, name=self.name)
        else: 
            scaling_diag_matrix = scaling_diag_matrix.to(device=self.dev, dtype=W.dtype)
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = _safe_svd(W_scale, name=self.name)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = _right_project_from_cholesky(
                self.VT[:num_s_after_trunc, :].cuda(),
                scaling_diag_matrix.cuda(),
            )
        self.truc_sigma = torch.diag(self.truc_s)
        self.new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v[:num_s_after_trunc, :]))
        # intialize H for close form solution
        self.updated_err = self.error = 0

    def add_batch_update_u(self, inp, out):
        inps = inp.view(inp.shape[0] * inp.shape[1], inp.shape[2])
        outs = out.view(out.shape[0] * out.shape[1], out.shape[2])
        new_w = torch.matmul(self.truc_u, torch.matmul(self.truc_sigma, self.truc_v))
        new_output = inps.matmul(new_w.t())
        self.error = torch.sqrt(torch.sum((outs - new_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"truncted error: {self.error}")
        x =  torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma)
        self.updated_uT = torch.linalg.lstsq(x,outs).solution
        updated_output = torch.matmul(torch.matmul(torch.matmul(inps, self.truc_v.T), self.truc_sigma), self.updated_uT)
        self.updated_error = torch.sqrt(torch.sum((outs - updated_output)**2)).item() / torch.norm(outs, p='fro').item()
        # print(f"updated error: {self.updated_error}")
        inps = outs = new_output = updated_output = x = new_w = None
        del inps, outs, new_output, updated_output, x, new_w
        torch.cuda.empty_cache()
        # print(f"Finish {self.name}"
    
    def fasterprune(self):
        sqrtSigma = torch.sqrt(self.truc_sigma)
        self.appendU = self.updated_uT.t().matmul(sqrtSigma)
        self.appendV = sqrtSigma.matmul(self.truc_v)
        return self.appendU, self.appendV


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--model', type=str, default='jeffwan/llama-7b-hf', help='LLaMA model to load, pass `jeffwan/llama-7b-hf`')
    parser.add_argument('--model_path', type=str, default=None, help='local compressed model path or whitening information path')
    parser.add_argument('--ratio', type=float, default=0.2, help='Target compression ratio,(0,1), default=0.2, means only keeping about 20% of the params.')
    parser.add_argument('--run_low_resource', action='store_true', help='whether to run whitening in low resource, exp, compress LLaMA-7B below 15G gpu')
    parser.add_argument('--enable_ccsr', action='store_true', help='Enable CCSR: compression-consistent sequential reprofiling')
    parser.add_argument('--dataset', type=str, default='wikitext2',help='Where to extract calibration data from [wikitext2, ptb, c4]')
    parser.add_argument('--whitening_nsamples', type=int, default=256, help='Number of calibration data samples for whitening.')
    parser.add_argument('--updating_nsamples', type=int, default=16, help='Number of calibration data samples for udpating.')
    parser.add_argument('--save_path', type=str, default=None, help='the path to save the compressed model checkpoints.`')
    parser.add_argument('--profiling_mat_path', type=str, default=None, help='Local path to load the profiling matrices`')
    parser.add_argument('--seed',type=int, default=0, help='Seed for sampling the calibration data')
    parser.add_argument('--DEV', type=str, default="cuda", help='device')
    parser.add_argument('--model_seq_len', type=int, default=2048, help='the default sequence length of the LLM')
    parser.add_argument('--eval_batch_size', type=int, default=4, help='inference bactch size')
    parser.add_argument('--gen_seq_len', type=int, default=1024, help='generated sequence len for efficiency evaluation')
    parser.add_argument('--step', type=int, default=4, help='the step to run the compression')
    parser.add_argument('--lora', type=str, default=None, help='the lora updated weight path to run the accuracy evaluation')
    parser.add_argument('--stage1_outlier_ratio', type=float, default=0.0, help='Stage 1: fraction of input channels to strip as dense outliers')
    parser.add_argument('--stage1_outlier_criterion', type=str, default='infinity_norm', choices=['infinity_norm', 'energy'], help='Stage 1: outlier channel criterion')
    parser.add_argument('--enable_laoa', action='store_true', help='Enable LAOA: layer-wise adaptive outlier allocation')
    parser.add_argument('--laoa_min_ratio', type=float, default=0.005, help='LAOA: minimum layer outlier ratio')
    parser.add_argument('--laoa_max_ratio', type=float, default=0.06, help='LAOA: maximum layer outlier ratio')
    parser.add_argument('--laoa_temperature', type=float, default=None, help='LAOA: softmax temperature; if not set, use mean(CV)')
    parser.add_argument('--enable_stage3', action='store_true', help='Stage 3: enable cross-sample stability selection')
    parser.add_argument('--stage3_lambda', type=float, default=1.0, help='Stage 3: variance penalty coefficient in stability score')
    parser.add_argument('--stage3_max_batches', type=int, default=None, help='Stage 3: optionally limit number of calibration batches for stability scoring')
    parser.add_argument('--enable_ecsvr', action='store_true', help='Enable EC-SVR: energy-conserving singular value recalibration after truncation')
    parser.add_argument('--ecsvr_max_scale', type=float, default=None, help='Optional cap for EC-SVR gamma to avoid over-amplification (e.g., 1.5)')
    parser.add_argument('--enable_sam', action='store_true', help='Enable SAM: refit up-projection by least squares with fixed subspace projection')
    parser.add_argument('--sam_damp', type=float, default=1e-4, help='SAM: damping coefficient for normal equation regularization')
    parser.add_argument('--sam_max_batches', type=int, default=None, help='SAM: optionally limit calibration batches for least-squares fitting')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    if args.step == 1:
        # model, tokenizer = get_model_from_huggingface(model_id=args.model)
        
        model_load_dtype = _select_model_load_dtype(args.model)
        print(f"Model load dtype: {model_load_dtype}")
        model_load_kwargs = _select_model_load_kwargs(args.model)
        if len(model_load_kwargs) > 0:
            print(f"Model load extra kwargs: {model_load_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=model_load_dtype,
            **model_load_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.seqlen = args.model_seq_len
        model = model.eval()

        if args.enable_ccsr and args.profiling_mat_path is not None:
            print("Warning: CCSR ignores --profiling_mat_path and reprofiles each layer sequentially.")

        cali_white_data = None
        need_calibration_data = (
            args.enable_ccsr
            or args.profiling_mat_path is None
            or args.enable_stage3
            or args.enable_sam
            or args.enable_laoa
            or (args.stage1_outlier_ratio > 0 and args.stage1_outlier_criterion == "infinity_norm")
        )
        if need_calibration_data:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)

        stage1_layer_ratio_map = None
        outlier_channel_stats = None

        if args.enable_ccsr:
            if args.enable_laoa:
                if args.stage1_outlier_ratio <= 0:
                    print("Warning: LAOA is enabled but stage1_outlier_ratio <= 0. LAOA is ignored.")
                elif args.stage1_outlier_criterion != "infinity_norm":
                    print("Warning: LAOA requires stage1_outlier_criterion=infinity_norm. LAOA is ignored.")
                else:
                    print("CCSR+LAOA: collecting one-shot channel max stats for layer-ratio allocation.")
                    if args.run_low_resource:
                        _, outlier_channel_stats = profle_svdllm_low_resource(
                            args.model, model, cali_white_data, args.DEV, return_outlier_stats=True
                        )
                    else:
                        _, outlier_channel_stats = profle_svdllm(
                            args.model, model, cali_white_data, args.DEV, return_outlier_stats=True
                        )
                    stage1_layer_ratio_map = _compute_laoa_layer_ratios(
                        outlier_channel_stats=outlier_channel_stats,
                        global_ratio=args.stage1_outlier_ratio,
                        min_ratio=args.laoa_min_ratio,
                        max_ratio=args.laoa_max_ratio,
                        temperature=args.laoa_temperature,
                    )

            whitening_sequential(
                model_name=args.model,
                model=model,
                ratio=args.ratio,
                dev=args.DEV,
                calib_loader=cali_white_data,
                stage1_outlier_ratio=args.stage1_outlier_ratio,
                stage1_outlier_criterion=args.stage1_outlier_criterion,
                stage1_layer_ratio_map=stage1_layer_ratio_map,
                enable_stage3=args.enable_stage3,
                stage3_lambda=args.stage3_lambda,
                stage3_max_batches=args.stage3_max_batches,
                enable_ecsvr=args.enable_ecsvr,
                ecsvr_max_scale=args.ecsvr_max_scale,
                enable_sam=args.enable_sam,
                sam_damp=args.sam_damp,
                sam_max_batches=args.sam_max_batches,
            )
        else:
            need_outlier_stats = args.stage1_outlier_ratio > 0 and args.stage1_outlier_criterion == "infinity_norm"
            if args.profiling_mat_path is None:
                if args.run_low_resource:
                    if need_outlier_stats:
                        profiling_mat, outlier_channel_stats = profle_svdllm_low_resource(
                            args.model, model, cali_white_data, args.DEV, return_outlier_stats=True
                        )
                    else:
                        profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
                else:
                    if need_outlier_stats:
                        profiling_mat, outlier_channel_stats = profle_svdllm(
                            args.model, model, cali_white_data, args.DEV, return_outlier_stats=True
                        )
                    else:
                        profiling_mat = profle_svdllm(args.model, model, cali_white_data, args.DEV)
                if args.save_path is not None:
                    torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
            else:
                profiling_mat = torch.load(args.profiling_mat_path)
                if need_outlier_stats:
                    if args.run_low_resource:
                        _, outlier_channel_stats = profle_svdllm_low_resource(
                            args.model, model, cali_white_data, args.DEV, return_outlier_stats=True
                        )
                    else:
                        _, outlier_channel_stats = profle_svdllm(
                            args.model, model, cali_white_data, args.DEV, return_outlier_stats=True
                        )

            if args.enable_laoa:
                if args.stage1_outlier_ratio <= 0:
                    print("Warning: LAOA is enabled but stage1_outlier_ratio <= 0. LAOA is ignored.")
                elif args.stage1_outlier_criterion != "infinity_norm":
                    print("Warning: LAOA requires stage1_outlier_criterion=infinity_norm. LAOA is ignored.")
                elif outlier_channel_stats is None:
                    print("Warning: LAOA needs channel max statistics, but none are available. LAOA is ignored.")
                else:
                    stage1_layer_ratio_map = _compute_laoa_layer_ratios(
                        outlier_channel_stats=outlier_channel_stats,
                        global_ratio=args.stage1_outlier_ratio,
                        min_ratio=args.laoa_min_ratio,
                        max_ratio=args.laoa_max_ratio,
                        temperature=args.laoa_temperature,
                    )
            whitening(
                args.model,
                model,
                profiling_mat,
                args.ratio,
                args.DEV,
                calib_loader=cali_white_data,
                stage1_outlier_ratio=args.stage1_outlier_ratio,
                stage1_outlier_criterion=args.stage1_outlier_criterion,
                stage1_layer_ratio_map=stage1_layer_ratio_map,
                outlier_channel_stats=outlier_channel_stats,
                enable_stage3=args.enable_stage3,
                stage3_lambda=args.stage3_lambda,
                stage3_max_batches=args.stage3_max_batches,
                enable_ecsvr=args.enable_ecsvr,
                ecsvr_max_scale=args.ecsvr_max_scale,
                enable_sam=args.enable_sam,
                sam_damp=args.sam_damp,
                sam_max_batches=args.sam_max_batches,
            )
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        # model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model_load_dtype = _select_model_load_dtype(args.model)
        print(f"Model load dtype: {model_load_dtype}")
        model_load_kwargs = _select_model_load_kwargs(args.model)
        if len(model_load_kwargs) > 0:
            print(f"Model load extra kwargs: {model_load_kwargs}")
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=model_load_dtype,
            **model_load_kwargs,
        )
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model.seqlen = args.model_seq_len
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        model = model.eval()
        model = model.float()  # need to set to float
        if args.profiling_mat_path is None:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
            if args.run_low_resource:
                profiling_mat = profle_svdllm_low_resource(args.model, model, cali_white_data, args.DEV)
            else:
                profiling_mat = profle_svdllm(args.model, model, cali_white_data, args.DEV)
            if args.save_path is not None:
                torch.save(profiling_mat, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") + '_profiling_'+ args.dataset + '_' + str(args.whitening_nsamples)  + '_' + str(args.seed)+ '.pt')
        else:
            profiling_mat = torch.load(args.profiling_mat_path)
        whitening_local_update(args.model, model, dataloader, profiling_mat, args.ratio, args.DEV)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_then_update_' + str(args.ratio) + '.pt')  # fp32
    elif args.step == 3:
        model, tokenizer = get_model_from_huggingface(args.model)
        model = model.eval()
        model = model.float()
        dataloader, _ = get_loaders(args.dataset, nsamples=args.updating_nsamples, seed=args.seed, tokenizer=tokenizer, seqlen=args.model_seq_len)
        whitening_local_update(model_name=args.model, model=model, dataloader=dataloader, profiling_mat=None, ratio=args.ratio, dev=args.DEV, direct_update=True)
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_update_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step >= 4:
        print(f"evaluating {args.model_path}...")
        if args.model_path == "original":
            model, tokenizer = get_model_from_huggingface(args.model)
        else:
            model, tokenizer = get_model_from_local(args.model_path)
            if args.lora is not None:
                from utils.peft import PeftModel
                model = PeftModel.from_pretrained(
                    model,
                    args.lora,
                    torch_dtype=torch.float16,
                )
                model = model.merge_and_unload()
                torch.save({'model': model, 'tokenizer': tokenizer}, args.lora + '/merge.pt')
        model.eval()
        model = model.float()
        model = model.to(args.DEV)
        if args.step == 4:
            ppl_eval(model, tokenizer, datasets=['wikitext2'], model_seq_len=args.model_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
        elif args.step == 5:
            eff_eval(model, tokenizer, generated_len=args.gen_seq_len, batch_size=args.eval_batch_size, device=args.DEV)
