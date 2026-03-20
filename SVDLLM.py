#coding:utf8
import os
import sys
import argparse
import itertools
import torch.jit
from tqdm import tqdm
import torch
import torch.nn as nn

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



@torch.no_grad()
def profle_svdllm(name, model, calib_loader, dev, return_outlier_stats=False):
    if "llama" in name or "mistral" in name or "vicuna" in name:
        layers = model.model.layers
    elif "opt" in name:
        layers = model.model.decoder.layers
    model = model.to(dev)
    print("Start obtaining the whitening matrix...")
    def hook(module, input, output):
        inp = input[0].detach().float()
        if inp.dim() == 2:   # for opt
            inp = inp.unsqueeze(0)
        adds = torch.matmul(inp.transpose(1,2), inp)
        adds_sum = torch.sum(adds, dim=0)
        module.raw_scaling_diag_matrix += adds_sum
        channel_max_abs = inp.abs().amax(dim=(0, 1))
        module.channel_max_abs = torch.maximum(module.channel_max_abs, channel_max_abs)
        del inp, adds, adds_sum
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
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
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
    if "opt" in model_name:
        layers = model.model.decoder.layers
        model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
        model.model.decoder.final_layer_norm = model.model.decoder.final_layer_norm.to(dev)
        model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    else:
        layers = model.model.layers
        model.model.embed_tokens = model.model.embed_tokens.to(dev)
        model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(calib_loader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp.cpu()
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask'].cpu()
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids'].cpu()
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask'].cpu()), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids'].cpu()), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in calib_loader:
        try:
            batch = {k: v.to(dev) for k, v in batch.items()}
            model(**batch)
        except ValueError:
            pass
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
    for i in tqdm(range(len(layers))):
        layer_profile = {}
        layer_outlier = {}
        layer = layers[i].to(dev)
        subset = find_layers(layer)        
        def hook(module, input, output):
            inp = input[0].detach().float()
            if inp.dim() == 2:  # for opt
                inp = inp.unsqueeze(0)
            adds = torch.matmul(inp.transpose(1,2), inp)
            adds_sum = torch.sum(adds, dim=0)
            module.scaling_diag_matrix += adds_sum
            channel_max_abs = inp.abs().amax(dim=(0, 1))
            module.channel_max_abs = torch.maximum(module.channel_max_abs, channel_max_abs)
            del inp, adds, adds_sum, output
            torch.cuda.empty_cache()
        handles = []
        for name in subset:
            subset[name].scaling_diag_matrix = 0
            subset[name].channel_max_abs = torch.zeros(subset[name].in_features, device=dev, dtype=torch.float32)
            handles.append(subset[name].register_forward_hook(hook))
        for j in range(inps.shape[0]):
            if "opt" not in model_name:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev), position_ids=position_ids[j].unsqueeze(0).to(dev))[0]
            else:
                outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_masks[j].unsqueeze(0).to(dev))[0]
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
            try:
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
            except Exception as e:
                print("Warning: eigen scaling_diag_matrix is not positive!")
                eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
                raw_scaling_diag_matrix += (- eigenvalues[0] + 1e-6) * torch.eye(raw_scaling_diag_matrix.shape[0]).to(dev)
                scaling_diag_matrix = torch.linalg.cholesky(raw_scaling_diag_matrix)
                eigenvalues = None
                del eigenvalues
            layer_profile[name] = scaling_diag_matrix.cpu()
            scaling_diag_matrix = raw_scaling_diag_matrix = subset[name].raw_scaling_diag_matrix = None
            del scaling_diag_matrix, raw_scaling_diag_matrix, subset[name].raw_scaling_diag_matrix
            torch.cuda.empty_cache()
        layers[i] = layer.cpu()
        profiling_mat[i] = layer_profile
        outlier_stats[i] = layer_outlier
        inps = outs
        torch.cuda.empty_cache()
    if return_outlier_stats:
        return profiling_mat, outlier_stats
    return profiling_mat
     
 
def _safe_cholesky(raw_scaling_diag_matrix, dev):
    raw_scaling_diag_matrix = raw_scaling_diag_matrix.float()
    try:
        return torch.linalg.cholesky(raw_scaling_diag_matrix)
    except Exception:
        print("Warning: scaling_diag_matrix is not positive definite, adding jitter.")
        eigenvalues = torch.linalg.eigvalsh(raw_scaling_diag_matrix)
        jitter = (-eigenvalues[0] + 1e-6).item()
        raw_scaling_diag_matrix = raw_scaling_diag_matrix + jitter * torch.eye(raw_scaling_diag_matrix.shape[0], device=dev)
        return torch.linalg.cholesky(raw_scaling_diag_matrix)


def _safe_inverse(scaling_diag_matrix, dev):
    try:
        return torch.linalg.inv(scaling_diag_matrix)
    except Exception:
        print("Warning: scaling_diag_matrix is not full rank, adding jitter.")
        scaling_diag_matrix = scaling_diag_matrix + 1e-6 * torch.eye(scaling_diag_matrix.shape[0], device=dev)
        return torch.linalg.inv(scaling_diag_matrix)


def _target_rank(rows, cols, ratio, max_rank):
    if max_rank <= 0:
        return 0
    rank = int(rows * cols * ratio / (rows + cols))
    rank = max(1, rank)
    return min(rank, max_rank)


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
                "sum_energy": torch.zeros_like(info["singular_values"], device=dev),
                "sum_sq_energy": torch.zeros_like(info["singular_values"], device=dev),
                "total_seqs": 0,
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

                # R_tensor: [batch, seq_len, rank]
                response_tensor = torch.matmul(inp, info["proj_matrix"].transpose(0, 1))
                # seq_energy: [batch, rank]
                seq_energy = torch.sum(response_tensor * response_tensor, dim=1)

                info["sum_energy"] += seq_energy.sum(dim=0)
                info["sum_sq_energy"] += (seq_energy * seq_energy).sum(dim=0)
                info["total_seqs"] += seq_energy.shape[0]
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
            total_seqs = info["total_seqs"]
            if total_seqs <= 0:
                score = torch.full_like(info["sum_energy"], -1e10)
            else:
                mean_energy = info["sum_energy"] / total_seqs
                var_energy = (info["sum_sq_energy"] / total_seqs) - mean_energy * mean_energy
                var_energy = torch.clamp(var_energy, min=0)
                score = mean_energy - stability_lambda * torch.sqrt(var_energy)
            target_rank = decomposition_book[i][name]["target_rank"]
            target_rank = min(target_rank, score.numel())
            selected_idx = torch.topk(score, k=target_rank, largest=True).indices
            decomposition_book[i][name]["selected_idx"] = selected_idx.cpu()

            info["normal_idx"] = info["proj_matrix"] = None
            info["sum_energy"] = info["sum_sq_energy"] = None
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
    outlier_channel_stats=None,
    enable_stage3=False,
    stage3_lambda=1.0,
    stage3_max_batches=None,
):
    model.eval()
    if "opt" in model_name:
        layers = model.model.decoder.layers
    else:
        layers = model.model.layers

    print("Start Stage 1 + Stage 2 decomposition...")
    decomposition_book = {}

    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        layer_book = {}
        for name in subset:
            module = subset[name]
            W = module.weight.data.float().to(dev)
            scaling_diag_matrix = profiling_mat[i][name].to(dev).float()
            channel_max_abs = None
            if outlier_channel_stats is not None and i in outlier_channel_stats and name in outlier_channel_stats[i]:
                channel_max_abs = outlier_channel_stats[i][name]
            normal_idx, outlier_idx = _select_channel_partitions(
                scaling_diag_matrix,
                stage1_outlier_ratio,
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

            scaling_matrix_inv = _safe_inverse(scaling_diag_matrix_normal, dev).float()
            W_normal = W.index_select(1, normal_idx)
            W_scale = torch.matmul(W_normal, scaling_diag_matrix_normal)
            U, singular_values, VT = torch.linalg.svd(W_scale, full_matrices=False)
            right_proj = torch.matmul(VT, scaling_matrix_inv)
            proj_matrix = singular_values.unsqueeze(1) * right_proj
            target_rank = _target_rank(W_normal.shape[0], W_normal.shape[1], ratio, singular_values.numel())

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

            W = W_normal = W_scale = scaling_diag_matrix = scaling_diag_matrix_normal = scaling_matrix_inv = None
            U = singular_values = VT = right_proj = proj_matrix = None
            del W, W_normal, W_scale, scaling_diag_matrix, scaling_diag_matrix_normal, scaling_matrix_inv, U, singular_values, VT, right_proj, proj_matrix
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
    for i in tqdm(range(len(layers))):
        layer = layers[i]
        subset = find_layers(layer)
        for name in subset:
            module = subset[name]
            info = decomposition_book[i][name]

            selected_idx = info["selected_idx"]
            if selected_idx.numel() == 0 and info["normal_idx"].numel() > 0:
                selected_idx = torch.arange(min(1, info["singular_values"].numel()), dtype=torch.long)

            U_sel = info["U"][:, selected_idx].float()
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


@torch.no_grad()
def whitening_local_update(model_name, model, dataloader, profiling_mat, ratio, dev, direct_update=False):
    print("Start SVD decomposition then update...")
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
    model.model.norm = model.model.norm.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (len(dataloader), model.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {'i': 0, 'attention_mask': None, "position_ids": None}
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module
        def forward(self, inp, **kwargs):
            inps[cache['i']] = inp
            cache['i'] += 1
            if cache['attention_mask'] is None:
                cache['attention_mask'] = kwargs['attention_mask']
                if "opt" not in model_name:
                    cache['position_ids'] = kwargs['position_ids']
            else:
                cache['attention_mask'] = torch.cat((cache['attention_mask'], kwargs['attention_mask']), dim=0)
                if "opt" not in model_name:
                    cache['position_ids'] = torch.cat((cache['position_ids'], kwargs['position_ids']), dim=0)
            raise ValueError
    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(dev))
        except ValueError:
            pass
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
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
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
        if "opt" not in model_name:
            outs = layer(inps, attention_mask=attention_masks, position_ids=position_ids)[0]
        else:
            outs = layer(inps, attention_mask=attention_masks)[0]
        layers[i] = layer.cpu()
        del gpts
        torch.cuda.empty_cache()
        inps = outs
        outs = None
        del outs
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
            self.U, self.S, self.VT = torch.linalg.svd(W.data, full_matrices=False)
        else: 
            try:
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            except Exception as e:
                print("Warning: scaling_diag_matrix is not full rank!")
                scaling_diag_matrix += 1e-6 * torch.eye(scaling_diag_matrix.shape[0])
                scaling_matrix_inv = torch.linalg.inv(scaling_diag_matrix)
            scaling_diag_matrix = scaling_diag_matrix.float()
            scaling_matrix_inv = scaling_matrix_inv.float()
            W_scale = torch.matmul(W, scaling_diag_matrix)
            self.U, self.S, self.VT = torch.linalg.svd(W_scale, full_matrices=False)  
        # trucation SVD
        num_s_after_trunc = int(W.shape[0] * W.shape[1] * ratio / (W.shape[0] + W.shape[1]))
        self.truc_s = self.S[:num_s_after_trunc].cuda()
        self.truc_u = self.U[:, :num_s_after_trunc].cuda()
        if direct_update:
            self.truc_v = self.VT[:num_s_after_trunc, :].cuda()
        else:
            self.truc_v = torch.matmul(self.VT[:num_s_after_trunc, :].cuda(), scaling_matrix_inv)
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
    parser.add_argument('--enable_stage3', action='store_true', help='Stage 3: enable cross-sample stability selection')
    parser.add_argument('--stage3_lambda', type=float, default=1.0, help='Stage 3: variance penalty coefficient in stability score')
    parser.add_argument('--stage3_max_batches', type=int, default=None, help='Stage 3: optionally limit number of calibration batches for stability scoring')
    
    args = parser.parse_args()
    args.ratio = 1- args.ratio
    if args.step == 1:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
        model = model.eval()
        need_outlier_stats = args.stage1_outlier_ratio > 0 and args.stage1_outlier_criterion == "infinity_norm"
        cali_white_data = None
        if args.profiling_mat_path is None or args.enable_stage3 or need_outlier_stats:
            cali_white_data = get_calib_train_data(args.dataset, tokenizer, args.whitening_nsamples, seqlen=args.model_seq_len)
        outlier_channel_stats = None
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
        whitening(
            args.model,
            model,
            profiling_mat,
            args.ratio,
            args.DEV,
            calib_loader=cali_white_data,
            stage1_outlier_ratio=args.stage1_outlier_ratio,
            stage1_outlier_criterion=args.stage1_outlier_criterion,
            outlier_channel_stats=outlier_channel_stats,
            enable_stage3=args.enable_stage3,
            stage3_lambda=args.stage3_lambda,
            stage3_max_batches=args.stage3_max_batches,
        )
        if args.save_path is not None:
            torch.save({'model': model, 'tokenizer': tokenizer}, args.save_path + "/" + args.model.replace("/", "_").replace("-", "_") +'_whitening_only_' + str(args.ratio) + '.pt')   # fp32
    elif args.step == 2:
        model, tokenizer = get_model_from_huggingface(model_id=args.model)
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
