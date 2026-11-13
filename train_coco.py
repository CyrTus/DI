import csv
import importlib.util
import math
import os
import random
import time
from pathlib import Path

import click
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch import autocast
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.utils.tensorboard import SummaryWriter
import tqdm

from torch_utils import distributed as dist
from torch_utils.get_optimzer_lr import get_adam_step_size_stats
from loss_config_utils import load_loss_config
import solver_utils
import solvers_train  # noqa: F401 - kept for parity with the original training entry.
import solvers_traj

def make_eps_list(eps_min, eps_max, n, device=None, dtype=torch.float32):
    if n <= 0:
        raise ValueError("n 必须 > 0")
    c = n // 2  # 奇数取最中间，偶数取中间靠后
    eps_list = torch.empty(n, device=device, dtype=dtype)

    # 左侧(含中心): i in [0, c], i=0 -> eps_max, i=c -> eps_min
    if c == 0:
        eps_list[0] = eps_min
    else:
        i = torch.arange(c + 1, device=device, dtype=dtype)
        t_left = (c - i) / c
        eps_list[:c + 1] = eps_min + (eps_max - eps_min) * t_left

    # 右侧: i in [c+1, n-1], i=c+1 -> > eps_min, i=n-1 -> eps_max
    right_len = n - c - 1
    if right_len > 0:
        j = torch.arange(1, right_len + 1, device=device, dtype=dtype)
        t_right = j / right_len
        eps_list[c + 1:] = eps_min + (eps_max - eps_min) * t_right

    return eps_list


def _load_base_module():
    base_path = Path(__file__).with_name("train-coco-euler.py")
    spec = importlib.util.spec_from_file_location("train_coco_euler", base_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_loss_module():
    loss_path = Path(__file__).with_name("train-euler.py")
    spec = importlib.util.spec_from_file_location("train_euler", loss_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


base = _load_base_module()
StackedRandomGenerator = base.StackedRandomGenerator
create_model = base.create_model
get_denoised = base.get_denoised
parse_int_list = base.parse_int_list

loss_base = _load_loss_module()
DI_Dataset = loss_base.DI_Dataset  # dict-target aware version
InceptionFeatureExtractor = loss_base.InceptionFeatureExtractor
LPIPSLoss = loss_base.LPIPSLoss
inception_feature_mse_loss = loss_base.inception_feature_mse_loss
lpips_final_loss = loss_base.lpips_final_loss
multiscale_lowfreq_loss = loss_base.multiscale_lowfreq_loss
gradient_edge_loss = loss_base.gradient_edge_loss
ms_ssim_loss = loss_base.ms_ssim_loss
structure_loss = loss_base.structure_loss
pseudo_huber_loss = loss_base.pseudo_huber_loss
loss_items_use_type = loss_base.loss_items_use_type
loss_component_key = loss_base.loss_component_key
maybe_normalize_state_loss_pair = loss_base.maybe_normalize_state_loss_pair
build_step_loss_plan = loss_base.build_step_loss_plan
get_training_targets = loss_base.get_training_targets
get_predictions_for_loss = loss_base.get_predictions_for_loss


class DINOv2FeatureExtractor(nn.Module):
    def __init__(self, device, model_name="dinov2_vits14"):
        super().__init__()
        try:
            model = torch.hub.load("facebookresearch/dinov2", model_name)
        except Exception as exc:
            raise RuntimeError(
                "Failed to load DINOv2 via torch.hub. Check network access or the torch hub cache "
                "before using a loss_config with type: dinov2."
            ) from exc
        model.eval().requires_grad_(False)
        self.model = model.to(device)
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = image.to(dtype=torch.float32)
        image = torch.clamp(image / 2 + 0.5, 0, 1)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            raise ValueError(f"DINOv2 feature loss expects 1 or 3 image channels, got {image.shape[1]}")
        image = F.interpolate(image, size=(224, 224), mode="bilinear", align_corners=False)
        mean = self.mean.to(device=image.device, dtype=image.dtype)
        std = self.std.to(device=image.device, dtype=image.dtype)
        image = (image - mean) / std
        features = self.model.forward_features(image)
        if isinstance(features, dict):
            cls_token = features["x_norm_clstoken"].unsqueeze(1)
            patch_tokens = features["x_norm_patchtokens"]
            features = torch.cat([cls_token, patch_tokens], dim=1)
        return features


def dinov2_feature_mse_loss(feature_extractor, x_next, x_0, net, model_source):
    if model_source == "ldm":
        x_next = net.model.differentiable_decode_first_stage(x_next)
        with torch.no_grad():
            x_0 = net.model.differentiable_decode_first_stage(x_0)
    with torch.no_grad():
        target_features = feature_extractor(x_0)
    pred_features = feature_extractor(x_next)
    return F.mse_loss(pred_features, target_features)


def unit_normalize_channels(features):
    return features / (features.norm(dim=1, keepdim=True) + 1e-6)


class LPLLoss(nn.Module):
    """Decoder-feature latent perceptual loss (LPL).

    Runs predicted/target latents through the frozen first-stage VAE decoder and
    compares LPIPS-style unit-normalized intermediate decoder features, so the
    perceptual signal is obtained without leaving latent space training.
    """

    def __init__(self, net):
        super().__init__()
        if not hasattr(net, "model") or not hasattr(net.model, "first_stage_model"):
            raise ValueError("LPL loss requires an LDM model with a first_stage_model decoder")
        self.ldm = net.model
        decoder = self.ldm.first_stage_model.decoder
        self.feature_modules = [decoder.mid.block_2] + [up.block[-1] for up in decoder.up]
        self._features = None
        for module in self.feature_modules:
            module.register_forward_hook(self._capture_hook)

    def _capture_hook(self, module, inputs, output):
        if self._features is not None:
            self._features.append(output)

    def extract_features(self, z):
        self._features = []
        try:
            self.ldm.differentiable_decode_first_stage(z)
            features = self._features
        finally:
            self._features = None
        return features

    def forward(self, x_next, x_0):
        pred_features = self.extract_features(x_next)
        with torch.no_grad():
            target_features = self.extract_features(x_0)
        loss = None
        for pred, target in zip(pred_features, target_features):
            layer_loss = F.mse_loss(unit_normalize_channels(pred), unit_normalize_channels(target.detach()))
            loss = layer_loss if loss is None else loss + layer_loss
        return loss / len(pred_features)


def compute_selected_loss(loss_type, criterion, feature_losses, x_next, x_0, net, model_source, pseudo_huber_delta=None):
    if loss_type == "l2":
        return criterion(x_next, x_0)
    if loss_type == "l1":
        return F.l1_loss(x_next, x_0)
    if loss_type == "pseudo_huber":
        return pseudo_huber_loss(x_next, x_0, pseudo_huber_delta)
    if loss_type == "inception":
        return inception_feature_mse_loss(feature_losses["inception"], x_next, x_0, net, model_source)
    if loss_type == "lpips":
        return lpips_final_loss(feature_losses["lpips"], x_next, x_0, net, model_source)
    if loss_type == "dinov2":
        return dinov2_feature_mse_loss(feature_losses["dinov2"], x_next, x_0, net, model_source)
    if loss_type == "lpl":
        return feature_losses["lpl"](x_next, x_0)
    if loss_type == "multiscale":
        return multiscale_lowfreq_loss(x_next, x_0, net, model_source)
    if loss_type == "gradient":
        return gradient_edge_loss(x_next, x_0, net, model_source)
    if loss_type == "ms_ssim":
        return ms_ssim_loss(x_next, x_0, net, model_source)
    if loss_type == "structure":
        return structure_loss(x_next, x_0, net, model_source)
    raise ValueError(f"Unsupported loss type: {loss_type}")


def compute_composite_loss(loss_items, loss_weights, criterion, feature_losses, predictions, targets, net, model_source, step_sigma=None, state_normalization_config=None):
    total_loss = None
    component_losses = {}
    for idx, (item, loss_weight) in enumerate(zip(loss_items, loss_weights)):
        pred_for_loss, target_for_loss = maybe_normalize_state_loss_pair(
            predictions[item["target"]],
            targets[item["target"]],
            item,
            step_sigma,
            state_normalization_config,
        )
        component_loss = compute_selected_loss(
            item["type"],
            criterion,
            feature_losses,
            pred_for_loss,
            target_for_loss,
            net,
            model_source,
            item.get("delta"),
        )
        weighted_loss = loss_weight * component_loss
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        component_losses[loss_component_key(item, idx)] = component_loss.detach()
    return total_loss, component_losses


def ensure_feature_losses(feature_losses, planned_loss_items, net, device):
    if loss_items_use_type(planned_loss_items, "inception") and "inception" not in feature_losses:
        dist.print0("Loading Inception-v3 feature extractor for configured loss...")
        feature_losses["inception"] = InceptionFeatureExtractor(device)
    if loss_items_use_type(planned_loss_items, "lpips") and "lpips" not in feature_losses:
        dist.print0("Loading LPIPS loss for configured loss...")
        feature_losses["lpips"] = LPIPSLoss(device)
    if loss_items_use_type(planned_loss_items, "dinov2") and "dinov2" not in feature_losses:
        dist.print0("Loading DINOv2 feature extractor for configured loss...")
        feature_losses["dinov2"] = DINOv2FeatureExtractor(device)
    if loss_items_use_type(planned_loss_items, "lpl") and "lpl" not in feature_losses:
        dist.print0("Building decoder-feature LPL loss from the first stage model...")
        feature_losses["lpl"] = LPLLoss(net)


def _world_size():
    return dist.get_world_size()


def _rank():
    return dist.get_rank()


def _barrier():
    if _world_size() > 1:
        torch.distributed.barrier()


def _sync_grads(params):
    if _world_size() == 1:
        return
    for param in params:
        if param is not None and param.grad is not None:
            # Preserve the original single-GPU accumulation scale.
            torch.distributed.all_reduce(param.grad, op=torch.distributed.ReduceOp.SUM)


def _sync_loss(loss_sum, loss_count, device):
    stats = torch.tensor([loss_sum, loss_count], device=device, dtype=torch.float64)
    if _world_size() > 1:
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
    if stats[1].item() == 0:
        return 0.0
    return (stats[0] / stats[1]).item()


def _sync_max_time(seconds, device):
    value = torch.tensor([seconds], device=device, dtype=torch.float64)
    if _world_size() > 1:
        torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.MAX)
    return value.item()


def _is_empty_metadata(value):
    return isinstance(value, list) and len(value) == 0


def _move_conditioning(net, dataset_name, solver_kwargs, class_labels, c, uc, device, batch_size):
    if _is_empty_metadata(class_labels):
        class_labels = None
    if _is_empty_metadata(c):
        c = None

    if not net.label_dim:
        return class_labels, c, uc

    if solver_kwargs["model_source"] == "adm":
        class_labels = class_labels.to(device)
    elif solver_kwargs["model_source"] == "ldm" and dataset_name == "ms_coco":
        c = c.to(device) if c is not None else None
        if uc is not None:
            uc = uc.expand(batch_size, -1, -1)
    else:
        class_labels = class_labels.to(device)

    return class_labels, c, uc


def _get_traj_sampler(gt_solver):
    if gt_solver == "euler":
        return solvers_traj.euler_sampler
    if gt_solver == "heun":
        return solvers_traj.heun_sampler
    if gt_solver == "dpm":
        return solvers_traj.dpm_2_sampler
    if gt_solver == "ipndm":
        return solvers_traj.ipndm_sampler
    if gt_solver == "ipndm_v":
        return solvers_traj.ipndm_v_sampler
    if gt_solver == "dpmpp":
        return solvers_traj.dpm_pp_sampler
    if gt_solver == "unipc":
        return solvers_traj.unipc_sampler
    if gt_solver == "deis":
        return solvers_traj.deis_sampler
    raise ValueError(f"Unsupported gt_solver: {gt_solver}")


def _load_coco_captions():
    prompt_path = "src/prompts/MS-COCO_val2014_30k_captions.csv"
    sample_captions = []
    with open(prompt_path, "r") as file:
        reader = csv.DictReader(file)
        for row in reader:
            sample_captions.append(row["text"])
    return sample_captions


def _make_rank_batches(seeds, batch_size):
    num_batches = ((len(seeds) - 1) // (batch_size * _world_size()) + 1) * _world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    return all_batches[_rank() :: _world_size()]


def _save_or_load_gt_trajectory(
    dataset_name,
    gt_solver,
    gt_num_steps,
    seeds,
    base_seed,
    net,
    solver_kwargs,
    sample_captions,
    device,
):
    cache_dir = Path(f"traj/{dataset_name}/gt_num_steps-{gt_num_steps}/gt_solver={gt_solver}")
    cache_path = cache_dir / f"traj_seeds{seeds[0]}-{seeds[-1]}.pth"

    if cache_path.exists():
        _barrier()
        return torch.load(cache_path, map_location="cpu")

    gt_t_steps = solver_utils.get_schedule(
        gt_num_steps,
        solver_kwargs["sigma_min"],
        solver_kwargs["sigma_max"],
        device=device,
        schedule_type=solver_kwargs["schedule_type"],
        schedule_rho=solver_kwargs["schedule_rho"],
        net=net,
    )
    traj_kwargs = dict(solver_kwargs)
    traj_kwargs["t_steps"] = gt_t_steps
    traj_kwargs["gt_num_steps"] = gt_num_steps
    if gt_solver == "deis":
        traj_kwargs["coeff_list"] = solver_utils.get_deis_coeff_list(
            gt_t_steps, traj_kwargs["max_order"], deis_mode=traj_kwargs["deis_mode"]
        )
    sampler_fn = _get_traj_sampler(gt_solver)

    records = []
    start = time.perf_counter()
    rank_batches = _make_rank_batches(seeds, solver_kwargs["gt_batch"])
    for batch_seeds in tqdm.tqdm(rank_batches, unit="batch", disable=(_rank() != 0)):
        batch_size = len(batch_seeds)
        if batch_size == 0:
            continue

        rnd = StackedRandomGenerator(device, batch_seeds)
        latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
        init_latents = latents.cpu()
        class_labels = c = uc = None

        if net.label_dim:
            if traj_kwargs["model_source"] == "adm":
                class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
                class_labels_cpu = class_labels.detach().cpu()
                c_cpu = None
            elif traj_kwargs["model_source"] == "ldm" and dataset_name == "ms_coco":
                if traj_kwargs["prompt"] is None:
                    start_idx = int(batch_seeds[0].item()) - base_seed
                    end_idx = int(batch_seeds[-1].item()) + 1 - base_seed
                    prompts = sample_captions[start_idx:end_idx]
                else:
                    prompts = [traj_kwargs["prompt"] for _ in range(batch_size)]
                if traj_kwargs["guidance_rate"] != 1.0:
                    uc = net.model.get_learned_conditioning(batch_size * [""])
                if isinstance(prompts, tuple):
                    prompts = list(prompts)
                c = net.model.get_learned_conditioning(prompts)
                class_labels_cpu = None
                c_cpu = c.detach().cpu()
            else:
                class_labels = torch.eye(net.label_dim, device=device)[
                    rnd.randint(net.label_dim, size=[batch_size], device=device)
                ]
                class_labels_cpu = class_labels.detach().cpu()
                c_cpu = None
        else:
            class_labels_cpu = None
            c_cpu = None

        with torch.no_grad():
            if traj_kwargs["model_source"] == "ldm":
                with autocast("cuda"):
                    with net.model.ema_scope():
                        traj_latents, denoised = sampler_fn(
                            net,
                            latents,
                            condition=c,
                            unconditional_condition=uc,
                            return_inters=True,
                            **traj_kwargs,
                        )
            else:
                traj_latents, denoised = sampler_fn(
                    net, latents, class_labels=class_labels, return_inters=True, **traj_kwargs
                )

        records.append(
            {
                "start": int(batch_seeds[0].item()),
                "gt_trajectory": traj_latents.detach().cpu(),
                "denoised": denoised.detach().cpu(),
                "latents": init_latents,
                "class_labels": class_labels_cpu,
                "c": c_cpu,
            }
        )

    sample_time = _sync_max_time(time.perf_counter() - start, device)
    if _world_size() > 1:
        gathered = [None for _ in range(_world_size())] if _rank() == 0 else None
        torch.distributed.gather_object(records, object_gather_list=gathered, dst=0)
    else:
        gathered = [records]

    if _rank() == 0:
        merged_records = [record for rank_records in gathered for record in rank_records]
        merged_records.sort(key=lambda record: record["start"])
        traj = {
            "gt_trajectory": [record["gt_trajectory"] for record in merged_records],
            "denoised_list": [record["denoised"] for record in merged_records],
            "latents": [record["latents"] for record in merged_records],
            "class_labels": [
                record["class_labels"] for record in merged_records if record["class_labels"] is not None
            ],
            "c": [record["c"] for record in merged_records if record["c"] is not None],
            "sample_time": f"{sample_time:.6f}s",
        }
        cache_dir.mkdir(parents=True, exist_ok=True)
        torch.save(traj, cache_path)
        dist.print0(f"sample takes {sample_time:.6f}s")

    _barrier()
    return torch.load(cache_path, map_location="cpu")


def _repeat_emb_list(dataset_name, emb_list, uncond_only):
    if dataset_name in ["cifar10"]:
        return [emb.repeat(34, 1) for emb in emb_list]
    if dataset_name == "imagenet64":
        return [emb.repeat(36, 1, 1) for emb in emb_list]
    if dataset_name == "ffhq":
        return [emb.repeat(45, 1) for emb in emb_list]
    if dataset_name == "imagenet256":
        return [emb.repeat(37, 1, 1) for emb in emb_list]
    if dataset_name == "ms_coco":
        if uncond_only:
            return [emb.repeat(25, 1, 1) for emb in emb_list]
        return [emb.repeat(25, 2, 1) for emb in emb_list]
    if dataset_name == "lsun_bedroom_ldm":
        return [emb.repeat(25, 1, 1) for emb in emb_list]
    if dataset_name == "lsun_bedroom":
        return [emb.repeat(37, 1) for emb in emb_list]
    return emb_list


def _build_train_loader(dataset, mini_batch, shuffle, di_seed, emb_idx):
    sampler = None
    if _world_size() > 1:
        sampler = DistributedSampler(
            dataset,
            num_replicas=_world_size(),
            rank=_rank(),
            shuffle=shuffle,
            seed=di_seed + emb_idx,
            drop_last=False,
        )
    return DataLoader(
        dataset,
        batch_size=mini_batch,
        shuffle=(shuffle and sampler is None),
        sampler=sampler,
        drop_last=False,
    )


def _update_latents_list(net, dataset_name, solver_kwargs, dataset, latents_list, mini_batch, t_cur, t_next, emb, uc, device):
    loader = DataLoader(dataset, batch_size=mini_batch, shuffle=False, drop_last=False)
    offset = 0
    for x_cur, _, class_labels, c in loader:
        x_cur = x_cur.to(device)
        batch_size = x_cur.shape[0]
        class_labels, c, uc_batch = _move_conditioning(
            net, dataset_name, solver_kwargs, class_labels, c, uc, device, batch_size
        )
        with torch.no_grad():
            denoised = get_denoised(
                net,
                x_cur,
                t_cur,
                emb,
                solver_kwargs["use_res"],
                class_labels=class_labels,
                condition=c,
                unconditional_condition=uc_batch,
            )
            d_cur = (x_cur - denoised) / t_cur
            x_next = x_cur + (t_next - t_cur) * d_cur
        latents_list[offset : offset + batch_size] = x_next.detach().cpu()
        offset += batch_size

    if _world_size() > 1:
        synced = latents_list.to(device)
        torch.distributed.broadcast(synced, src=0)
        latents_list.copy_(synced.cpu())


@click.command()
@click.option("--dataset_name", help="Name of the dataset", metavar="STR", type=str, required=True)
@click.option("--prompt", help="Prompt for Stable Diffusion sampling", metavar="STR", type=str)
@click.option("--use_fp16", help="Whether to use mixed precision", metavar="BOOL", type=bool, default=False)
@click.option("--outdir", help="Where to save the output parameters", metavar="DIR", type=str)
@click.option("--batch", "max_batch_size", help="Global batch size per optimizer step", metavar="INT", type=click.IntRange(min=1), default=128, show_default=True)
@click.option("--gt_batch", help="Batch size used to generate teacher trajectories per GPU", metavar="INT", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--epochs", help="Num of training epochs", metavar="INT", type=click.IntRange(min=1), default=2, show_default=True)
@click.option("--guidance_type", help="Guidance type", type=click.Choice(["cg", "cfg", "uncond"]), default=None, show_default=True)
@click.option("--guidance_rate", help="Guidance rate", type=float)
@click.option("--save_interval", help="interval of epochs to save model", metavar="INT", type=click.IntRange(min=1), default=1, show_default=True)
@click.option("--schedule_type", help="Time discretization schedule", metavar="STR", type=click.Choice(["polynomial", "logsnr", "time_uniform", "discrete"]), default="polynomial", show_default=True)
@click.option("--schedule_rho", help="Time step exponent", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option("--num_steps", help="Number of sampling steps", metavar="INT", type=click.IntRange(min=1), default=6, show_default=True)
@click.option("--gt_num_steps", help="Number of sampling steps", metavar="INT", type=click.IntRange(min=1), default=61, show_default=True)
@click.option("--predict_x0", help="Whether to use data prediction mode", metavar="BOOL", type=bool, default=True)
@click.option("--lower_order_final", help="Whether to lower the order at final stages", metavar="BOOL", type=bool, default=True)
@click.option("--lr", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option("--lr_min", help="Learning rate", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option("--solver", help="Name of the solver", metavar="many solvers", type=click.Choice(["euler", "ipndm", "ipndm_v", "heun", "dpm", "dpmpp", "deis", "unipc"]))
@click.option("--gt_solver", help="Name of the solver", metavar="many solvers", type=click.Choice(["euler", "ipndm", "ipndm_v", "heun", "dpm", "dpmpp", "deis", "unipc"]))
@click.option("--sigma_min", help="Lowest noise level  [default: varies]", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=0.002)
@click.option("--sigma_max", help="Highest noise level  [default: varies]", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=80.0)
@click.option("--weight_decay", help="Lowest noise level  [default: varies]", metavar="FLOAT", type=click.FloatRange(min=0), default=1e-8)
@click.option("--seeds", help="Random seeds (e.g. 1,2,5-10)", metavar="LIST", type=parse_int_list, default="0-63", show_default=True)
@click.option("--afs", help="Whether to use AFS", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--max_order", help="Max order for solvers", metavar="INT", type=click.IntRange(min=1), default=4)
@click.option("--uncond_only", help="Whether to use AFS", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--use_res", help="Whether to use AFS", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--use_final", help="Whether to use AFS", metavar="BOOL", type=bool, default=True, show_default=True)
@click.option("--eps", help="Highest noise level  [default: varies]", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1e-5)
@click.option("--eps_min", help="Highest noise level  [default: varies]", metavar="FLOAT", type=click.FloatRange(min=0, min_open=True), default=1e-3)
@click.option("--di_seed", help="Max order for solvers", metavar="INT", type=click.IntRange(min=0), default=0)
@click.option("--shuffle", help="Whether to use AFS", metavar="BOOL", type=bool, default=False, show_default=True)
@click.option("--mini_batch", help="Per-GPU micro batch size", metavar="INT", type=click.IntRange(min=1), default=64, show_default=True)
@click.option("--loss_config", help="YAML file that defines composite middle/endpoint losses", type=str, required=True)
def main(
    dataset_name,
    max_batch_size,
    outdir,
    solver,
    num_steps,
    epochs,
    lr,
    save_interval,
    seeds,
    weight_decay,
    gt_solver,
    gt_num_steps,
    max_order,
    uncond_only,
    **solver_kwargs,
):
    del save_interval
    dist.init()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)

    assert dataset_name == "ms_coco", "train_coco.py is intended for MS-COCO training."
    assert (gt_num_steps - 1) % (num_steps - 1) == 0, "gt_num_steps%num_steps must equal zero!"

    mini_batch = solver_kwargs["mini_batch"]
    global_micro_batch = mini_batch * _world_size()
    assert max_batch_size % global_micro_batch == 0, "batch must be divisible by mini_batch * world_size!"
    mini_epochs = max_batch_size // global_micro_batch

    solver_kwargs["max_order"] = max_order
    base_seed = seeds[0]
    sample_captions = _load_coco_captions() if solver_kwargs["prompt"] is None else None

    if _rank() != 0:
        _barrier()
    net, solver_kwargs["model_source"] = create_model(
        dataset_name, solver_kwargs["guidance_type"], solver_kwargs["guidance_rate"], device
    )
    net.use_fp16 = False
    if _rank() == 0:
        _barrier()

    solver_kwargs["sigma_min"] = net.sigma_min
    solver_kwargs["sigma_max"] = net.sigma_max
    traj = _save_or_load_gt_trajectory(
        dataset_name,
        gt_solver,
        gt_num_steps,
        seeds,
        base_seed,
        net,
        solver_kwargs,
        sample_captions,
        device,
    )
    gt_trajectory = traj["gt_trajectory"]
    denoised_list = traj["denoised_list"]
    class_labels_list = traj["class_labels"]
    c_list = traj["c"]

    del net
    torch.cuda.empty_cache()
    _barrier()

    if _rank() != 0:
        _barrier()
    net, solver_kwargs["model_source"] = create_model(
        dataset_name, solver_kwargs["guidance_type"], solver_kwargs["guidance_rate"], device, use_di=True
    )
    net.use_fp16 = False
    if _rank() == 0:
        _barrier()

    solver_kwargs["sigma_min"] = net.sigma_min
    solver_kwargs["sigma_max"] = net.sigma_max
    t_steps = solver_utils.get_schedule(
        num_steps,
        solver_kwargs["sigma_min"],
        solver_kwargs["sigma_max"],
        device=device,
        schedule_type=solver_kwargs["schedule_type"],
        schedule_rho=solver_kwargs["schedule_rho"],
        net=net,
    )
    solver_kwargs["t_steps"] = t_steps

    emb_list = net.get_time_emb(t_steps)
    if solver_kwargs["use_res"]:
        emb_list = [torch.zeros_like(emb) for emb in emb_list]
    emb_list = _repeat_emb_list(dataset_name, emb_list, uncond_only)
    emb_list = [emb.cpu() for emb in emb_list]

    gt_trajectory = torch.cat(gt_trajectory, dim=1)
    denoised_list = torch.cat(denoised_list, dim=1)
    if len(class_labels_list) > 0:
        class_labels_list = torch.cat(class_labels_list, dim=0)
    if len(c_list) > 0:
        c_list = torch.cat(c_list, dim=0)
    latents_list = gt_trajectory[0]

    use_final = solver_kwargs["use_final"]
    if use_final:
        gt_trajectory = gt_trajectory[-1]
        denoised_list = denoised_list[-1]

    criterion = nn.MSELoss()
    feature_losses = {}
    os.makedirs(f"{outdir}/{dataset_name}", exist_ok=True)

    # LDM checkpoint backward differentiates against model params even when
    # we only optimize emb/uc, so params must keep requires_grad=True.
    net.eval().requires_grad_(True)

    di_seed = solver_kwargs["di_seed"]
    torch.manual_seed(di_seed)
    np.random.seed(di_seed)
    random.seed(di_seed)

    shuffle = solver_kwargs["shuffle"]
    loss_config_path = solver_kwargs["loss_config"]
    loss_config, loss_config_text = load_loss_config(loss_config_path)
    loss_tag = loss_config["loss_tag"]
    loss_hash = loss_config["loss_hash"]
    loss_id = loss_config["loss_id"]
    snr_gate_config = loss_config.get("snr_gate")
    rollout_config = loss_config.get("rollout", {"enabled": False})
    if bool(rollout_config.get("enabled", False)):
        raise ValueError("rollout training is not supported by train_coco.py")
    state_normalization_config = loss_config.get("state_normalization")

    writer = None
    if _rank() == 0:
        writer = SummaryWriter(
            f'runs/{dataset_name}/schedule_rho={solver_kwargs["schedule_rho"]}/{solver}/'
            f"gt_num_steps{gt_num_steps}-step{num_steps}/di_seed={di_seed}/loss_id={loss_id}/"
            f"train_seed={seeds[0]}-{seeds[-1]}"
        )
        writer.add_text("config/loss_tag", loss_tag, 0)
        writer.add_text("config/loss_hash", loss_hash, 0)
        writer.add_text("config/loss_id", loss_id, 0)
        writer.add_text("config/loss_config", f"```\n{loss_config_text}\n```", 0)
        if snr_gate_config is not None:
            writer.add_text("config/snr_gate", str(snr_gate_config), 0)
        if state_normalization_config is not None:
            writer.add_text("config/state_normalization", str(state_normalization_config), 0)

    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is None:
            continue
        if key == "max_order" and solver in ["euler", "heun", "dpm"]:
            continue
        if key in ["predict_x0", "lower_order_final"] and solver not in ["dpmpp", "unipc"]:
            continue
        if key in ["variant"] and solver not in ["unipc"]:
            continue
        if key in ["deis_mode"] and solver != "deis":
            continue
        if key in ["prompt"] and dataset_name != "ms_coco":
            continue
        if key in ["t_steps", "coeff_list"]:
            continue
        dist.print0(f"\t{key}: {value}")
    dist.print0(f"\tworld_size: {_world_size()}")
    dist.print0(f"\tglobal_batch: {max_batch_size}")
    dist.print0(f"\tper_gpu_mini_batch: {mini_batch}")
    dist.print0(f"loss_id={loss_id}")

    eps = solver_kwargs["eps"]
    initial_lr = lr
    initial_eps = eps
    lr_min = solver_kwargs["lr_min"]
    lr_max = initial_lr
    eps_min = solver_kwargs["eps_min"]
    eps_max = initial_eps
    patience = 10
    
    eps_list = make_eps_list(eps_min, eps_max, num_steps-1).to(device)
    eps_list = torch.cat([eps_list, torch.tensor([1e-9]).to(device)])
    print(f'eps_list={eps_list}')

    uc_list = []
    uc = net.model.get_learned_conditioning([""])
    for _ in range(num_steps):
        uc_list.append(uc.detach().cpu())

    model_list = None
    start = time.perf_counter()

    for emb_idx, (emb_ori, t_cur, t_next) in enumerate(zip(emb_list[:-1], t_steps[:-1], t_steps[1:])):
        dist.print0(f"-----------emb_idx={emb_idx}---------------")
        step_loss_plan = build_step_loss_plan(emb_idx, num_steps, t_steps, loss_config, snr_gate_config)[:1]
        planned_loss_items = [item for plan in step_loss_plan for item in plan["loss_items"]]

        for plan in step_loss_plan:
            step_idx = plan["step_idx"]
            log_prefix = f"emb_idx={emb_idx}"
            if plan["snr_gate_value"] is not None:
                if writer is not None:
                    writer.add_scalar(f"snr-gate/{log_prefix}/g_i", plan["snr_gate_value"], 0)
                    writer.add_scalar(f"snr-gate/{log_prefix}/log_snr", plan["log_snr"], 0)
                dist.print0(f'snr_gate step_idx={step_idx}: log_snr={plan["log_snr"]:.6f}, g_i={plan["snr_gate_value"]:.6f}')
            for idx, (item, runtime_weight) in enumerate(zip(plan["loss_items"], plan["loss_weights"])):
                component_key = loss_component_key(item, idx)
                if writer is not None:
                    writer.add_scalar(f"loss-weights/{log_prefix}/{component_key}", runtime_weight, 0)
                dist.print0(f"loss weight step_idx={step_idx} {component_key}: {runtime_weight:.6f}")

        # rank0 downloads pretrained feature extractors first to avoid races.
        if _rank() != 0:
            _barrier()
        ensure_feature_losses(feature_losses, planned_loss_items, net, device)
        if _rank() == 0:
            _barrier()

        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * emb_idx / (num_steps - 1)))
        # if emb_idx == num_steps - 2:
        #     eps = 1e-9
        # else:
        #     eps = eps_max + (eps_min - eps_max) * (emb_idx / (num_steps - 2))
        eps = eps_list[emb_idx]
        dist.print0(f"Current lr: {lr:.8f}, Current eps: {eps:.8f}")

        emb = nn.Parameter(emb_ori.clone().to(device))
        uc = nn.Parameter(uc_list[emb_idx].detach().clone().to(device))

        optimizer = optim.Adam([emb, uc], lr=lr, weight_decay=weight_decay)
        optimizer_params = [emb, uc]

        if writer is not None:
            writer.add_scalar("eps", eps, emb_idx)

        loss_targets = get_training_targets(emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps)
        dataset = DI_Dataset(latents_list, loss_targets, class_labels_list, c_list)
        data_loader = _build_train_loader(dataset, mini_batch, shuffle, di_seed, emb_idx)

        no_improve = 0
        last_sum_loss = None
        for epoch in range(epochs):
            if isinstance(data_loader.sampler, DistributedSampler):
                data_loader.sampler.set_epoch(epoch)

            actual_lr = get_adam_step_size_stats(optimizer)
            if actual_lr is None:
                actual_lr = lr
            if writer is not None:
                writer.add_scalar(f"lr/emb_idx={emb_idx}", actual_lr, epoch)

            optimizer.zero_grad()
            loss_sum = 0.0
            loss_count = 0
            component_sums = {}
            for plan in step_loss_plan:
                for idx, item in enumerate(plan["loss_items"]):
                    component_sums[loss_component_key(item, idx)] = 0.0
            denoised_temp = []
            x_0_temp = []

            for batch_idx, (x_cur, x_0, class_labels, c) in enumerate(data_loader):
                x_cur = x_cur.to(device)
                if isinstance(x_0, dict):
                    x_0 = {key: value.to(device) for key, value in x_0.items()}
                else:
                    x_0 = x_0.to(device)
                batch_size = x_cur.shape[0]
                class_labels, c, uc_batch = _move_conditioning(
                    net, dataset_name, solver_kwargs, class_labels, c, uc, device, batch_size
                )

                denoised = get_denoised(
                    net,
                    x_cur,
                    t_cur,
                    emb,
                    solver_kwargs["use_res"],
                    class_labels=class_labels,
                    condition=c,
                    unconditional_condition=uc_batch,
                )
                d_cur = (x_cur - denoised) / t_cur
                x_next = x_cur + (t_next - t_cur) * d_cur
                predictions = get_predictions_for_loss(x_next, denoised)

                loss, component_losses = compute_composite_loss(
                    step_loss_plan[0]["loss_items"],
                    step_loss_plan[0]["loss_weights"],
                    criterion,
                    feature_losses,
                    predictions,
                    x_0,
                    net,
                    solver_kwargs["model_source"],
                    step_sigma=t_next,
                    state_normalization_config=state_normalization_config,
                )
                loss.backward()
                net.zero_grad(set_to_none=True)
                loss_sum += loss.item() * batch_size
                loss_count += batch_size
                for key, value in component_losses.items():
                    component_sums[key] += value.item() * batch_size

                if writer is not None and batch_idx >= len(data_loader) - mini_epochs:
                    display_target = x_0["state"] if isinstance(x_0, dict) else x_0
                    with torch.no_grad():
                        denoised_for_log = net.model.decode_first_stage(denoised)
                        x_0_for_log = net.model.decode_first_stage(display_target)
                    denoised_temp.append(denoised_for_log.detach().cpu())
                    x_0_temp.append(x_0_for_log.detach().cpu())

                should_step = ((batch_idx + 1) % mini_epochs == 0) or (batch_idx + 1 == len(data_loader))
                if should_step:
                    _sync_grads(optimizer_params)
                    optimizer.step()
                    optimizer.zero_grad()

            sum_loss = _sync_loss(loss_sum, loss_count, device)
            synced_components = {
                key: _sync_loss(value, loss_count, device) for key, value in component_sums.items()
            }

            if writer is not None:
                writer.add_scalar(f"train-loss/emb_idx={emb_idx}", sum_loss, epoch)
                for key, value in synced_components.items():
                    writer.add_scalar(f"train-loss-components/emb_idx={emb_idx}/{key}", value, epoch)
                if denoised_temp:
                    denoised_temp = torch.cat(denoised_temp, dim=0)
                    x_0_temp = torch.cat(x_0_temp, dim=0)
                    image = torch.clamp(denoised_temp / 2 + 0.5, 0, 1)
                    writer.add_images(f"train-x_cur/emb_idx={emb_idx}", image, global_step=epoch)
                    image = torch.clamp(x_0_temp / 2 + 0.5, 0, 1)
                    writer.add_images(f"train-x_0/emb_idx={emb_idx}", image, global_step=epoch)
                writer.flush()

            if last_sum_loss is not None:
                delta = (last_sum_loss - sum_loss) / last_sum_loss
                if writer is not None:
                    writer.add_scalars(f"early_stop/emb_idx={emb_idx}", {"delta": delta}, epoch)
                    writer.add_scalars(f"early_stop/emb_idx={emb_idx}", {"eps": eps}, epoch)

                dist.print0(
                    f"last_sum_loss:{last_sum_loss:.8f},sum_loss:{sum_loss:.8f},"
                    f"delta:{delta:.8f},epoch:{epoch}"
                )
                dist.print0(f"uc={uc.mean()}")

                if (delta < eps and sum_loss < last_sum_loss) or epoch == epochs - 1:
                    no_improve += 1
                    if no_improve >= patience or epoch == epochs - 1:
                        _update_latents_list(
                            net,
                            dataset_name,
                            solver_kwargs,
                            dataset,
                            latents_list,
                            mini_batch,
                            t_cur,
                            t_next,
                            emb,
                            uc,
                            device,
                        )
                        _barrier()
                        break
                else:
                    no_improve = 0
            last_sum_loss = sum_loss

        emb_list[emb_idx] = emb.detach().cpu()
        uc_list[emb_idx] = uc.detach().cpu()

    train_time = _sync_max_time(time.perf_counter() - start, device)
    dist.print0(f"training takes {train_time:.6f}")

    if writer is not None:
        writer.close()

    use_res = solver_kwargs["use_res"]
    output_dir = (
        f"{outdir}/{dataset_name}/{solver}/step{num_steps}/gt_num_steps{gt_num_steps}/"
        f'schedule_type={solver_kwargs["schedule_type"]}/seed{seeds[0]}-{seeds[-1]}/lr={initial_lr}/'
        f"use_final={use_final}/use_res={use_res}/di_seed={di_seed}/shuffle={shuffle}/loss_id={loss_id}"
    )
    if _rank() == 0:
        os.makedirs(output_dir, exist_ok=True)
        result = {
            "emb_list": emb_list,
            "model_list": model_list,
            "time": f"{train_time:.6f}",
            "uc_list": uc_list,
            "loss_config": loss_config,
            "loss_config_text": loss_config_text,
            "loss_tag": loss_tag,
            "loss_hash": loss_hash,
            "loss_id": loss_id,
        }
        with open(os.path.join(output_dir, "loss_config.yaml"), "w") as f:
            f.write(loss_config_text)
        torch.save(
            result,
            f"{output_dir}/{dataset_name}_{num_steps}_epoch_{epochs}_seed{seeds[0]}-{seeds[-1]}.pth",
        )
    _barrier()
    dist.print0("Done.")


if __name__ == "__main__":
    main()

