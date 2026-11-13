import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.utils import make_grid, save_image
from torch_utils.download_util import check_file_by_key
from torch import autocast
from torch_utils import distributed as dist
import pickle
import torch
import PIL.Image
import dnnlib
import os
import click
import solver_utils
import time
import tqdm
import solvers_train
import solvers_traj
import solvers
import solver_utils
import re
import torch.utils.data
import csv
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
# from tensorboardX import SummaryWriter
import copy
import math
from loss_config_utils import load_loss_config
from torch_utils.get_optimzer_lr import get_adam_avg_actual_lr, get_effective_lr_stats, get_adam_step_size_stats

# from interpolation_algorithms import square_interpolation,exponential_interpolation

class DI_Dataset(Dataset):
    def __init__(self, data_list, label_list, class_labels_list=None,c_list=None,transform=None):
        self.data_list = data_list
        self.label_list = label_list
        self.transform = transform
        self.class_labels_list = class_labels_list
        self.c_list = c_list
        # print(f'class_labels_list={class_labels_list}')
        # print(f'c_list={c_list}')
        # print(f'class_labels_list={len(class_labels_list)}')
        # print(f'c_list={len(c_list)}')
        # exit()

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        
        data = self.data_list[idx]
        if isinstance(self.label_list, dict):
            label = {key: value[idx] for key, value in self.label_list.items()}
        else:
            label = self.label_list[idx]

        if len(self.class_labels_list)>0:
            class_labels = self.class_labels_list[idx]
        else:
            class_labels = []
            
        if len(self.c_list)>0:
            c = self.c_list[idx]
        else:
            c = []
        
        if self.transform:
            data = self.transform(data)

        return data, label, class_labels, c
    
# class DI_Dataset_cond(Dataset):
#     def __init__(self, data_list, label_list, class_labels_list, transform=None):
#         self.data_list = data_list
#         self.label_list = label_list
#         self.class_labels_list = class_labels_list
#         self.transform = transform

#     def __len__(self):
#         return len(self.data_list)

#     def __getitem__(self, idx):
#         data = self.data_list[idx]
#         label = self.label_list[idx]
#         class_labels = self.class_labels_list[idx]
#         if self.transform:
#             data = self.transform(data)

#         return data, label, class_labels

def get_denoised(net, x, t, emb, use_res, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):       # models from LDM and Stable Diffusion
        # print('guidance_type')
        # exit()
        denoised = net(x, t, emb, use_res=use_res, condition=condition, unconditional_condition=unconditional_condition)
    else:
        # print('norm')
        # exit()
        denoised = net(x, t, emb, use_res=use_res, class_labels=class_labels)
       
    return denoised

class InceptionFeatureExtractor(nn.Module):
    def __init__(self, device):
        super().__init__()
        try:
            weights = torchvision.models.Inception_V3_Weights.DEFAULT
            model = torchvision.models.inception_v3(weights=weights, aux_logits=True)
        except AttributeError:
            model = torchvision.models.inception_v3(pretrained=True, aux_logits=True, transform_input=False)
        if hasattr(model, 'transform_input'):
            model.transform_input = False
        model.fc = nn.Identity()
        model.eval().requires_grad_(False)
        self.model = model.to(device)
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, image):
        image = image.to(dtype=torch.float32)
        image = torch.clamp(image / 2 + 0.5, 0, 1)
        if image.shape[1] == 1:
            image = image.repeat(1, 3, 1, 1)
        elif image.shape[1] != 3:
            raise ValueError(f'Inception feature loss expects 1 or 3 image channels, got {image.shape[1]}')
        image = F.interpolate(image, size=(299, 299), mode='bilinear', align_corners=False)
        mean = self.mean.to(device=image.device, dtype=image.dtype)
        std = self.std.to(device=image.device, dtype=image.dtype)
        image = (image - mean) / std
        features = self.model(image)
        if isinstance(features, tuple):
            features = features[0]
        return features

def inception_feature_mse_loss(feature_extractor, x_next, x_0, net, model_source):
    if model_source == 'ldm':
        x_next = net.model.differentiable_decode_first_stage(x_next)
        with torch.no_grad():
            x_0 = net.model.differentiable_decode_first_stage(x_0)
    with torch.no_grad():
        target_features = feature_extractor(x_0)
    pred_features = feature_extractor(x_next)
    return F.mse_loss(pred_features, target_features)

class LPIPSLoss(nn.Module):
    def __init__(self, device):
        super().__init__()
        try:
            lpips = __import__('lpips')
        except ImportError as exc:
            raise ImportError('LPIPS loss requires the "lpips" package. Please install it before using a loss_config with type: lpips.') from exc
        self.loss_fn = lpips.LPIPS(net='alex').eval().requires_grad_(False).to(device)

    def forward(self, x_next, x_0):
        x_next = prepare_lpips_image(x_next)
        x_0 = prepare_lpips_image(x_0)
        return self.loss_fn(x_next, x_0).mean()

def prepare_lpips_image(image):
    image = image.to(dtype=torch.float32)
    if image.shape[1] == 1:
        image = image.repeat(1, 3, 1, 1)
    elif image.shape[1] != 3:
        raise ValueError(f'LPIPS loss expects 1 or 3 image channels, got {image.shape[1]}')
    return torch.clamp(image, -1, 1)

def lpips_final_loss(lpips_loss_fn, x_next, x_0, net, model_source):
    if model_source == 'ldm':
        x_next = net.model.differentiable_decode_first_stage(x_next)
        with torch.no_grad():
            x_0 = net.model.differentiable_decode_first_stage(x_0)
    else:
        x_0 = x_0.detach()
    return lpips_loss_fn(x_next, x_0)

def get_image_space_pair(x_next, x_0, net, model_source):
    if model_source == 'ldm':
        x_next = net.model.differentiable_decode_first_stage(x_next)
        with torch.no_grad():
            x_0 = net.model.differentiable_decode_first_stage(x_0)
    else:
        x_0 = x_0.detach()
    return x_next.to(dtype=torch.float32), x_0.to(dtype=torch.float32)

def multiscale_lowfreq_loss(x_next, x_0, net, model_source):
    x_next, x_0 = get_image_space_pair(x_next, x_0, net, model_source)
    loss = x_next.new_tensor(0.)
    total_weight = 0.
    for scale, weight in [(1, 1.0), (2, 0.5), (4, 0.25)]:
        if scale > 1:
            pred = F.avg_pool2d(x_next, kernel_size=scale, stride=scale)
            target = F.avg_pool2d(x_0, kernel_size=scale, stride=scale)
        else:
            pred, target = x_next, x_0
        loss = loss + weight * F.l1_loss(pred, target)
        total_weight += weight
    return loss / total_weight

def gradient_edge_loss(x_next, x_0, net, model_source):
    x_next, x_0 = get_image_space_pair(x_next, x_0, net, model_source)
    loss = x_next.new_tensor(0.)
    total_weight = 0.
    for scale, weight in [(1, 1.0), (2, 0.5), (4, 0.25)]:
        if scale > 1:
            pred = F.avg_pool2d(x_next, kernel_size=scale, stride=scale)
            target = F.avg_pool2d(x_0, kernel_size=scale, stride=scale)
        else:
            pred, target = x_next, x_0
        pred_grad_x, pred_grad_y = sobel_gradients(pred)
        target_grad_x, target_grad_y = sobel_gradients(target)
        loss = loss + weight * (F.l1_loss(pred_grad_x, target_grad_x) + F.l1_loss(pred_grad_y, target_grad_y))
        total_weight += weight
    return loss / total_weight

def sobel_gradients(image):
    channels = image.shape[1]
    kernel_x = image.new_tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
    kernel_y = image.new_tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
    kernel_x = kernel_x.repeat(channels, 1, 1, 1)
    kernel_y = kernel_y.repeat(channels, 1, 1, 1)
    grad_x = F.conv2d(image, kernel_x, padding=1, groups=channels)
    grad_y = F.conv2d(image, kernel_y, padding=1, groups=channels)
    return grad_x, grad_y

def structure_loss(x_next, x_0, net, model_source):
    return multiscale_lowfreq_loss(x_next, x_0, net, model_source) + 0.5 * gradient_edge_loss(x_next, x_0, net, model_source)

def ms_ssim_loss(x_next, x_0, net, model_source):
    x_next, x_0 = get_image_space_pair(x_next, x_0, net, model_source)
    x_next = torch.clamp(x_next / 2 + 0.5, 0, 1)
    x_0 = torch.clamp(x_0 / 2 + 0.5, 0, 1)
    return 1. - ms_ssim(x_next, x_0)

def ms_ssim(x, y):
    weights = x.new_tensor([0.0448, 0.2856, 0.3001, 0.2363, 0.1333])
    levels = min(weights.numel(), max(1, int(math.log2(min(x.shape[-2], x.shape[-1]))) - 1))
    weights = weights[:levels]
    weights = weights / weights.sum()
    mssim = []
    mcs = []
    for level in range(levels):
        ssim_val, cs = ssim_per_channel(x, y)
        mssim.append(ssim_val)
        mcs.append(cs)
        if level < levels - 1:
            x = F.avg_pool2d(x, kernel_size=2, stride=2)
            y = F.avg_pool2d(y, kernel_size=2, stride=2)
    mssim = torch.stack(mssim)
    mcs = torch.stack(mcs)
    if levels == 1:
        return torch.clamp(mssim[0], min=0., max=1.)
    ms_components = torch.cat([torch.clamp(mcs[:-1], min=1e-6), torch.clamp(mssim[-1:], min=1e-6)])
    return torch.prod(ms_components ** weights)

def ssim_per_channel(x, y):
    c1 = 0.01 ** 2
    c2 = 0.03 ** 2
    kernel_size = 3
    padding = kernel_size // 2
    mu_x = F.avg_pool2d(x, kernel_size=kernel_size, stride=1, padding=padding)
    mu_y = F.avg_pool2d(y, kernel_size=kernel_size, stride=1, padding=padding)
    sigma_x = F.avg_pool2d(x * x, kernel_size=kernel_size, stride=1, padding=padding) - mu_x.square()
    sigma_y = F.avg_pool2d(y * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_y.square()
    sigma_xy = F.avg_pool2d(x * y, kernel_size=kernel_size, stride=1, padding=padding) - mu_x * mu_y
    cs = (2 * sigma_xy + c2) / (sigma_x + sigma_y + c2)
    ssim_val = ((2 * mu_x * mu_y + c1) * (2 * sigma_xy + c2)) / ((mu_x.square() + mu_y.square() + c1) * (sigma_x + sigma_y + c2))
    return ssim_val.mean(), cs.mean()

def pseudo_huber_loss(pred, target, delta):
    diff = pred - target
    delta = torch.as_tensor(delta, device=diff.device, dtype=diff.dtype)
    return delta.square() * (torch.sqrt(1. + (diff / delta).square()) - 1.).mean()

def pseudo_huber_loss_vector(pred, target, c):
    diff = pred - target
    diff = diff.flatten(1)
    dist2 = diff.square().sum(dim=1)
    return (torch.sqrt(dist2 + c ** 2) - c).mean()

def loss_items_use_type(loss_items, loss_type):
    return any(item['type'] == loss_type for item in loss_items)

def loss_component_key(item, idx):
    return f'{idx:02d}_{item["type"]}_{item["target"]}'

def normalize_edm_state_pair(pred, target, sigma, sigma_data):
    sigma = torch.as_tensor(sigma, device=pred.device, dtype=pred.dtype)
    sigma_data = torch.as_tensor(sigma_data, device=pred.device, dtype=pred.dtype)
    scale = torch.rsqrt(sigma.square() + sigma_data.square())
    return pred * scale, target * scale

def maybe_normalize_state_loss_pair(pred, target, item, step_sigma, state_normalization_config):
    if state_normalization_config is None or not state_normalization_config.get('enabled', False):
        return pred, target
    if item['target'] != 'state' or item['type'] not in ['l1', 'l2', 'pseudo_huber']:
        return pred, target
    if step_sigma is None:
        raise ValueError('step_sigma is required when state_normalization is enabled for state l2/pseudo_huber loss')
    return normalize_edm_state_pair(pred, target, step_sigma, state_normalization_config['sigma_data'])

def compute_snr_gate_value(sigma, gate_config):
    sigma_tensor = torch.as_tensor(sigma).detach().float().cpu()
    sigma_data = torch.tensor(gate_config['sigma_data'], dtype=sigma_tensor.dtype)
    log_snr = 2. * torch.log(sigma_data) - 2. * torch.log(torch.clamp(sigma_tensor, min=1e-12))
    gate = torch.sigmoid((log_snr - gate_config['tau']) / gate_config['kappa'])
    return gate.item(), log_snr.item()

def compute_runtime_loss_weights(loss_items, snr_gate_value, gate_config):
    runtime_weights = []
    for item in loss_items:
        base_weight = item['weight']
        gate_type = item.get('gate')
        if gate_type is None:
            runtime_weights.append(base_weight)
        elif gate_type == 'state':
            runtime_weights.append(base_weight * (gate_config['w_min'] + (1. - snr_gate_value) ** gate_config['p']))
        elif gate_type == 'clean':
            runtime_weights.append(base_weight * snr_gate_value)
        elif gate_type == 'struct':
            runtime_weights.append(base_weight * (snr_gate_value ** gate_config['q']))
        else:
            raise ValueError(f'Unsupported loss gate: {gate_type}')
    return runtime_weights

def build_step_loss_plan(start_emb_idx, num_steps, t_steps, loss_config, snr_gate_config):
    step_loss_plan = []
    for step_idx in range(start_emb_idx, num_steps - 1):
        is_final_step = step_idx == num_steps - 2
        loss_items = loss_config['endpoint'] if is_final_step else loss_config['middle']
        if snr_gate_config is None:
            loss_weights = [item['weight'] for item in loss_items]
            snr_gate_value = None
            log_snr = None
        else:
            snr_gate_value, log_snr = compute_snr_gate_value(t_steps[step_idx + 1], snr_gate_config)
            loss_weights = compute_runtime_loss_weights(loss_items, snr_gate_value, snr_gate_config)

        step_loss_plan.append({
            'step_idx': step_idx,
            'stage': 'endpoint' if is_final_step else 'middle',
            'loss_items': loss_items,
            'loss_weights': loss_weights,
            'snr_gate_value': snr_gate_value,
            'log_snr': log_snr,
        })
    return step_loss_plan

def compute_selected_loss(loss_type, criterion, inception_feature_extractor, lpips_loss_fn, x_next, x_0, net, model_source, pseudo_huber_delta=None):
    if loss_type == 'l2':
        return criterion(x_next, x_0)
    if loss_type == 'l1':
        return F.l1_loss(x_next, x_0)
    if loss_type == 'pseudo_huber':
        return pseudo_huber_loss_vector(x_next, x_0, pseudo_huber_delta)
    if loss_type == 'inception':
        return inception_feature_mse_loss(inception_feature_extractor, x_next, x_0, net, model_source)
    if loss_type == 'lpips':
        return lpips_final_loss(lpips_loss_fn, x_next, x_0, net, model_source)
    if loss_type == 'multiscale':
        return multiscale_lowfreq_loss(x_next, x_0, net, model_source)
    if loss_type == 'gradient':
        return gradient_edge_loss(x_next, x_0, net, model_source)
    if loss_type == 'ms_ssim':
        return ms_ssim_loss(x_next, x_0, net, model_source)
    if loss_type == 'structure':
        return structure_loss(x_next, x_0, net, model_source)
    raise ValueError(f'Unsupported loss type: {loss_type}')

def compute_composite_loss(loss_items, loss_weights, criterion, inception_feature_extractor, lpips_loss_fn, predictions, targets, net, model_source, step_sigma=None, state_normalization_config=None):
    total_loss = None
    component_losses = {}
    for idx, (item, loss_weight) in enumerate(zip(loss_items, loss_weights)):
        loss_type = item['type']
        target = item['target']
        pred_for_loss, target_for_loss = maybe_normalize_state_loss_pair(
            predictions[target],
            targets[target],
            item,
            step_sigma,
            state_normalization_config,
        )
        component_loss = compute_selected_loss(
            loss_type,
            criterion,
            inception_feature_extractor,
            lpips_loss_fn,
            pred_for_loss,
            target_for_loss,
            net,
            model_source,
            item.get('delta'),
        )
        weighted_loss = loss_weight * component_loss
        total_loss = weighted_loss if total_loss is None else total_loss + weighted_loss
        component_losses[loss_component_key(item, idx)] = component_loss.detach()
    return total_loss, component_losses

def get_training_target(loss_target, emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps):
    if use_final:
        if loss_target == 'state':
            return gt_trajectory
        if loss_target == 'clean':
            return denoised_list

    step_stride = (gt_num_steps - 1) // (num_steps - 1)
    if loss_target == 'state':
        return gt_trajectory[(emb_idx + 1) * step_stride]
    if loss_target == 'clean':
        return denoised_list[(emb_idx + 1) * step_stride]
    raise ValueError(f'Unsupported loss target: {loss_target}')

def get_training_targets(emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps):
    return {
        'state': get_training_target('state', emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps),
        'clean': get_training_target('clean', emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps),
    }

def get_rollout_training_targets(start_emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps):
    targets = {}
    for step_idx in range(start_emb_idx, num_steps - 1):
        for loss_target in ['state', 'clean']:
            targets[f'{step_idx}_{loss_target}'] = get_training_target(
                loss_target,
                step_idx,
                use_final,
                gt_trajectory,
                denoised_list,
                gt_num_steps,
                num_steps,
            )
    return targets

def get_rollout_step_targets(batch_targets, step_idx):
    return {
        'state': batch_targets[f'{step_idx}_state'],
        'clean': batch_targets[f'{step_idx}_clean'],
    }

def get_prediction_for_loss(loss_target, x_next, denoised):
    if loss_target == 'state':
        return x_next
    if loss_target == 'clean':
        return denoised
    raise ValueError(f'Unsupported loss target: {loss_target}')

def get_predictions_for_loss(x_next, denoised):
    return {'state': x_next, 'clean': denoised}

class StackedRandomGenerator:
    def __init__(self, device, seeds):
        super().__init__()
        self.generators = [torch.Generator(device).manual_seed(int(seed) % (1 << 32)) for seed in seeds]

    def randn(self, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randn(size[1:], generator=gen, **kwargs) for gen in self.generators])

    def randn_like(self, input):
        return self.randn(input.shape, dtype=input.dtype, layout=input.layout, device=input.device)

    def randint(self, *args, size, **kwargs):
        assert size[0] == len(self.generators)
        return torch.stack([torch.randint(*args, size=size[1:], generator=gen, **kwargs) for gen in self.generators])

#----------------------------------------------------------------------------
# Parse a comma separated list of numbers or ranges and return a list of ints.
# Example: '1,2,5-10' returns [1, 2, 5, 6, 7, 8, 9, 10]

def parse_int_list(s):
    if isinstance(s, list): return s
    ranges = []
    range_re = re.compile(r'^(\d+)-(\d+)$')
    for p in s.split(','):
        m = range_re.match(p)
        if m:
            ranges.extend(range(int(m.group(1)), int(m.group(2))+1))
        else:
            ranges.append(int(p))
    return ranges

def process_numbers(train_idx):
    # 将字符串按空格分割，并转换为整数列表
    numbers_list = [int(x) for x in train_idx.split()]
    # click.echo(f'Received numbers: {numbers_list}')
    return numbers_list

#----------------------------------------------------------------------------
# Load pre-trained models from the LDM codebase (https://github.com/CompVis/latent-diffusion) 
# and Stable Diffusion codebase (https://github.com/CompVis/stable-diffusion)

def load_ldm_model(config, ckpt, verbose=False):
    from models.ldm.util import instantiate_from_config
    pl_sd = torch.load(ckpt, map_location="cpu")
    # print(f'ckpt={ckpt}')
    # exit()
    if "global_step" in pl_sd:
        dist.print0(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)

    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)
    return model

#----------------------------------------------------------------------------

def create_model(dataset_name=None, guidance_type=None, guidance_rate=None, device=None, use_di=False):
    model_path, classifier_path = check_file_by_key(dataset_name)
    dist.print0(f'Loading the pre-trained diffusion model from "{model_path}"...')

    if dataset_name in ['cifar10', 'ffhq', 'afhqv2', 'imagenet64']:         # models from EDM
        with dnnlib.util.open_url(model_path, verbose=(dist.get_rank() == 0)) as f:
            net = pickle.load(f)['ema'].to(device)
        net.sigma_min = 0.002
        net.sigma_max = 80.0
        model_source = 'edm'

        # print(f'net={net.model}')
        # exit()

        if use_di:
            from models.networks_edm_opt import EDMPrecond
            if dataset_name in ['cifar10', 'ffhq']:
                from models.networks_edm_opt import SongUNet
                net.forward=EDMPrecond.forward.__get__(net)
                net.get_time_emb=EDMPrecond.get_time_emb.__get__(net)
                net.model.forward=SongUNet.forward.__get__(net.model)
                net.model.get_time_emb=SongUNet.get_time_emb.__get__(net.model)
            if dataset_name in ['imagenet64']:
                from models.networks_edm_opt import DhariwalUNet
                net.forward=EDMPrecond.forward.__get__(net)
                net.get_time_emb=EDMPrecond.get_time_emb.__get__(net)
                net.model.forward=DhariwalUNet.forward.__get__(net.model)
                net.model.get_time_emb=DhariwalUNet.get_time_emb.__get__(net.model)
 
        # net = EDMPrecond()

    elif dataset_name in ['lsun_bedroom', 'lsun_cat']:                      # models from Consistency Models
        if use_di:
            from models.cm.cm_model_loader_opt import load_cm_model
        else:
            from models.cm.cm_model_loader import load_cm_model
        if use_di:
            from models.networks_edm_opt import CMPrecond
        else:
            from models.networks_edm import CMPrecond
        net = load_cm_model(model_path)
        net = CMPrecond(net).to(device)
        model_source = 'cm'
    else:
        if guidance_type == 'cg':            # clssifier guidance           # models from ADM
            assert classifier_path is not None
            from models.guided_diffusion.cg_model_loader import load_cg_model
            if use_di:
                from models.networks_edm_opt import CMPrecond
            else:
                from models.networks_edm import CMPrecond
            net, classifier = load_cg_model(model_path, classifier_path)
            net = CGPrecond(net, classifier, guidance_rate=guidance_rate).to(device)
            model_source = 'adm'
        elif guidance_type in ['uncond', 'cfg']:                            # models from LDM
            from omegaconf import OmegaConf
            if use_di:
                from models.networks_edm_opt import CFGPrecond
            else:
                from models.networks_edm import CFGPrecond
            if dataset_name in ['lsun_bedroom_ldm']:
                if use_di:
                    config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4_opt.yaml')
                else:
                    config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                # config = OmegaConf.load('./models/ldm/configs/latent-diffusion/lsun_bedrooms-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
                # print(f'net={net}')
                # exit()
            elif dataset_name in ['ffhq_ldm']:
                config = OmegaConf.load('./models/ldm/configs/latent-diffusion/ffhq-ldm-vq-4.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=3, guidance_rate=1., guidance_type='uncond', label_dim=0).to(device)
            elif dataset_name in ['ms_coco']:
                assert guidance_type == 'cfg'
                if use_di:
                    config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference_opt.yaml')
                else:
                    config = OmegaConf.load('./models/ldm/configs/stable-diffusion/v1-inference.yaml')
                net = load_ldm_model(config, model_path)
                net = CFGPrecond(net, img_resolution=64, img_channels=4, guidance_rate=guidance_rate, guidance_type='classifier-free', label_dim=True).to(device)
            model_source = 'ldm'
    if net is None:
        raise ValueError("Got wrong settings: check dataset_name and guidance_type!")
    net.eval()

    return net, model_source

@click.command()
@click.option('--dataset_name',            help='Name of the dataset', metavar='STR',                               type=str, required=True)
@click.option('--prompt',                  help='Prompt for Stable Diffusion sampling', metavar='STR',              type=str)
@click.option('--use_fp16',                help='Whether to use mixed precision', metavar='BOOL',                   type=bool, default=False)
@click.option('--outdir',                  help='Where to save the output parameters', metavar='DIR',               type=str)
@click.option('--batch', 'max_batch_size', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=128, show_default=True)
@click.option('--gt_batch', help='Maximum batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--epochs',                  help='Num of training epochs', metavar='INT',                            type=click.IntRange(min=1), default=2, show_default=True)
@click.option('--guidance_type',           help='Guidance type',                                                    type=click.Choice(['cg', 'cfg', 'uncond', None]), default=None, show_default=True)
@click.option('--guidance_rate',           help='Guidance rate',                                                    type=float)
@click.option('--save_interval',           help='interval of epochs to save model', metavar='INT',                  type=click.IntRange(min=1), default=1, show_default=True)
@click.option('--schedule_type',           help='Time discretization schedule', metavar='STR',                      type=click.Choice(['polynomial', 'logsnr', 'time_uniform', 'discrete']), default='polynomial', show_default=True)
@click.option('--schedule_rho',            help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=7, show_default=True)
@click.option('--num_steps',               help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=6, show_default=True)
@click.option('--gt_num_steps',            help='Number of sampling steps', metavar='INT',                          type=click.IntRange(min=1), default=61, show_default=True)
@click.option('--predict_x0',              help='Whether to use data prediction mode', metavar='BOOL',              type=bool, default=True)
@click.option('--lower_order_final',       help='Whether to lower the order at final stages', metavar='BOOL',       type=bool, default=True)
@click.option('--lr',                      help='Learning rate', metavar='FLOAT',                                   type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--lr_min',                      help='Learning rate', metavar='FLOAT',                                   type=click.FloatRange(min=0, min_open=True), default=1e-3, show_default=True)
@click.option('--solver',                  help='Name of the solver', metavar='many solvers',                       type=click.Choice(['euler', 'ipndm', 'ipndm_v', 'heun', 'dpm', 'dpmpp', 'deis', 'unipc']))
@click.option('--gt_solver',               help='Name of the solver', metavar='many solvers',                       type=click.Choice(['euler', 'ipndm', 'ipndm_v', 'heun', 'dpm', 'dpmpp', 'deis', 'unipc']))
@click.option('--sigma_min',               help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True), default=0.002)
@click.option('--sigma_max',               help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=80.)
@click.option('--weight_decay',            help='Lowest noise level  [default: varies]', metavar='FLOAT',           type=click.FloatRange(min=0, min_open=True), default=1e-8)
@click.option('--seeds',                   help='Random seeds (e.g. 1,2,5-10)', metavar='LIST',                     type=parse_int_list, default='0-63', show_default=True)
@click.option('--afs',                     help='Whether to use AFS', metavar='BOOL',                               type=bool, default=False, show_default=True)
# @click.option('--train_idx',               help='Num of training epochs', metavar='STR',                            type=process_numbers, required=True)
# @click.option('--sigma_d',                 help='Time step exponent', metavar='FLOAT',                              type=click.FloatRange(min=0, min_open=True), default=50., show_default=True)
@click.option('--max_order',               help='Max order for solvers', metavar='INT',                             type=click.IntRange(min=1),default=4)
@click.option('--uncond_only',             help='Whether to use AFS', metavar='BOOL',                               type=bool, default=False, show_default=True)
@click.option('--use_res',             help='Whether to use AFS', metavar='BOOL',                               type=bool, default=True, show_default=True)
@click.option('--use_final',             help='Whether to use AFS', metavar='BOOL',                               type=bool, default=True, show_default=True)
@click.option('--eps',            help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=1e-5)
@click.option('--eps_min',            help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=1e-3)

# @click.option('--eps',                     help='Whether to use AFS', metavar='BOOL',                               type=bool, default=False, show_default=True)
@click.option('--di_seed',               help='Max order for solvers', metavar='INT',                             type=click.IntRange(min=0),default=0)
@click.option('--shuffle',             help='Whether to use AFS', metavar='BOOL',                                   type=bool, default=False, show_default=True)
@click.option('--mini_batch',          help=' batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--loss_config',         help='YAML file that defines composite middle/endpoint losses',          type=str, required=True)

def main(dataset_name, max_batch_size, outdir, solver, num_steps, epochs, lr, save_interval, seeds, weight_decay, gt_solver, gt_num_steps, max_order, uncond_only, device=torch.device('cuda'), **solver_kwargs):
    
    assert (gt_num_steps-1)%(num_steps-1)==0 , 'gt_num_steps%num_steps must equal zero!'
    mini_batch=solver_kwargs['mini_batch']
    assert max_batch_size % mini_batch == 0, 'max_batch_size must be divisible by mini_batch!'
    mini_epochs = max_batch_size // mini_batch
    
    num_batches = ((len(seeds) - 1) // (solver_kwargs['gt_batch'] * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]

    if dataset_name in ['ms_coco'] and solver_kwargs['prompt'] is None:
        # Loading MS-COCO captions for FID-30k evaluaion
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        prompt_path, _ = check_file_by_key('prompts')
        prompt_path = "src/prompts/MS-COCO_val2014_heldout_30k_captions.csv"
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)
    
    # if dataset_name in ['ms_coco']:
    #     solver_kwargs['max_order'] = 2
    # else:
    #     solver_kwargs['max_order'] = 3
    if dataset_name in ['ms_coco']:
        solver_kwargs['max_order'] = min(max_order, 2)
    else:
        solver_kwargs['max_order'] = max_order

    # 实例化网络并将其移动到GPU（如果可用）
    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device)
    net.use_fp16 = False
    # print(f'net={net}')
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    # exit()
    if os.path.exists(f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/schedule_type={solver_kwargs["schedule_type"]}/gt_solver={gt_solver}/traj_seeds{seeds[0]}-{seeds[-1]}.pth'):
        traj = torch.load(f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/schedule_type={solver_kwargs["schedule_type"]}/gt_solver={gt_solver}/traj_seeds{seeds[0]}-{seeds[-1]}.pth')
        gt_trajectory=traj["gt_trajectory"]
        denoised_list=traj["denoised_list"]
        latents_list=traj["latents"]
        class_labels_list=traj["class_labels"]
        c_list=traj["c"]
        sample_time=traj["sample_time"]
       
    else:
        gt_t_steps = solver_utils.get_schedule(gt_num_steps, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
                                        schedule_type=solver_kwargs["schedule_type"], schedule_rho=solver_kwargs["schedule_rho"], net=net)
        # print(f'gt_t_steps={gt_t_steps}')
        # exit()
        # print(f'net={net}')
        # exit()
        # Get the time schedule
        solver_kwargs['t_steps'] = gt_t_steps
        solver_kwargs['gt_num_steps'] = gt_num_steps
        # print(f'gt_t_steps={gt_t_steps}')
        # exit()
        
        # Construct solver, 8 solvers are provided
        if gt_solver == 'euler':
            sampler_fn = solvers_traj.euler_sampler
        elif gt_solver == 'heun':
            sampler_fn = solvers_traj.heun_sampler
        elif gt_solver == 'dpm':
            sampler_fn = solvers_traj.dpm_2_sampler
        elif gt_solver == 'ipndm':
            sampler_fn = solvers_traj.ipndm_sampler
        elif gt_solver == 'ipndm_v':
            sampler_fn = solvers_traj.ipndm_v_sampler
        elif gt_solver == 'dpmpp':
            sampler_fn = solvers_traj.dpm_pp_sampler
        elif gt_solver == 'unipc':
            sampler_fn = solvers_traj.unipc_sampler
        elif gt_solver == 'deis':
            sampler_fn = solvers_traj.deis_sampler   # use deis_tab algorithm by default
            # Construct a matrix to store the problematic coefficients for every sampling step
            solver_kwargs['coeff_list'] = solver_utils.get_deis_coeff_list(t_steps, solver_kwargs['max_order'], deis_mode=solver_kwargs["deis_mode"])

        gt_trajectory=[]
        latents_list=[]
        # latents_list_ori=[]
        # sigma_max_di=solver_kwargs['sigma_max_di']
        denoised_list=[]
        class_labels_list=[]
        c_list=[]
        start=time.perf_counter()

        for batch_seeds in tqdm.tqdm(rank_batches, unit='batch', disable=(dist.get_rank() != 0)):
            # torch.distributed.barrier()
            batch_size = len(batch_seeds)
            if batch_size == 0:
                continue

            # Pick latents and labels.
            rnd = StackedRandomGenerator(device, batch_seeds)
            latents = rnd.randn([batch_size, net.img_channels, net.img_resolution, net.img_resolution], device=device)
            # latents_list.append((latents * gt_t_steps[0]).cpu())
            # latents_list.append((latents * sigma_max).cpu())
            latents_list.append(latents.cpu())
            class_labels = c = uc = None
            if net.label_dim:
                if solver_kwargs['model_source'] == 'adm':
                    class_labels = rnd.randint(net.label_dim, size=(batch_size,), device=device)
                    class_labels_list.append(class_labels.detach().cpu())
                elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                    if solver_kwargs['prompt'] is None:
                        prompts = sample_captions[batch_seeds[0]:batch_seeds[-1]+1]
                    else:
                        prompts = [solver_kwargs['prompt'] for i in range(batch_size)]
                    if solver_kwargs['guidance_rate'] != 1.0:
                        uc = net.model.get_learned_conditioning(batch_size * [""])
                    if isinstance(prompts, tuple):
                        prompts = list(prompts)
                    c = net.model.get_learned_conditioning(prompts)
                    c_list.append(c.cpu())
                else:
                    class_labels = torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]
                    class_labels_list.append(class_labels.cpu())
            # Generate images.
            with torch.no_grad():
                if solver_kwargs['model_source'] == 'ldm':
                    with autocast("cuda"):
                        with net.model.ema_scope():
                            latents,denoised = sampler_fn(net, latents, condition=c, unconditional_condition=uc, return_inters=True, **solver_kwargs)
                            denoised_list.append(denoised.cpu())
                            gt_trajectory.append(latents.cpu())
                            # images = net.model.decode_first_stage(latents)
    
                else:
                    images,denoised = sampler_fn(net, latents, class_labels=class_labels, return_inters=True,**solver_kwargs)
                    denoised_list.append(denoised.cpu())
                    gt_trajectory.append(images.cpu())
        # images = net.model.decode_first_stage(images)
        end=time.perf_counter()
        print(f'sample takes {end-start:.6f}s')
        # images = torch.clamp(images/ 2 + 0.5, 0, 1)
        # nrows = int(images.shape[0] ** 0.5)
        # image_grid = make_grid(images, nrows, padding=0)
        # save_image(image_grid,"grid1.png")
        # exit()
        # latents_list = torch.cat(latents_list,dim=0)           
        # gt_trajectory = torch.cat(gt_trajectory,dim=0)
        # print(f'gt_trajectory={len(gt_trajectory),gt_trajectory[0].shape}')
        # print(f'denoised_list={len(denoised_list),denoised_list[0].shape}')
        # exit()
        # print(f'c_list={len(c_list),c_list[0].shape}')
        # exit()
        # time.sleep(10)
        traj={'gt_trajectory':gt_trajectory,'denoised_list':denoised_list,'latents':latents_list,'class_labels':class_labels_list,'c':c_list,'sample_time':f'{end-start:.6f}s'}
        os.makedirs(f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/schedule_type={solver_kwargs["schedule_type"]}/gt_solver={gt_solver}',exist_ok=True)
        torch.save(traj, f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/schedule_type={solver_kwargs["schedule_type"]}/gt_solver={gt_solver}/traj_seeds{seeds[0]}-{seeds[-1]}.pth')
        # exit()
    # print('sample_time=',sample_time)
    # exit()
    del net
    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device, use_di=True)
    net.use_fp16 = False # solver_kwargs['use_fp16']
    # solver_kwargs['use_fp16'] = False
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    # print(f'get_time_emb:{net.model.get_time_emb}')
    # exit()
    t_steps = solver_utils.get_schedule(num_steps, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
                                        schedule_type=solver_kwargs["schedule_type"], schedule_rho=solver_kwargs["schedule_rho"], net=net)
    gt_t_steps = solver_utils.get_schedule(gt_num_steps, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
                                        schedule_type=solver_kwargs["schedule_type"], schedule_rho=solver_kwargs["schedule_rho"], net=net)
    # d_steps = t_steps.clone()
    # for i in range(len(d_steps)):
    #     if d_steps[i]>sigma_d:
    #         d_steps[i]=sigma_d
    #     else:
    #         break
    # t_steps = solver_utils.get_schedule(num_steps, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
    #                                     schedule_type=solver_kwargs["schedule_type"], schedule_rho=2, net=net)
    # print(f't_steps-2={t_steps}')
    # t_steps = solver_utils.get_schedule(num_steps, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
    #                                     schedule_type=solver_kwargs["schedule_type"], schedule_rho=3, net=net)
    # print(f't_steps-3={t_steps}')
    # exit()
    # print(f'd_steps={d_steps}')
    # print(f'net={net}')
    # print(net.sigma_max,net.sigma_min)
    
    solver_kwargs['t_steps'] = t_steps
    # print(f't_steps={t_steps}')
    # print(f'gt_t_steps={gt_t_steps}')
    # exit()
    # if dataset_name == 'lsun_bedroom':
    #     emb_list = net.get_time_emb_deep(t_steps)
    # else:
    #     emb_list = net.get_time_emb(t_steps)
    emb_list = net.get_time_emb(t_steps)
    # exit()
    
    
    use_res=solver_kwargs['use_res']
    if use_res:
        for i in range(len(emb_list)):
            # 将emb_list[i]中的每个元素都设置为0
            emb_list[i] = torch.zeros_like(emb_list[i])
  
        # emb_list = torch.zeros_like(emb_list)
        # emb_list = net.get_time_emb(t_steps)
    # print(f'emb_list={len(emb_list),emb_list[0].shape,emb_list[0][0][:5]}')
    # exit()

    # print(f'emb_list={len(emb_list),emb_list[0].shape}')

    # exit()

    if dataset_name in ["cifar10"]:
        emb_list = [emb.repeat(34, 1) for emb in emb_list] # 15 encoder 19 decoder
    elif dataset_name == "imagenet64":
        emb_list = [emb.repeat(36,1,1) for emb in emb_list] # 15 encoder 21 decoder
    elif  dataset_name == "ffhq":
        emb_list = [emb.repeat(45, 1) for emb in emb_list] # 20 encoder 25 decoder
    elif  dataset_name == "imagenet256":
        emb_list = [emb.repeat(37,1,1) for emb in emb_list] # encoder=18 middle_block=1 decoder=18
    elif  dataset_name == "ms_coco":
        if uncond_only:
            emb_list = [emb.repeat(25,1,1) for emb in emb_list] # encoder=12 middle_block=1 decoder=12
        else:
            emb_list = [emb.repeat(25,2,1) for emb in emb_list] # encoder=12 middle_block=1 decoder=12 
    if dataset_name=='lsun_bedroom_ldm':
        emb_list = [emb.repeat(25,1,1) for emb in emb_list] # encoder=12 middle_block=1 decoder=12 
    elif dataset_name=='lsun_bedroom':
        emb_list = [emb.repeat(37,1) for emb in emb_list] # encoder=18 middle_block=1 decoder=18 

    # emb_list = [nn.Parameter(emb) for emb in emb_list]
    # if dataset_name!='lsun_bedroom':
    #     emb_list = [emb.cpu() for emb in emb_list]
    emb_list = [emb.cpu() for emb in emb_list]
    # print(f'emb_list={len(emb_list),emb_list[0].shape}')
    # print(f'=={emb_list[0][0]==emb_list[0][1]}')
 
    # exit()
    
    if solver == 'euler':
        sampler_fn = solvers_train.euler_sampler
    elif solver == 'heun':
        sampler_fn = solvers_train.heun_sampler
    elif solver == 'dpm':
        sampler_fn = solvers_train.dpm_2_sampler
    elif solver == 'ipndm':
        sampler_fn = solvers_train.ipndm_sampler
        buffer_model=[]
    elif solver == 'ipndm_v':
        sampler_fn = solvers_train.ipndm_v_sampler
    elif solver == 'dpmpp':
        sampler_fn = solvers_train.dpm_pp_sampler
    elif solver == 'unipc':
        sampler_fn = solvers_train.unipc_sampler
    elif solver == 'deis':
        sampler_fn = solvers_train.deis_sampler   # use deis_tab algorithm by default
        # Construct a matrix to store the problematic coefficients for every sampling step
        solver_kwargs['coeff_list'] = solver_utils.get_deis_coeff_list(t_steps, solver_kwargs['max_order'], deis_mode=solver_kwargs["deis_mode"])

    # print(f'gt_trajectory={gt_trajectory[0][0,0,0,0,:5]}')
    # print(f'latents_list={latents_list[0][0,0,0,:5]*80}')
    # print(f'denoised_list={denoised_list[0][0,0,0,:5]}')
    # exit()

    # print(f'Before gt_trajectory shape:{len(gt_trajectory),gt_trajectory[0].shape}')
    # print(f'Before denoised_list shape:{len(denoised_list),denoised_list[0].shape}')
    # print(f'Before latents_list shape:{len(latents_list),latents_list[0].shape}')
    # print(f'Before class_labels_list shape:{len(class_labels_list),class_labels_list[0].shape}')
    # print(f'Before c_list shape:{len(c_list),c_list[0].shape}')
    gt_trajectory=torch.cat(gt_trajectory,dim=1)
    denoised_list=torch.cat(denoised_list,dim=1)
    
    if len(class_labels_list)>0:
        class_labels_list=torch.cat(class_labels_list,dim=0)
    if len(c_list)>0:
        c_list=torch.cat(c_list,dim=0)
    # latents_list=torch.cat(latents_list,dim=0)
    
    latents_list=gt_trajectory[0]
    
    use_final=solver_kwargs['use_final']
    if use_final:
        gt_trajectory=gt_trajectory[-1]
        denoised_list=denoised_list[-1]
        
    # print(f'gt_trajectory={gt_trajectory.shape}')
    # print(f'denoised_list={denoised_list.shape}')
    # print(f'latents_list={latents_list.shape}')
    # images = net.model.decode_first_stage(denoised_list[0].unsqueeze(0))
    # images = torch.clamp(images / 2 + 0.5, 0, 1)
    # save_image(images, "denoised_list.png")
    # exit()
 
    # 定义损失函数和优化器
    criterion = nn.MSELoss() # nn.MSELoss() nn.L2Loss() 
    # criterion = nn.L1Loss()
    inception_feature_extractor = None
    lpips_loss_fn = None
    
    # emb_list=emb_list.cpu()
    # optimizers = [optim.Adam([emb], lr=lr, weight_decay=weight_decay) for emb in emb_list]

    os.makedirs(f'{outdir}/{dataset_name}',exist_ok=True)
    
    net.eval()
    
    import numpy as np
    import random
    # 设置训练随机种子
    di_seed=solver_kwargs['di_seed']
    torch.manual_seed(di_seed)
    np.random.seed(di_seed)
    random.seed(di_seed)
    
    # latents_list.append((latents * gt_t_steps[0]).cpu())
    # print(f'latents_list={len(latents_list),latents_list[0].shape}')
    # exit()
    
    shuffle=solver_kwargs['shuffle']
    loss_config_path=solver_kwargs['loss_config']
    loss_config, loss_config_text = load_loss_config(loss_config_path)
    loss_tag=loss_config['loss_tag']
    loss_hash=loss_config['loss_hash']
    loss_id=loss_config['loss_id']
    snr_gate_config=loss_config.get('snr_gate')
    rollout_config=loss_config.get('rollout', {'enabled': False})
    rollout_enabled=bool(rollout_config.get('enabled', False))
    state_normalization_config=loss_config.get('state_normalization')
    if rollout_enabled and solver != 'euler':
        raise ValueError('rollout training is currently implemented for solver="euler" only')
    if rollout_enabled and use_final:
        raise ValueError('rollout training currently requires use_final=False')
    if rollout_enabled and dataset_name == 'ms_coco':
        raise ValueError('rollout training for ms_coco conditioning is not implemented yet')

    if solver_kwargs['model_source'] == 'ldm':
        writer = SummaryWriter(f'runs/{dataset_name}/schedule_type={solver_kwargs["schedule_type"]}/schedule_rho={solver_kwargs["schedule_rho"]}/{solver}/gt_num_steps{gt_num_steps}-step{num_steps}/use_final={use_final}/use_res={use_res}/di_seed={di_seed}/shuffle={shuffle}/loss_id={loss_id}/train_seed={seeds[0]}-{seeds[-1]}')
    else:
        writer = SummaryWriter(f'runs/{dataset_name}/schedule_type={solver_kwargs["schedule_type"]}/schedule_rho={solver_kwargs["schedule_rho"]}/{solver}/gt_num_steps{gt_num_steps}-step{num_steps}/use_final={use_final}/use_res={use_res}/di_seed={di_seed}/shuffle={shuffle}/loss_id={loss_id}/train_seed={seeds[0]}-{seeds[-1]}')
    writer.add_text('config/loss_tag', loss_tag, 0)
    writer.add_text('config/loss_hash', loss_hash, 0)
    writer.add_text('config/loss_id', loss_id, 0)
    writer.add_text('config/loss_config', f'```\n{loss_config_text}\n```', 0)
    if snr_gate_config is not None:
        writer.add_text('config/snr_gate', str(snr_gate_config), 0)
    writer.add_text('config/rollout', str(rollout_config), 0)
    if state_normalization_config is not None:
        writer.add_text('config/state_normalization', str(state_normalization_config), 0)

    dist.print0("Solver settings:")
    for key, value in solver_kwargs.items():
        if value is None:
            continue
        elif key == 'max_order' and solver in ['euler', 'heun', 'dpm']:
            continue
        elif key in ['predict_x0', 'lower_order_final'] and solver not in ['dpmpp', 'unipc']:
            continue
        elif key in ['variant'] and solver not in ['unipc']:
            continue
        elif key in ['deis_mode'] and solver not in ['deis']:
            continue
        elif key in ['prompt'] and dataset_name not in ['ms_coco']:
            continue
        elif key in ['t_steps', 'coeff_list']:
            continue
        dist.print0(f"\t{key}: {value}")
    dist.print0(f"loss_id={loss_id}")
    
    eps=solver_kwargs['eps']
    
    # print(f'gt_trajectory={len(gt_trajectory),gt_trajectory[0].shape,gt_trajectory[0].device}')
    # exit()
    uc_list=None
    if dataset_name == 'ms_coco':
        uc_list=[]
        uc = net.model.get_learned_conditioning([""])
        for i in range(num_steps):
            uc_list.append(uc.cpu())
            
    
                        
    sum_time=0.
    start=time.perf_counter()
    eps=solver_kwargs['eps']
    
    # Save initial values for exponential decay scheduling
    initial_lr = lr
    initial_eps = eps
    lr_min = solver_kwargs['lr_min']
    lr_max = initial_lr
    
    eps_min = solver_kwargs['eps_min']
    eps_max = initial_eps
    alpha=0.5
    
    patience = 10
    # latents_list=gt_trajectory[14]
    
    for emb_idx,(emb_ori,t_cur,t_next) in enumerate(zip(emb_list[:-1],t_steps[:-1],t_steps[1:])):
        print(f'-----------emb_idx={emb_idx}---------------')
        step_loss_plan = build_step_loss_plan(emb_idx, num_steps, t_steps, loss_config, snr_gate_config)
        if not rollout_enabled:
            step_loss_plan = step_loss_plan[:1]
        planned_loss_items = [item for plan in step_loss_plan for item in plan['loss_items']]

        for plan in step_loss_plan:
            step_idx = plan['step_idx']
            log_prefix = f'rollout_start={emb_idx}/step_idx={step_idx}' if rollout_enabled else f'emb_idx={emb_idx}'
            if plan['snr_gate_value'] is not None:
                writer.add_scalar(f'snr-gate/{log_prefix}/g_i', plan['snr_gate_value'], 0)
                writer.add_scalar(f'snr-gate/{log_prefix}/log_snr', plan['log_snr'], 0)
                dist.print0(f'snr_gate step_idx={step_idx}: log_snr={plan["log_snr"]:.6f}, g_i={plan["snr_gate_value"]:.6f}')
            for idx, (item, runtime_weight) in enumerate(zip(plan['loss_items'], plan['loss_weights'])):
                component_key = loss_component_key(item, idx)
                writer.add_scalar(f'loss-weights/{log_prefix}/{component_key}', runtime_weight, 0)
                dist.print0(f'loss weight step_idx={step_idx} {component_key}: {runtime_weight:.6f}')

        if loss_items_use_type(planned_loss_items, 'inception') and inception_feature_extractor is None:
            dist.print0('Loading Inception-v3 feature extractor for configured loss...')
            inception_feature_extractor = InceptionFeatureExtractor(device)
        if loss_items_use_type(planned_loss_items, 'lpips') and lpips_loss_fn is None:
            dist.print0('Loading LPIPS loss for configured loss...')
            lpips_loss_fn = LPIPSLoss(device)
        
        
        
        lr = lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * emb_idx / (num_steps-1)))
        
        # print(f'num_steps-2={num_steps-2}')
        # exit()
        if emb_idx==num_steps-2:
            eps = 1e-9
            # lr = 1e-3
        else:
            eps = eps_max + (eps_min - eps_max) * ( emb_idx / (num_steps-2))
            # lr=1e-2
        

        
        print(f'Current lr: {lr:.8f}, Current eps: {eps:.8f}')

        if rollout_enabled:
            train_embs = [
                nn.Parameter(emb_list[step_idx].clone().to(device))
                for step_idx in range(emb_idx, num_steps - 1)
            ]
            emb = train_embs[0]
        else:
            train_embs = None
            emb=emb_ori.clone().to(device)
            emb=nn.Parameter(emb)
        
        if dataset_name == 'ms_coco':
            uc = uc_list[emb_idx].to(device)
            uc.requires_grad_(True)
        
        if dataset_name == 'ms_coco':
            optimizer=optim.Adam([emb, uc], lr=lr, weight_decay=weight_decay)
        elif rollout_enabled:
            optimizer=optim.Adam(train_embs, lr=lr, weight_decay=weight_decay)
        else:
            optimizer=optim.Adam([emb], lr=lr, weight_decay=weight_decay)
        
        # Note: With Adam optimizer and few epochs per emb, ExponentialLR scheduler
        # is not very effective. The inter-emb decay is more important.
        
        # writer.add_scalar('lr', lr, emb_idx)
        writer.add_scalar('eps', eps, emb_idx)
        # writer.add_text('num_steps[emb_idx+1]', str(t_steps[emb_idx+1]), emb_idx)
        # writer.add_text('gt_num_steps[(emb_idx+1)*((gt_num_steps-1)//(num_steps-1))]', str(gt_t_steps[(emb_idx+1)*((gt_num_steps-1)//(num_steps-1))]), emb_idx)
        
        if rollout_enabled:
            loss_targets = get_rollout_training_targets(emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps)
        else:
            loss_targets = get_training_targets(emb_idx, use_final, gt_trajectory, denoised_list, gt_num_steps, num_steps)
        if use_final:
            dataset=DI_Dataset(latents_list,loss_targets,class_labels_list,c_list)
            dataLoader=DataLoader(dataset,batch_size=mini_batch,shuffle=shuffle,drop_last=False)
        else:
            dataset=DI_Dataset(latents_list,loss_targets,class_labels_list,c_list)
            dataLoader=DataLoader(dataset,batch_size=mini_batch,shuffle=shuffle,drop_last=False)
  
        
        no_improve = 0
        for epoch in range(epochs):
            sum_loss=0.

            actual_lr = get_adam_step_size_stats(optimizer)
            if actual_lr is None:
                actual_lr = initial_lr
            writer.add_scalar(f'lr/emb_idx={emb_idx}', actual_lr, epoch)
            
            optimizer.zero_grad()
            denoised_temp=[]
            x_0_temp=[]
            sum_component_loss = {}
            for plan in step_loss_plan:
                prefix = f'step={plan["step_idx"]}_{plan["stage"]}/' if rollout_enabled else ''
                for idx, item in enumerate(plan['loss_items']):
                    sum_component_loss[f'{prefix}{loss_component_key(item, idx)}'] = 0.
            for batch_idx, (x_cur, x_0 ,class_labels, c) in enumerate(dataLoader):
                # 获取输入数据
                x_cur = x_cur.to(device)
                if isinstance(x_0, dict):
                    x_0 = {key: value.to(device) for key, value in x_0.items()}
                else:
                    x_0 = x_0.to(device)

                if uc_list is None:
                    uc = None
                # class_labels = c = uc = None
                if class_labels==[]:
                    class_labels=None
                if c==[]:
                    c=None

                if net.label_dim:
                    if solver_kwargs['model_source'] == 'adm':
                        class_labels = class_labels.to(device)
                    elif solver_kwargs['model_source'] == 'ldm' and dataset_name == 'ms_coco':
                        c=c.to(device)
                        # print(f'uc={uc.shape}')
                        if uc is not None:
                            uc=uc.expand(mini_batch, -1, -1)
                        # print(f'uc={uc.shape}')
                        # exit()
                        # c = net.model.get_learned_conditioning(prompts)
                    else:
                        class_labels = class_labels.to(device)  # torch.eye(net.label_dim, device=device)[rnd.randint(net.label_dim, size=[batch_size], device=device)]

                if rollout_enabled:
                    loss = None
                    rollout_x = x_cur
                    denoised = None
                    x_next = None
                    for plan in step_loss_plan:
                        step_idx = plan['step_idx']
                        step_emb = train_embs[step_idx - emb_idx]
                        step_t_cur = t_steps[step_idx]
                        step_t_next = t_steps[step_idx + 1]
                        denoised = get_denoised(net, rollout_x, step_t_cur, step_emb, use_res, class_labels=class_labels, condition=c, unconditional_condition=uc)
                        d_cur = (rollout_x - denoised) / step_t_cur
                        x_next = rollout_x + (step_t_next - step_t_cur) * d_cur
                        predictions = get_predictions_for_loss(x_next, denoised)
                        step_targets = get_rollout_step_targets(x_0, step_idx)
                        step_loss, component_losses = compute_composite_loss(
                            plan['loss_items'],
                            plan['loss_weights'],
                            criterion,
                            inception_feature_extractor,
                            lpips_loss_fn,
                            predictions,
                            step_targets,
                            net,
                            solver_kwargs['model_source'],
                            step_sigma=step_t_next,
                            state_normalization_config=state_normalization_config,
                        )
                        loss = step_loss if loss is None else loss + step_loss
                        prefix = f'step={step_idx}_{plan["stage"]}/'
                        for key, value in component_losses.items():
                            sum_component_loss[f'{prefix}{key}'] += value.item()
                        rollout_x = x_next
                else:
                    # Generate images.                
                    denoised = get_denoised(net, x_cur, t_cur, emb, use_res, class_labels=class_labels, condition=c, unconditional_condition=uc)
                    d_cur = (x_cur - denoised) / t_cur
                    x_next = x_cur + (t_next - t_cur) * d_cur
                    predictions = get_predictions_for_loss(x_next, denoised)
                    
                    loss, component_losses = compute_composite_loss(
                        step_loss_plan[0]['loss_items'],
                        step_loss_plan[0]['loss_weights'],
                        criterion,
                        inception_feature_extractor,
                        lpips_loss_fn,
                        predictions,
                        x_0,
                        net,
                        solver_kwargs['model_source'],
                        step_sigma=t_next,
                        state_normalization_config=state_normalization_config,
                    )
                    for key, value in component_losses.items():
                        sum_component_loss[key] += value.item()
                loss.backward()
                sum_loss+=loss.item()
                
                if batch_idx>=len(dataLoader)-mini_epochs:
                    if rollout_enabled:
                        display_target = get_rollout_step_targets(x_0, step_loss_plan[-1]['step_idx'])['state']
                    else:
                        display_target = x_0['state'] if isinstance(x_0, dict) else x_0
                    if solver_kwargs['model_source'] == 'ldm':
                        denoised = net.model.decode_first_stage(denoised)
                        x_0=net.model.decode_first_stage(display_target)
                    elif isinstance(x_0, dict):
                        x_0 = display_target
                    denoised_temp.append(denoised.detach().cpu())
                    x_0_temp.append(x_0.detach().cpu())
                    
                    
                if batch_idx % mini_epochs==mini_epochs-1:    
                    
                    optimizer.step()
                    optimizer.zero_grad()
                    
                    
            sum_loss=sum_loss/(len(latents_list)/mini_batch)  
            for key in sum_component_loss:
                sum_component_loss[key]=sum_component_loss[key]/(len(latents_list)/mini_batch)
            
            denoised_temp=torch.cat(denoised_temp,dim=0)
            x_0_temp=torch.cat(x_0_temp,dim=0)
            
            writer.add_scalar(f'train-loss/emb_idx={emb_idx}', sum_loss, epoch) 
            for key, value in sum_component_loss.items():
                writer.add_scalar(f'train-loss-components/emb_idx={emb_idx}/{key}', value, epoch)
            
            image=torch.clamp(denoised_temp / 2 + 0.5, 0, 1)
            # image = transforms.Resize((64, 64))(image)
            
            writer.add_images(f'train-x_cur/emb_idx={emb_idx}', image, global_step=epoch)
            
            image=torch.clamp(x_0_temp / 2 + 0.5, 0, 1)
            # image = transforms.Resize((64, 64))(image)
            writer.add_images(f'train-x_0/emb_idx={emb_idx}', image, global_step=epoch) 
            
            writer.flush()
            
            # if epoch==0:
            #     print(f'sum_loss:{sum_loss:.8f},epoch:{epoch}')

            if epoch>0:
                delta = (last_sum_loss - sum_loss) / last_sum_loss
     
                writer.add_scalars(f'early_stop/emb_idx={emb_idx}', {'delta': delta}, epoch)
                writer.add_scalars(f'early_stop/emb_idx={emb_idx}', {'eps': eps}, epoch)
                
                print(f'last_sum_loss:{last_sum_loss:.8f},sum_loss:{sum_loss:.8f},delta:{delta:.8f},epoch:{epoch}')
                
                if dataset_name == 'ms_coco':
                    print(f'uc={uc.mean()}')
                # delta = (last_sum_loss - sum_loss) / last_sum_loss
                if (delta<eps and sum_loss<last_sum_loss) or epoch==epochs-1:
                    no_improve += 1
                    # print(f'last_sum_loss:{last_sum_loss:.8f},sum_loss:{sum_loss:.8f},loss distance:{abs((last_sum_loss-sum_loss)):.8f},epoch:{epoch}')
                    # exit()
                    c = class_labels = None
                    if emb_idx==len(emb_list)-1:
                        break
                    if no_improve >= patience or epoch==epochs-1:
                        for batch_idx, (x_cur, _, class_labels, c) in enumerate(dataLoader):
                            x_cur = x_cur.to(device)
                            if len(class_labels_list)>0:
                                class_labels = class_labels.to(device)
                            if len(c_list)>0:
                                c = c.to(device)
                                
                            # torch.cuda.empty_cache()
                            with torch.no_grad():
                                denoised = get_denoised(net, x_cur, t_cur, emb, use_res, class_labels=class_labels, condition=c, unconditional_condition=uc)
                            # x_next = sampler_fn(x_cur, denoised, t_cur, t_next)
                            d_cur = (x_cur - denoised) / t_cur
                            x_next = x_cur + (t_next - t_cur) * d_cur
                            # latents_list[i]=x_next.detach().cpu()
                            latents_list[batch_idx*mini_batch:batch_idx*mini_batch+mini_batch]=x_next.detach().cpu()
                            # del x_cur, x_0, d_cur
 
                        break
                else:
                    no_improve = 0
                # if epoch>3:
                #     exit()
            last_sum_loss=sum_loss
            
            # exit()


        # exit()        
        
        if rollout_enabled:
            for offset, trained_emb in enumerate(train_embs):
                emb_list[emb_idx + offset]=trained_emb.detach().cpu()
        else:
            emb_list[emb_idx]=emb.detach().cpu()
        if dataset_name == 'ms_coco':
            uc_list[emb_idx]=uc.detach().cpu()
        # print(f'emb_list[emb_idx]={emb_list[emb_idx].shape,emb_list[emb_idx].dtype}')
        # print(f'emb={emb.shape,emb.dtype}')
        # exit()
    end=time.perf_counter()
    print(f'training takes {end-start:.6f}')
    # exit()

    # x_next = net.model.decode_first_stage(x_next)
    # images = torch.clamp(x_next / 2 + 0.5, 0, 1)
    # nrows = int(images.shape[0] ** 0.5)
    # image_grid = make_grid(images, nrows, padding=0)
    # save_image(image_grid,"grid3.png")
    # exit()

    save_dir = f'{outdir}/{dataset_name}/{solver}/step{num_steps}/gt_num_steps{gt_num_steps}/schedule_type={solver_kwargs["schedule_type"]}/seed{seeds[0]}-{seeds[-1]}/lr={initial_lr}/use_final={use_final}/use_res={use_res}/di_seed={di_seed}/shuffle={shuffle}/loss_id={loss_id}'
    os.makedirs(save_dir,exist_ok=True)
    # 保存最终训练后的模型
    result={
        'emb_list':emb_list,
        'model_list':None,
        'time':f'{end-start:.6f}',
        'loss_config':loss_config,
        'loss_config_text':loss_config_text,
        'loss_tag':loss_tag,
        'loss_hash':loss_hash,
        'loss_id':loss_id,
    }
    if dataset_name == 'ms_coco':
        result['uc_list']=uc_list
    with open(os.path.join(save_dir, 'loss_config.yaml'), 'w') as f:
        f.write(loss_config_text)
    torch.save(result, f'{save_dir}/{dataset_name}_{num_steps}_epoch_{epochs}_seed{seeds[0]}-{seeds[-1]}.pth')




if __name__ == "__main__":
    main()