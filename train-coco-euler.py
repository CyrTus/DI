import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
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
@click.option('--denoised_type',          help='Type of denoised', metavar='STR',                                  type=click.Choice(['linear', 'quad']))
@click.option('--eps',            help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=1e-5)
@click.option('--eps_min',            help='Highest noise level  [default: varies]', metavar='FLOAT',          type=click.FloatRange(min=0, min_open=True), default=1e-3)

# @click.option('--eps',                     help='Whether to use AFS', metavar='BOOL',                               type=bool, default=False, show_default=True)
@click.option('--di_seed',               help='Max order for solvers', metavar='INT',                             type=click.IntRange(min=0),default=0)
@click.option('--train_model',             help='Whether to use AFS', metavar='BOOL',                               type=bool, default=False, show_default=True)
@click.option('--shuffle',             help='Whether to use AFS', metavar='BOOL',                                   type=bool, default=False, show_default=True)
@click.option('--mini_batch',          help=' batch size', metavar='INT',                                type=click.IntRange(min=1), default=64, show_default=True)
@click.option('--use_dropout',         help='Whether to use dropout', metavar='BOOL',                           type=bool, default=False, show_default=True)

def main(dataset_name, max_batch_size, outdir, solver, num_steps, epochs, lr, save_interval, seeds, weight_decay, gt_solver, gt_num_steps, max_order, uncond_only, device=torch.device('cuda'), **solver_kwargs):
    
    assert (gt_num_steps-1)%(num_steps-1)==0 , 'gt_num_steps%num_steps must equal zero!'
    mini_batch=solver_kwargs['mini_batch']
    assert max_batch_size % mini_batch == 0, 'max_batch_size must be divisible by mini_batch!'
    mini_epochs = max_batch_size // mini_batch
    
    num_batches = ((len(seeds) - 1) // (solver_kwargs['gt_batch'] * dist.get_world_size()) + 1) * dist.get_world_size()
    all_batches = torch.as_tensor(seeds).tensor_split(num_batches)
    rank_batches = all_batches[dist.get_rank() :: dist.get_world_size()]
    
    solver_kwargs['max_order']=max_order
    base_seed=seeds[0]

    if dataset_name in ['ms_coco'] and solver_kwargs['prompt'] is None:
        # Loading MS-COCO captions for FID-30k evaluaion
        # We use the selected 30k captions from https://github.com/boomb0om/text2image-benchmark
        # prompt_path, _ = check_file_by_key('prompts')
        prompt_path = "src/prompts/MS-COCO_val2014_30k_captions.csv"
        sample_captions = []
        with open(prompt_path, 'r') as file:
            reader = csv.DictReader(file)
            for row in reader:
                text = row['text']
                sample_captions.append(text)
    
    # print(f'sample_captions={len(sample_captions)},sample_captions={sample_captions[:10]}')
    # exit()

    # 实例化网络并将其移动到GPU（如果可用）
    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device)
    net.use_fp16 = False
    # print(f'net={net}')
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    # exit()
    if os.path.exists(f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/gt_solver={gt_solver}/traj_seeds{seeds[0]}-{seeds[-1]}.pth'):
        traj = torch.load(f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/gt_solver={gt_solver}/traj_seeds{seeds[0]}-{seeds[-1]}.pth')
        gt_trajectory=traj["gt_trajectory"]
        denoised_list=traj["denoised_list"]
        latents_list=traj["latents"]
        class_labels_list=traj["class_labels"]
        c_list=traj["c"]
        # batch_seeds=traj["seeds"]
        sample_time=traj["sample_time"]
       
        # print(f'gt_trajectory={len(gt_trajectory),gt_trajectory[0].shape}')
        # print(f'denoised_list={len(denoised_list),denoised_list[0].shape}')
        # print(f'latents_list={len(latents_list),latents_list[0].shape}')
        # print(f'class_labels_list={len(class_labels_list),class_labels_list[0].shape}')
        # print(f'class_labels_list={class_labels_list}')
        # print(f'c_list={c_list}')
        # # print(f'batch_seeds={batch_seeds}')
        # print(f'sample_time={sample_time}')
        # exit()
        # images=gt_trajectory[0][-1]
        # images = torch.clamp(images/ 2 + 0.5, 0, 1)
        # nrows = int(images.shape[0] ** 0.5)
        # image_grid = make_grid(images, nrows, padding=0)
        # save_image(image_grid,"grid1.png")
        # exit()
    else:
        gt_t_steps = solver_utils.get_schedule(gt_num_steps, solver_kwargs['sigma_min'], solver_kwargs['sigma_max'], device=device, \
                                        schedule_type=solver_kwargs["schedule_type"], schedule_rho=solver_kwargs["schedule_rho"], net=net)
    
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
                        prompts = sample_captions[batch_seeds[0]-base_seed:batch_seeds[-1]+1-base_seed]
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
        os.makedirs(f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/gt_solver={gt_solver}',exist_ok=True)
        torch.save(traj, f'traj/{dataset_name}/gt_num_steps-{gt_num_steps}/gt_solver={gt_solver}/traj_seeds{seeds[0]}-{seeds[-1]}.pth')
        # exit()
    # print('sample_time=',sample_time)
    # exit()
    del net
    net, solver_kwargs['model_source'] = create_model(dataset_name, solver_kwargs['guidance_type'], solver_kwargs['guidance_rate'], device, use_di=True)
    net.use_fp16 = False # solver_kwargs['use_fp16']
    # solver_kwargs['use_fp16'] = False
    solver_kwargs['sigma_min'] = net.sigma_min
    solver_kwargs['sigma_max'] = net.sigma_max
    # print(f'sigma_min={solver_kwargs["sigma_min"]}, sigma_max={solver_kwargs["sigma_max"]}')
    # exit()
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
    
    # print(f'denoised_list={denoised_list.shape}')
    # print(f'latents_list={latents_list.shape}')
    # images = net.model.decode_first_stage(denoised_list[0].unsqueeze(0))
    # images = torch.clamp(images / 2 + 0.5, 0, 1)
    # save_image(images, "denoised_list.png")

    # exit()
    # else:
    #     denoised_idx=
    # latents_list=gt_trajectory
    # print(f'After gt_trajectory shape:{len(gt_trajectory),gt_trajectory[0].shape}')
    # print(f'After denoised_list shape:{len(denoised_list),denoised_list[0].shape}')
    # print(f'After latents_list shape:{latents_list.shape,latents_list[0,0,0,0]*80.0}')
    # print(f'latents_list=gt_trajectory[0]{gt_trajectory[0][0,0,0,0]==latents_list[0,0,0,0]*80.0,latents_list[0,0,0,0]*80.0,gt_trajectory[0][0,0,0,0]}')
    # print(f'After class_labels_list shape:{len(class_labels_list),class_labels_list[0].shape}')
    # print(f'After c_list shape:{len(c_list),c_list[0].shape}')
    # exit()

    
    
    # 定义损失函数和优化器
    criterion = nn.MSELoss() # nn.MSELoss() nn.L2Loss() 
    # criterion = nn.L1Loss()
    
    # emb_list=emb_list.cpu()
    # optimizers = [optim.Adam([emb], lr=lr, weight_decay=weight_decay) for emb in emb_list]

    os.makedirs(f'{outdir}/{dataset_name}',exist_ok=True)
    
    use_dropout=solver_kwargs['use_dropout']
    if use_dropout:
        net.train()
    else:
        net.eval()
    # net.eval()
    train_model=solver_kwargs['train_model']
    if train_model:
        model_list=[]
        if use_dropout:
            net.train().requires_grad_(True)
        else:
            net.eval().requires_grad_(True)
    # net.train().requires_grad_(True)
    
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
    denoised_type=solver_kwargs['denoised_type']
    if solver_kwargs['model_source'] == 'ldm':
        writer = SummaryWriter(f'runs/{dataset_name}/schedule_rho={solver_kwargs["schedule_rho"]}/{solver}/gt_num_steps{gt_num_steps}-step{num_steps}/di_seed={di_seed}/train_seed={seeds[0]}-{seeds[-1]}')
    else:
        writer = SummaryWriter(f'runs/{dataset_name}/{solver}/gt_num_steps{gt_num_steps}-step{num_steps}/di_seed={di_seed}/train_seed={seeds[0]}-{seeds[-1]}')

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
    
    eps=solver_kwargs['eps']
    
    denoised_seq=[]
    # print(f'denoised_type={denoised_type}')
    if denoised_type in ["quad","exp"]:
        # print('in1')
        if denoised_type=="quad":
            # print('in2')
            denoised_seq=square_interpolation(gt_num_steps,num_steps)
        elif denoised_type=="exp":
            # print('in3')
            denoised_seq=exponential_interpolation(gt_num_steps,num_steps,power=3)
        # denoised_seq=[i-1 for i in denoised_seq]
        # denoised_seq=denoised_seq-1
        denoised_seq.reverse()
        denoised_seq=denoised_seq[1:]
    # print(f'denoised_seq={denoised_seq}')
    # exit()
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

        emb=emb_ori.clone().to(device)
        emb=nn.Parameter(emb)
        
        if dataset_name == 'ms_coco':
            uc = uc_list[emb_idx].to(device)
            uc.requires_grad_(True)
        
        if train_model:
            optimizer=optim.Adam([emb], lr=lr, weight_decay=weight_decay)
            optimizer_model=optim.Adam(net.parameters(), lr=5e-5, weight_decay=weight_decay)
        else:
            if dataset_name == 'ms_coco':
                optimizer=optim.Adam([emb, uc], lr=lr, weight_decay=weight_decay)
            else:
                optimizer=optim.Adam([emb], lr=lr, weight_decay=weight_decay)
            optimizer_model=None
        
        # Note: With Adam optimizer and few epochs per emb, ExponentialLR scheduler
        # is not very effective. The inter-emb decay is more important.
        
        # writer.add_scalar('lr', lr, emb_idx)
        writer.add_scalar('eps', eps, emb_idx)
        # writer.add_text('num_steps[emb_idx+1]', str(t_steps[emb_idx+1]), emb_idx)
        # writer.add_text('gt_num_steps[(emb_idx+1)*((gt_num_steps-1)//(num_steps-1))]', str(gt_t_steps[(emb_idx+1)*((gt_num_steps-1)//(num_steps-1))]), emb_idx)
        
        if use_final:
            dataset=DI_Dataset(latents_list,gt_trajectory,class_labels_list,c_list)
            dataLoader=DataLoader(dataset,batch_size=mini_batch,shuffle=shuffle,drop_last=False)
        else:
            
            if denoised_type == 'linear':
                # print(f'idx={(emb_idx+1)*((gt_num_steps-1)//(num_steps-1))}')
                dataset=DI_Dataset(latents_list,gt_trajectory[(emb_idx+1)*((gt_num_steps-1)//(num_steps-1))],class_labels_list,c_list)
            dataLoader=DataLoader(dataset,batch_size=mini_batch,shuffle=shuffle,drop_last=False)
  
        
        no_improve = 0
        for epoch in range(epochs):
            sum_loss=0.
            if optimizer_model is not None:
                    optimizer_model.zero_grad()
                    
            actual_lr = get_adam_step_size_stats(optimizer)
            if actual_lr is None:
                actual_lr = initial_lr
            writer.add_scalar(f'lr/emb_idx={emb_idx}', actual_lr, epoch)
            
            optimizer.zero_grad()
            denoised_temp=[]
            x_0_temp=[]
            for batch_idx, (x_cur, x_0 ,class_labels, c) in enumerate(dataLoader):
                # 获取输入数据
                x_cur , x_0 = x_cur.to(device), x_0.to(device)
                # print(f'x_cur={x_cur.shape},{x_cur[0,0,0,:5]}')
                # print(f'x_0={x_0.shape},{x_0[0,0,0,:5]}')
                # print(f't_cur={t_cur},{t_next}')
                # print(f'class_labels={torch.argmax(class_labels,dim=1)}')
                # print(f'c={c}')
                # exit()
                # exit()
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

                # Generate images.
                # # with torch.no_grad():
                # if solver_kwargs['model_source'] == 'ldm':
                #     # with autocast("cuda"):
                #     with net.model.ema_scope():
                #         # images = sampler_fn(net, x_cur, condition=c, unconditional_condition=uc, **solver_kwargs)
                #         denoised = get_denoised(net, x_cur, t_cur, emb, use_res, class_labels=class_labels, condition=c, unconditional_condition=uc)
                #         # denoised = net.model.differentiable_decode_first_stage(denoised)
                #         # x_0=net.model.differentiable_decode_first_stage(x_0)
                #             # print(f'images={images.shape}')
                # else:
                # emb_fwd = emb.to(net.model.dtype if hasattr(net, "model") else net.dtype)
                # denoised = get_denoised(net, x_cur, t_cur, emb_fwd, use_res, class_labels=class_labels, condition=c, unconditional_condition=uc)

 
                denoised = get_denoised(net, x_cur, t_cur, emb, use_res, class_labels=class_labels, condition=c, unconditional_condition=uc)
                d_cur = (x_cur - denoised) / t_cur
                x_next = x_cur + (t_next - t_cur) * d_cur
                
                loss = criterion(x_next, x_0)
                loss.backward()
                sum_loss+=loss.item()

                
                if batch_idx>=len(dataLoader)-mini_epochs:
                    if solver_kwargs['model_source'] == 'ldm':
                        denoised = net.model.decode_first_stage(denoised)
                        x_0=net.model.decode_first_stage(x_0)
                    denoised_temp.append(denoised.detach().cpu())
                    x_0_temp.append(x_0.detach().cpu())
                    
                    
                # print(f'denoised={denoised.shape}')
                # print(f'x_0={x_0.shape}')
                # net_copy = copy.deepcopy(net)
                # print(f'before')
                # print(f'net_copy={net_copy.model.enc["32x32_conv"].weight[:1,:1,:1,:5]}')   
                # print(f'net={net.model.enc["32x32_conv"].weight[:1,:1,:1,:5]}')
                # print(f'net_copy={net_copy.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]}')
                # print(f'net={net.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]}')
                # print(f'net=net_copy:{net_copy.model.enc["32x32_conv"].weight[:1,:1,:1,:5]==net.model.enc["32x32_conv"].weight[:1,:1,:1,:5]}')
                # print(f'net=net_copy:{net_copy.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]==net.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]}')
                if batch_idx % mini_epochs==mini_epochs-1:    
                    
                    optimizer.step()
                    if optimizer_model is not None:
                        optimizer_model.step()
                
                    if optimizer_model is not None:
                        optimizer_model.zero_grad()
                    optimizer.zero_grad()
                    
                    # denoised_temp=torch.cat(denoised_temp,dim=0)
                    # x_0_temp=torch.cat(x_0_temp,dim=0)
                    
                    
                    # print(f'denoised_temp={denoised_temp.shape}')
                    # print(f'x_0_temp={x_0_temp.shape}')
                    # exit()
                    # print(f'after')
                    # print(f'net_copy={net_copy.model.enc["32x32_conv"].weight[:1,:1,:1,:5]}')
                    # print(f'net={net.model.enc["32x32_conv"].weight[:1,:1,:1,:5]}')
                    # print(f'net_copy={net_copy.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]}')
                    # print(f'net={net.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]}')
                    
                    # print(f'net=net_copy:{net_copy.model.enc["32x32_conv"].weight[:1,:1,:1,:5]==net.model.enc["32x32_conv"].weight[:1,:1,:1,:5]}')
                    # print(f'net=net_copy:{net_copy.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]==net.model.enc["32x32_block0"].conv0.weight[:1,:1,:1,:5]}')
                    # exit()
                    
                    
            sum_loss=sum_loss/(len(latents_list)/mini_batch)  
            
            denoised_temp=torch.cat(denoised_temp,dim=0)
            x_0_temp=torch.cat(x_0_temp,dim=0)
            
            writer.add_scalar(f'train-loss/emb_idx={emb_idx}', sum_loss, epoch) 
            
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
        
        emb_list[emb_idx]=emb.detach().cpu()
        if dataset_name == 'ms_coco':
            uc_list[emb_idx]=uc.detach().cpu()
        # print(f'emb_list[emb_idx]={emb_list[emb_idx].shape,emb_list[emb_idx].dtype}')
        # print(f'emb={emb.shape,emb.dtype}')
        # exit()
        # lr and eps are now dynamically adjusted at the start of each loop iteration
        # No need to manually decay them here
        if train_model:
            model_list.append(net.state_dict())
    end=time.perf_counter()
    print(f'training takes {end-start:.6f}')
    # exit()

    # x_next = net.model.decode_first_stage(x_next)
    # images = torch.clamp(x_next / 2 + 0.5, 0, 1)
    # nrows = int(images.shape[0] ** 0.5)
    # image_grid = make_grid(images, nrows, padding=0)
    # save_image(image_grid,"grid3.png")
    # exit()

    os.makedirs(f'{outdir}/{dataset_name}/{solver}/step{num_steps}/gt_num_steps{gt_num_steps}/seed{seeds[0]}-{seeds[-1]}/lr={initial_lr}/use_final={use_final}/use_res={use_res}/train_model={train_model}/di_seed={di_seed}/shuffle={shuffle}/use_dropout={use_dropout}/denoised_type={denoised_type}',exist_ok=True)
    # 保存最终训练后的模型
    if train_model:
        result={'emb_list':emb_list, 'model_list':model_list, 'time':f'{end-start:.6f}'}
        if dataset_name == 'ms_coco':
            result['uc_list']=uc_list
    else:
        result={'emb_list':emb_list, 'model_list':None, 'time':f'{end-start:.6f}'}
        if dataset_name == 'ms_coco':
            result['uc_list']=uc_list
    torch.save(result, f'{outdir}/{dataset_name}/{solver}/step{num_steps}/gt_num_steps{gt_num_steps}/seed{seeds[0]}-{seeds[-1]}/lr={initial_lr}/use_final={use_final}/use_res={use_res}/train_model={train_model}/di_seed={di_seed}/shuffle={shuffle}/use_dropout={use_dropout}/denoised_type={denoised_type}/{dataset_name}_{num_steps}_epoch_{epochs}_seed{seeds[0]}-{seeds[-1]}.pth')




if __name__ == "__main__":
    main()