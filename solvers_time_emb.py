# Implementation of various ODE solvers for diffusion models.

import torch
from solver_utils import *
# from time_emb_utils import *
import pickle
from emb_utils import *

#----------------------------------------------------------------------------
# Get the denoised output from the pre-trained diffusion models.

def get_denoised(net, x, t, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):       # models from LDM and Stable Diffusion
        denoised = net(x, t, condition=condition, unconditional_condition=unconditional_condition)
    else:
        denoised = net(x, t, class_labels=class_labels)
    return denoised

def get_denoised_time_embs(net, x, t, time_emb_list, use_res, class_labels=None, condition=None, unconditional_condition=None):
    if hasattr(net, 'guidance_type'):       # models from LDM and Stable Diffusion
        # print(f'condition={condition.shape}')
        denoised = net(x, t, time_emb_list, use_res=use_res, condition=condition, unconditional_condition=unconditional_condition)
    else:
        # if dataset_name in ["cifar10" , "ffhq" , "imagenet64"]:
        #     denoised = net.edm_forward(x, t, w, dataset_name, time_emb_list, class_labels=class_labels)
        # else:
        #     # print(f'class_labels={class_labels.shape,class_labels}')
        #     denoised = net.CGPrecond_forward(x, t, time_emb_list, class_labels=class_labels)
        # print(f'time_emb_list={time_emb_list.shape}')
        # exit()
        denoised = net(x, t, time_emb_list, use_res=use_res, class_labels=class_labels)
        # denoised = net(x, t, time_emb_list)
    return denoised

#----------------------------------------------------------------------------

@torch.no_grad()
def euler_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    t_steps=None,
    **kwargs
):  
    """
    Euler sampler (equivalent to the DDIM sampler: https://arxiv.org/abs/2010.02502).

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    emb_data=kwargs['emb_data']
    use_res=kwargs['use_res']
    # last_n=kwargs['last_n']
    # train_model=kwargs['train_model']
    # if train_model:
    #     model_list=kwargs['model_list']
        
    dataset_name=kwargs['dataset_name']
    if dataset_name in ['ms_coco']:
        uc_list=kwargs['uc_list']
    
    
    # d_steps=kwargs['d_steps']
    # print(f't_steps={t_steps}')
    # print(f'd_steps={d_steps}')
    # exit()
    # print(f'afs={afs}')
    # exit()
    time_emb_list=emb_data

    selected_time_emb_list=time_emb_list
    # print(f'selected_time_emb_list={len(selected_time_emb_list),selected_time_emb_list[0].shape}')
    # exit()

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    inters_eps = []
    
    uc_condition=None
    for i, (t_cur, t_next, time_emb_list) in enumerate(zip(t_steps[:-1], t_steps[1:], selected_time_emb_list)):   # 0, ..., N-1
        x_cur = x_next
        # print(f'x_cur={x_cur.shape}')

        # if train_model:
        #     net.load_state_dict(model_list[i])
        if dataset_name in ['ms_coco']:
            uc_condition=uc_list[i].to(latents.device)
            # print(f'uc_condition={uc_condition.shape}')
            # uc_condition=uc_condition.repeat(latents.shape[0]//2, 1, 1)
            uc_condition=uc_condition.repeat(latents.shape[0], 1, 1)
            
            
        # print(f'uc_condition={uc_condition.shape}')
        # exit()
        # Euler step.
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            # print(f'class_labels={class_labels}')
            # exit()
            time_emb_list=time_emb_list.to('cuda')
            denoised = get_denoised_time_embs(net, x_cur, t_cur, time_emb_list, use_res, class_labels=class_labels, condition=condition, unconditional_condition=uc_condition)

            # if num_steps-i-1<=last_n:
            #     time_emb_list=time_emb_list.to('cuda')
            #     denoised = get_denoised_time_embs(net, x_cur, t_cur, time_emb_list, use_res, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            # else:
            #     denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
    
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    
    if return_inters:
        if return_eps:
            return torch.cat(inters, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def heun_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False,
    return_eps=False, 
    t_steps=None, 
    **kwargs
):
    """
    Heun's second sampler. Introduced in EDM: https://arxiv.org/abs/2206.00364.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    before=kwargs['before']
    after=kwargs['after']
    emb_data=kwargs['emb_data']
    time_emb_list=emb_data['time_emb_list']
    t_steps_emb=emb_data['t_steps']
    w_start=kwargs['w_start']
    w_end=kwargs['w_end']
    
    # print(f'time_emb_list={time_emb_list.shape}')
    # print(f't_steps={t_steps_emb}')
    # print(f't_steps={t_steps}')
    # print(f't_steps_emb={t_steps_emb}')
    span=(len(t_steps_emb)-1)//(len(t_steps)-1)
    # span=int(span)
    # print(f'span={span}')
    # print((len(t_steps_emb)-1)/span)
    # exit()
    selected_time_emb_list=[]
    for i in range((len(t_steps_emb)-1)//span):
        selected_time_emb_list.append(time_emb_list[i*span])
        # print(f'{i}={t_steps_emb[i*span]}')
    selected_time_emb_list.append(time_emb_list[-1])
    if w_start==w_end:
        w_list=torch.ones_like(t_steps[:-1],device=latents.device)*w_start
    else:
        w_list=torch.arange(w_start,w_end,(w_end-w_start)/len(selected_time_emb_list),device=latents.device)
    # for i,time_emb in enumerate(selected_time_emb_list):
    #     for j,temp in enumerate(time_emb_list):
    #         if (temp==time_emb).sum()==512:
    #             print(f'i={i},j={j}')
    #             break
    # print('selected',len(selected_time_emb_list),selected_time_emb_list[0].shape)
    # print('t_steps[:-1]',t_steps[:-1].shape)
    # print('t_steps[1:]',t_steps[1:].shape)
    # exit()

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    inters_eps = []
    for i, (t_cur, t_next, time_emb, time_emb_next) in enumerate(zip(t_steps[:-1], t_steps[1:], selected_time_emb_list[:-1], selected_time_emb_list[1:])):                # 0, ..., N-1
        x_cur = x_next

        # Euler step.
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            if t_cur >= before and t_cur <= after:
                denoised = get_denoised_time_embs(net, x_cur, t_cur, time_emb, w_list[i], class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            else:
                denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            
            d_cur = (x_cur - denoised) / t_cur
        x_next = x_cur + (t_next - t_cur) * d_cur

        # Apply 2nd order correction.
        if t_next >= before and t_next <= after:
            denoised = get_denoised_time_embs(net, x_next, t_next, time_emb_next, w_list[i+1], class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        else:
            denoised = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        # denoised = get_denoised_time_embs(net, x_next, t_next, time_emb_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_next
        x_next = x_cur + (t_next - t_cur) * (0.5 * d_cur + 0.5 * d_prime)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
    # exit()
    if return_inters:
        if return_eps:
            return torch.cat(inters, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def dpm_2_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    r=0.5, 
    t_steps=None,
    **kwargs
):
    """
    DPM-Solver-2 sampler: https://arxiv.org/abs/2206.00927.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        r: A `float`. The hyperparameter controlling the location of the intermediate time step. r=0.5 recovers the original DPM-Solver-2.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)
    
    emb_data=kwargs['emb_data']
    time_emb_list=emb_data['time_emb_list']
    t_steps_emb=emb_data['t_steps']
    mid=kwargs['mid']
    before=kwargs['before']
    
    span=(len(t_steps_emb)-1)//(len(t_steps)-1)
    # print(f't_steps={t_steps}')
    # print(f't_steps_emb={t_steps_emb.shape}')
    # print(f'time_emb_list={time_emb_list.shape}')
    # span=int(span)
    # print(f'span={span}')
    # print((len(t_steps_emb)-1)/span)
    # exit()
    selected_time_emb_list=[]
    if mid:
        mid_list=[]
    for i in range((len(t_steps_emb)-1)//span):
        selected_time_emb_list.append(time_emb_list[i*span])
        # print(f'{i}={t_steps_emb[i*span]}')
        if mid:
            mid_list.append(time_emb_list[i*span+span//2])
            # print(f'{i}={t_steps_emb[i*span+span//2]}')
    # print('selected',len(selected_time_emb_list),selected_time_emb_list[0].shape)
    # print('mid_list',len(mid_list),mid_list[0].shape)
    # exit()
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    inters_eps = []
    for i, (t_cur, t_next,time_emb) in enumerate(zip(t_steps[:-1], t_steps[1:], selected_time_emb_list)):                # 0, ..., N-1
        x_cur = x_next
        # t_mid = (t_next ** r) * (t_cur ** (1 - r))
        # print(f'i={i},t_cur={t_cur},t_mid={t_mid},t_next={t_next}')
        # continue
        
        # Euler step.
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            if t_cur>before:
                denoised = get_denoised_time_embs(net, x_cur, t_cur, time_emb, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            else:
                denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        t_mid = (t_next ** r) * (t_cur ** (1 - r))
        x_next = x_cur + (t_mid - t_cur) * d_cur

        # Apply 2nd order correction.
        if mid:
            denoised = get_denoised_time_embs(net, x_next, t_mid, mid_emb, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        else:
            denoised = get_denoised(net, x_next, t_mid, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_prime = (x_next - denoised) / t_mid
        x_next = x_cur + (t_next - t_cur) * ((1 / (2*r)) * d_prime + (1 - 1 / (2*r)) * d_cur)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
    # exit()
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        if return_eps:
            return torch.cat(inters, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def ipndm_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    max_order=4, 
    t_steps=None,
    **kwargs
):
    """
    Improved PNDM sampler: https://arxiv.org/abs/2204.13902.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)  
    
    emb_data=kwargs['emb_data']
    use_res=kwargs['use_res']
    # last_n=kwargs['last_n']
    # d_steps=kwargs['d_steps']
    # time_emb_list=emb_data['time_emb_list']
    time_emb_list=emb_data
    # t_steps_emb=emb_data['t_steps']

    selected_time_emb_list=time_emb_list
 
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    inters_eps = []
    buffer_model = []
    for i, (t_cur, t_next, time_emb) in enumerate(zip(t_steps[:-1], t_steps[1:], selected_time_emb_list)):                # 0, ..., N-1
        x_cur = x_next

        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            # denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            time_emb=time_emb.to('cuda')
            denoised = get_denoised_time_embs(net, x_cur, t_cur, time_emb, use_res, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
            
        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            x_next = x_cur + (t_next - t_cur) * (3 * d_cur - buffer_model[-1]) / 2
        elif order == 3:    # Use two history points.
            x_next = x_cur + (t_next - t_cur) * (23 * d_cur - 16 * buffer_model[-1] + 5 * buffer_model[-2]) / 12
        elif order == 4:    # Use three history points.
            x_next = x_cur + (t_next - t_cur) * (55 * d_cur - 59 * buffer_model[-1] + 37 * buffer_model[-2] - 9 * buffer_model[-3]) / 24
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur
        else:
            buffer_model.append(d_cur)
        
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        if return_eps:
            return torch.cat(inters, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def ipndm_v_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    max_order=4, 
    t_steps=None,
    **kwargs
):
    """
    The variable-step version of the Adams-Bashforth methods.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    inters_eps = []
    buffer_model = []
    root_d = (latents.shape[1] * latents.shape[-1] ** 2) ** (0.5)
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):
        x_cur = x_next

        # afs
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        order = min(max_order, i+1)
        if order == 1:      # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:    # Use one history point.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            coeff1 = (2 + (h_n / h_n_1)) / 2
            coeff2 = -(h_n / h_n_1) / 2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1])
        elif order == 3:    # Use two history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            temp = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp
            coeff3 = temp * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2])
        elif order == 4:    # Use three history points.
            h_n = (t_next - t_cur)
            h_n_1 = (t_cur - t_steps[i-1])
            h_n_2 = (t_steps[i-1] - t_steps[i-2])
            h_n_3 = (t_steps[i-2] - t_steps[i-3])
            temp1 = (1 - h_n / (3 * (h_n + h_n_1)) * (h_n * (h_n + h_n_1)) / (h_n_1 * (h_n_1 + h_n_2))) / 2
            temp2 = ((1 - h_n / (3 * (h_n + h_n_1))) / 2 + (1 - h_n / (2 * (h_n + h_n_1))) * h_n / (6 * (h_n + h_n_1 + h_n_2))) \
                   * (h_n * (h_n + h_n_1) * (h_n + h_n_1 + h_n_2)) / (h_n_1 * (h_n_1 + h_n_2) * (h_n_1 + h_n_2 + h_n_3))
            coeff1 = (2 + (h_n / h_n_1)) / 2 + temp1 + temp2
            coeff2 = -(h_n / h_n_1) / 2 - (1 + h_n_1 / h_n_2) * temp1 - (1 + (h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3)))) * temp2
            coeff3 = temp1 * h_n_1 / h_n_2 + ((h_n_1 / h_n_2) + (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * (1 + h_n_2 / h_n_3)) * temp2
            coeff4 = -temp2 * (h_n_1 * (h_n_1 + h_n_2) / (h_n_2 * (h_n_2 + h_n_3))) * h_n_1 / h_n_2
            x_next = x_cur + (t_next - t_cur) * (coeff1 * d_cur + coeff2 * buffer_model[-1] + coeff3 * buffer_model[-2] + coeff4 * buffer_model[-3])
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        if return_eps:
            return torch.cat(inters, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def deis_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    max_order=4, 
    coeff_list=None, 
    t_steps=None,
    **kwargs
):
    """
    A pytorch implementation of DEIS: https://arxiv.org/abs/2204.13902.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 4
        coeff_list: A `list`. The pre-calculated coefficients for DEIS sampling.
    Returns:
        A pytorch tensor. A batch of generated samples or sampling trajectories if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 4
    assert coeff_list is not None
    
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    inters_eps = []
    buffer_model = []
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        
        use_afs = (afs and len(buffer_model) == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
        else:
            denoised = get_denoised(net, x_cur, t_cur, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        order = min(max_order, i+1)
        if order == 1:          # First Euler step.
            x_next = x_cur + (t_next - t_cur) * d_cur
        elif order == 2:        # Use one history point.
            coeff_cur, coeff_prev1 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1]
        elif order == 3:        # Use two history points.
            coeff_cur, coeff_prev1, coeff_prev2 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2]
        elif order == 4:        # Use three history points.
            coeff_cur, coeff_prev1, coeff_prev2, coeff_prev3 = coeff_list[i]
            x_next = x_cur + coeff_cur * d_cur + coeff_prev1 * buffer_model[-1] + coeff_prev2 * buffer_model[-2] + coeff_prev3 * buffer_model[-3]
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))
        
        if len(buffer_model) == max_order - 1:
            for k in range(max_order - 2):
                buffer_model[k] = buffer_model[k+1]
            buffer_model[-1] = d_cur.detach()
        else:
            buffer_model.append(d_cur.detach())
            
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        if return_eps:
            return torch.cat(inters, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next


#----------------------------------------------------------------------------

@torch.no_grad()
def dpm_pp_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    max_order=3, 
    predict_x0=True, 
    lower_order_final=True, 
    t_steps=None,
    **kwargs
):
    """
    Multistep DPM-Solver++ sampler: https://arxiv.org/abs/2211.01095. 

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform'].
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        return_eps: A `bool`. Whether to save intermediate d_cur, i.e. the gradient.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 3
        predict_x0: A `bool`. Whether to use the data prediction formulation. 
        lower_order_final: A `bool`. Whether to lower the order at the final stages of sampling. 
    Returns:
        A pytorch tensor. The sample at time `sigma_min` or the whole sampling trajectory if return_inters=True.
    """

    assert max_order >= 1 and max_order <= 3
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    emb_data=kwargs['emb_data']
    use_res=kwargs['use_res']
    # last_n=kwargs['last_n']
    selected_time_emb_list=emb_data
    # print(f'selected_time_emb_list={len(selected_time_emb_list),selected_time_emb_list[0].shape}')
    # exit()
    
    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    inters_eps = []
    buffer_model = []
    buffer_t = []
    # for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):   
    for i, (t_cur, t_next, time_emb) in enumerate(zip(t_steps[:-1], t_steps[1:], selected_time_emb_list)):# 0, ..., N-1
        x_cur = x_next
        
        use_afs = (afs and i == 0)
        if use_afs:
            d_cur = x_cur / ((1 + t_cur**2).sqrt())
            denoised = x_cur - t_cur * d_cur
        else:
            time_emb=time_emb.to('cuda')
            denoised = get_denoised_time_embs(net, x_cur, t_cur, time_emb, use_res, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
            d_cur = (x_cur - denoised) / t_cur
        
        buffer_model.append(dynamic_thresholding_fn(denoised)) if predict_x0 else buffer_model.append(d_cur)
        buffer_t.append(t_cur)
        if lower_order_final:
            order = i + 1 if i + 1 < max_order else min(max_order, num_steps - (i + 1))
        else:
            order = min(max_order, i + 1)
        x_next = dpm_pp_update(x_cur, buffer_model, buffer_t, t_next, order, predict_x0=predict_x0)
        if return_inters:
            inters.append(x_next.unsqueeze(0))
        if return_eps:
            inters_eps.append(d_cur.unsqueeze(0))

        if len(buffer_model) >= 3:
            buffer_model = [a.detach() for a in buffer_model[-3:]]
            buffer_t = [a.detach() for a in buffer_t[-3:]]
        else:
            buffer_model = [a.detach() for a in buffer_model]
            buffer_t = [a.detach() for a in buffer_t]
       
    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        if return_eps:
            return torch.cat(inters, dim=0).to(latents.device), torch.cat(inters_eps, dim=0).to(latents.device)
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next

#----------------------------------------------------------------------------

@torch.no_grad()
def unipc_sampler(
    net, 
    latents, 
    class_labels=None, 
    condition=None, 
    unconditional_condition=None,
    num_steps=None, 
    sigma_min=0.002, 
    sigma_max=80, 
    schedule_type='polynomial', 
    schedule_rho=7, 
    afs=False, 
    denoise_to_zero=False, 
    return_inters=False, 
    return_eps=False, 
    max_order=3, 
    predict_x0=True, 
    lower_order_final=True, 
    variant='bh2',
    t_steps=None,
    **kwargs
):
    """
    UniPC sampler: https://arxiv.org/abs/2302.04867.

    Args:
        net: A wrapped diffusion model.
        latents: A pytorch tensor. Input sample at time `sigma_max`.
        class_labels: A pytorch tensor. The condition for conditional sampling or guided sampling.
        condition: A pytorch tensor. The condition to the model used in LDM and Stable Diffusion
        unconditional_condition: A pytorch tensor. The unconditional condition to the model used in LDM and Stable Diffusion
        num_steps: A `int`. The total number of the time steps with `num_steps-1` spacings. 
        sigma_min: A `float`. The ending sigma during samping.
        sigma_max: A `float`. The starting sigma during sampling.
        schedule_type: A `str`. The type of time schedule. We support three types:
            - 'polynomial': polynomial time schedule. (Recommended in EDM.)
            - 'logsnr': uniform logSNR time schedule. (Recommended in DPM-Solver for small-resolution datasets.)
            - 'time_uniform': uniform time schedule. (Recommended in DPM-Solver for high-resolution datasets.)
            - 'discrete': time schedule used in LDM. (Recommended when using pre-trained diffusion models from the LDM and Stable Diffusion codebases.)
        schedule_rho: A `float`. Time step exponent. Need to be specified when schedule_type in ['polynomial', 'time_uniform']. 
        afs: A `bool`. Whether to use analytical first step (AFS) at the beginning of sampling.
        denoise_to_zero: A `bool`. Whether to denoise the sample to from `sigma_min` to `0` at the end of sampling.
        return_inters: A `bool`. Whether to save intermediate results, i.e. the whole sampling trajectory.
        max_order: A `int`. Maximum order of the solver. 1 <= max_order <= 3
        predict_x0: A `bool`. Whether to use the data prediction formulation. 
        lower_order_final: A `bool`. Whether to lower the order at the final stages of sampling. 
        variant: A `str`. Select between 'bh1' and 'bh2'. Type of the UniPC sampler.
    Returns:
        A pytorch tensor. The sample at time `sigma_min` or the whole sampling trajectory if return_inters=True.
    """

    assert max_order > 0 and max_order < 4
    max_order = max_order
    
    if t_steps is None:
        # Time step discretization.
        t_steps = get_schedule(num_steps, sigma_min, sigma_max, device=latents.device, schedule_type=schedule_type, schedule_rho=schedule_rho, net=net)

    # Main sampling loop.
    x_next = latents * t_steps[0]
    inters = [x_next.unsqueeze(0)]
    
    if afs:
        d_next = x_next / ((1 + t_steps[0]**2).sqrt())
        denoised = x_next - t_steps[0] * d_next
    else:
        denoised = get_denoised(net, x_next, t_steps[0], class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        d_next = (x_next - denoised) / t_steps[0]
    buffer_model = [dynamic_thresholding_fn(denoised)] if predict_x0 else [d_next]
    buffer_t = [t_steps[0]]
    for i, (t_cur, t_next) in enumerate(zip(t_steps[:-1], t_steps[1:])):                # 0, ..., N-1
        x_cur = x_next
        
        if i + 1 < max_order:
            order = i + 1
            use_corrector = True
            x_next, model_out = unipc_update(x_cur, buffer_model, buffer_t, t_next, order, \
                                              net=net, class_labels=class_labels, use_corrector=use_corrector, \
                                              predict_x0=predict_x0, variant=variant)
            buffer_model.append(model_out)
            buffer_t.append(t_next)
        else:
            order = min(max_order, num_steps - i - 1) if lower_order_final else max_order
            use_corrector = False if i == num_steps - 2 else True
            x_next, model_out = unipc_update(x_cur, buffer_model, buffer_t, t_next, order, \
                                              net=net, class_labels=class_labels, use_corrector=use_corrector, \
                                              predict_x0=predict_x0, variant=variant)
            for k in range(max_order - 1):
                buffer_model[k] = buffer_model[k + 1]
                buffer_t[k] = buffer_t[k + 1]
            buffer_t[-1] = t_next
            if i < num_steps - 2:
                buffer_model[-1] = model_out
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if denoise_to_zero:
        x_next = get_denoised(net, x_next, t_next, class_labels=class_labels, condition=condition, unconditional_condition=unconditional_condition)
        if return_inters:
            inters.append(x_next.unsqueeze(0))

    if return_inters:
        return torch.cat(inters, dim=0).to(latents.device)
    return x_next
