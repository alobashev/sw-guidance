import glob
import os.path as osp

import gc
import torch
import numpy as np
from tqdm import tqdm
from PIL import Image
from torch import optim
import torch.nn as nn

from transformers import CLIPTokenizer, CLIPTextModel
from diffusers import AutoencoderKL, UNet2DConditionModel, DDIMScheduler

from src.utils import rand_rotation_matrix, get_cdf
from src.metric_w_dist import compute_w_dist





# for time travel blocks in ddnm
def get_schedule_jump(T_sampling, travel_length, travel_repeat):
    jumps = {}
    for j in range(0, T_sampling - travel_length, travel_length):
        jumps[j] = travel_repeat - 1

    t = T_sampling
    ts = []

    while t >= 1:
        t = t-1
        ts.append(t)

        if jumps.get(t, 0) > 0:
            jumps[t] = jumps[t] - 1
            for _ in range(travel_length):
                t = t + 1
                ts.append(t)

    ts.append(-1)

    _check_times(ts, -1, T_sampling)
    return ts

def _check_times(times, t_0, T_sampling):
    # Check end
    assert times[0] > times[1], (times[0], times[1])

    # Check beginning
    assert times[-1] == -1, times[-1]

    # Steplength = 1
    for t_last, t_cur in zip(times[:-1], times[1:]):
        assert abs(t_last - t_cur) == 1, (t_last, t_cur)

    # Value range
    for t in times:
        assert t >= t_0, (t, t_0)
        assert t <= T_sampling, (t, T_sampling)


class TransformNet(nn.Module):
    def __init__(self, size):
        super(TransformNet, self).__init__()
        self.size = size
        self.net = nn.Sequential(nn.Linear(self.size, self.size))

    def forward(self, input):
        out = self.net(input)
        return out / torch.sqrt(torch.sum(out ** 2, dim=1, keepdim=True))
def rand_projections(dim, num_projections=1000):
    projections = torch.randn((num_projections, dim))
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def sliced_wasserstein_distance(first_samples, second_samples, num_projections=1000, p=2, device="cuda"):
    dim = second_samples.size(1)
    projections = rand_projections(dim, num_projections).to(device)
    first_projections = first_samples.matmul(projections.transpose(0, 1))
    # first_projections = projections.matmul(first_samples)
    second_projections = second_samples.matmul(projections.transpose(0, 1))
    # second_projections = projections.matmul(second_samples)
    
    sort_x = torch.sort(first_projections.T, dim=1)[0]
    sort_y = torch.sort(second_projections.T, dim=1)[0]
    wasserstein_distance = torch.abs(sort_x - sort_y)
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance


def compute_wasserstein_loss(pixels_gen, pixels_ref, num_slices=10, grid_size=400, 
                             use_differenatable_histogram_matching=False):
    """Compute the Wasserstein loss between generated and reference pixels."""
    loss = 0.0
    for slice_idx in range(num_slices):
        R_trans = rand_rotation_matrix(deflection=1.0)
        R_trans = torch.Tensor(R_trans).to(device)

        # Rotate generated and reference pixels
        pixels_gen_rotated = pixels_gen.T @ R_trans
        pixels_ref_rotated = pixels_ref.T @ R_trans

        # Calculate Wasserstein distance for each dimension
        for dim_idx in range(3):
            x_slice = pixels_gen_rotated[:, dim_idx]
            y_slice = pixels_ref_rotated[:, dim_idx]

            if not use_differenatable_histogram_matching:
                rand_idxes_len = min(len(x_slice), len(y_slice))
                rand_idxes = np.random.randint(0, rand_idxes_len, rand_idxes_len)
                x_slice, y_slice = x_slice[rand_idxes], y_slice[rand_idxes]

                # Sort and calculate Wasserstein distance for this slice
                x_sorted, y_sorted = torch.sort(x_slice).values, torch.sort(y_slice).values
                loss += torch.mean(torch.abs(x_sorted - y_sorted))
            else:
                # Using differentiable histogram matching
                min_range = min(x_slice.min().item(), y_slice.min().item())
                max_range = max(x_slice.max().item(), y_slice.max().item())

                full_range = torch.linspace(min_range - 0.05, max_range + 0.05, grid_size).to(device)
                x_cdf = get_cdf(x_slice, full_range)
                y_cdf = get_cdf(y_slice, full_range)
                
                loss += torch.mean(torch.abs(x_cdf - y_cdf))

    return loss
    
def distributional_sliced_wasserstein_distance(
    first_samples, second_samples, num_projections, f, f_op, p=2, max_iter=10, lam=1, device="cuda"
):
    embedding_dim = first_samples.size(1)
    pro = rand_projections(embedding_dim, num_projections).to(device)
    first_samples_detach = first_samples.detach()
    second_samples_detach = second_samples.detach()
    for _ in range(max_iter):
        projections = f(pro)
        cos = cosine_distance_torch(projections, projections)
        reg = lam * cos
        encoded_projections = first_samples_detach.matmul(projections.transpose(0, 1))
        distribution_projections = second_samples_detach.matmul(projections.transpose(0, 1))
        wasserstein_distance = torch.abs(
            (
                torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
                - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
            )
        )
        wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
        loss = reg - wasserstein_distance
        f_op.zero_grad()
        loss.backward(retain_graph=True)
        f_op.step()

    projections = f(pro)
    encoded_projections = first_samples.matmul(projections.transpose(0, 1))
    distribution_projections = second_samples.matmul(projections.transpose(0, 1))
    wasserstein_distance = torch.abs(
        (
            torch.sort(encoded_projections.transpose(0, 1), dim=1)[0]
            - torch.sort(distribution_projections.transpose(0, 1), dim=1)[0]
        )
    )
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance
def cosine_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.abs(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps)))


def cosine_sum_distance_torch(x1, x2=None, eps=1e-8):
    x2 = x1 if x2 is None else x2
    w1 = x1.norm(p=2, dim=1, keepdim=True)
    w2 = w1 if x2 is x1 else x2.norm(p=2, dim=1, keepdim=True)
    return torch.mean(torch.mm(x1, x2.t()) / (w1 * w2.t()).clamp(min=eps))



def gsw( first_samples, second_samples, num_projections=10, ftype="linear", degree=2, radius=1, device='cuda', p=1):
    
    dim = second_samples.size(1)
    
    if ftype=='linear':
        theta=torch.randn((num_projections,dim))
        theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta]).to(device)
        first_projections = linear(first_samples, theta)
        second_projections = linear(second_samples, theta)
    elif ftype=='poly':
        dpoly=homopoly(dim,degree)
        theta=torch.randn((num_projections,dpoly))
        theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta]).to(device)
        first_projections = poly(first_samples, theta, degree)
        second_projections = poly(second_samples, theta, degree)
    elif ftype=='circular':
        theta=torch.randn((num_projections,dim))
        theta=torch.stack([radius*th/torch.sqrt((th**2).sum()) for th in theta]).to(device)
        first_projections = circular(first_samples, theta)
        second_projections = circular(second_samples, theta)

    sort_x = torch.sort(first_projections, dim=0)[0]
    sort_y = torch.sort(second_projections, dim=0)[0]
    wasserstein_distance = torch.abs(sort_x - sort_y)
    wasserstein_distance = torch.pow(torch.sum(torch.pow(wasserstein_distance, p), dim=1).mean(), 1.0 / p)
    return wasserstein_distance



def linear(X,theta):
    if len(theta.shape)==1:
        return torch.matmul(X,theta)
    else:
        return torch.matmul(X,theta.t())

def poly(X,theta, degree=2):
    ''' The polynomial defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
        degree: degree of the polynomial
    '''
    N,d=X.shape
    assert theta.shape[1]==homopoly(d,degree)
    powers=list(get_powers(d,degree))
    HX=torch.ones((N,len(powers))).to(device)
    for k,power in enumerate(powers):
        for i,p in enumerate(power):
            HX[:,k]*=X[:,i]**p
    if len(theta.shape)==1:
        return torch.matmul(HX,theta)
    else:
        return torch.matmul(HX,theta.t())

def circular(X,theta):
    ''' The circular defining function for generalized Radon transform
        Inputs
        X:  Nxd matrix of N data samples
        theta: Lxd vector that parameterizes for L projections
    '''
    N,d=X.shape
    if len(theta.shape)==1:
        return torch.sqrt(torch.sum((X-theta)**2,dim=1))
    else:
        return torch.stack([torch.sqrt(torch.sum((X-th)**2,dim=1)) for th in theta],1)


def get_powers( dim, degree):
    """
    This function calculates the powers of a homogeneous polynomial
    e.g.

    list(get_powers(dim=2,degree=3))
    [(0, 3), (1, 2), (2, 1), (3, 0)]

    list(get_powers(dim=3,degree=2))
    [(0, 0, 2), (0, 1, 1), (0, 2, 0), (1, 0, 1), (1, 1, 0), (2, 0, 0)]
    """
    if dim == 1:
        yield (degree,)
    else:
        for value in range(degree + 1):
            for permutation in get_powers(dim - 1, degree - value):
                yield (value,) + permutation

def homopoly( dim, degree):
    """
    calculates the number of elements in a homogeneous polynomial
    """
    return len(list(get_powers(dim, degree)))


def ISEBSW( first_samples, second_samples, num_projections=10, device='cuda', p=1):
    
    dim = second_samples.size(1)
    theta=torch.randn((num_projections,dim))
    theta=torch.stack([th/torch.sqrt((th**2).sum()) for th in theta]).to(device)
    first_projections = linear(first_samples, theta)
    second_projections = linear(second_samples, theta)

    sort_x = torch.sort(first_projections, dim=0)[0]
    sort_y = torch.sort(second_projections, dim=0)[0]
    wasserstein_distance = torch.abs(sort_x - sort_y)
    wasserstein_distance = torch.pow(wasserstein_distance, p)
    weights = torch.softmax(wasserstein_distance, dim=1)
    wasserstein_distance = torch.pow(torch.sum(weights*wasserstein_distance, dim=1).mean(), 1.0 / p)
    
    return wasserstein_distance
    
    
device = "cuda:0"
debug_varince = False
use_differenatable_historgam_matching = False
num_slices = 10
height = 512
width = 512
########### v40
save_folder = 'generation_v40'
num_inference_steps = 10
M = 10 # iterations for each denoising steps
guidance_scale = 8
u_lr = 1/25
wsd_p = 1 # Sliced Wasserstein power
travel_repeat = 1 #Time travel
travel_length = 1
loss_type = 0  # 0 - mean_cov, 1 - swd, 2 - dswd, 3 - gswd, 4- eswd
ftype = 'poly' # for gswd
degree = 5 # for poly gswd
stop_guidance = 0.99 # stopping guidance if  i < int(scheduler.num_inference_steps*stop_guidance)
use_sarah = False # whether to use SARAH gradient or not


times  = get_schedule_jump(num_inference_steps, travel_length, travel_repeat)
time_pairs = list(zip(times[:-1], times[1:]))


captions = np.load('caption_urls_1000.npy', allow_pickle=True).item().get('captions')

ref_path = './unsplash/'
ref_img_names = [osp.basename(k) for k in glob.glob(ref_path + "/*.png")]
ref_img_names.sort()


tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
vae = AutoencoderKL.from_pretrained("stabilityai/sd-vae-ft-mse").to(device)
text_encoder = CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
unet = UNet2DConditionModel.from_pretrained("Lykon/dreamshaper-8", subfolder="unet").to(device)
scheduler = DDIMScheduler.from_config("CompVis/stable-diffusion-v1-4", subfolder="scheduler")

scheduler.set_timesteps(num_inference_steps)


# For DSWD
transform_net = TransformNet(3).to(device)
op_trannet = optim.Adam(transform_net.parameters(), betas=(0.5, 0.999))



def decode_to_image(latent_aux):
    with torch.no_grad():
        image = vae.decode(1 / 0.18215 * latent_aux).sample
    image = (image / 2 + 0.5).clamp(0, 1)
    image = image.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (image * 255).round().astype("uint8")
    images =  Image.fromarray(images[0])
    return images

for idx, prompt in enumerate(tqdm(captions)):
    prompt = [prompt]

    batch_size = 1

    text_input = tokenizer(prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
    with torch.no_grad():
        prompt_embeddings = text_encoder(text_input.input_ids.to(device))[0]
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer(
        [""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt"
    )
    with torch.no_grad():
        uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
    text_embeddings = torch.cat([
                                 uncond_embeddings, 
                                 prompt_embeddings,
                                ])

    ref_im = Image.open(
        "unsplash/"+ref_img_names[idx]
    ).convert('RGB')
    
    ref_im = ref_im.resize((512,512))

    pixels_ref = torch.Tensor(np.array(ref_im)/255).permute(2,1,0).reshape(3,-1).to(device)
    ref_mean = torch.mean(pixels_ref, dim=1).to(device)
    ref_cov = torch.cov(pixels_ref).to(device)
    
    seed = np.random.randint(0,10000000000)
    #print(f"seed: {seed}")
    generator = torch.manual_seed(seed)
    
    x_t = torch.randn(
      (batch_size, unet.config.in_channels, height // 8, width // 8),
      generator=generator,
    )
    x_t = x_t.to(device)
    
    
    x_t = x_t * scheduler.init_noise_sigma


    if use_sarah:
        ##########################################
        # SARAH 
        v = torch.zeros_like(x_t, requires_grad=False)
        buf_grad = torch.zeros_like(v)
        ##########################################
    
    gc.collect()
    torch.cuda.empty_cache()

    for i, j in tqdm(time_pairs):
        if i > 1:
            if j < i:
                i = scheduler.num_inference_steps - i
                j = scheduler.num_inference_steps - j
    
                timesteps = scheduler.timesteps[i]
                at = scheduler.alphas_cumprod[timesteps]

                u = torch.zeros_like(x_t, requires_grad=True)

                ## DEl if not needed
                if  i < int(scheduler.num_inference_steps*stop_guidance):
                    for _ in range(M):
                        u = u.detach()
                        u.requires_grad = True
                        x_hat_t = x_t.detach() + u

                      
                        noise_pred = unet(x_hat_t, timesteps, encoder_hidden_states=prompt_embeddings).sample

                        
                        x_0 = (x_hat_t - (1-at)**(0.5) * noise_pred) /  at ** (0.5)
                        size = np.random.randint(48, 64)
                        x_0 = torch.nn.functional.interpolate(x_0, (size,size), mode='bicubic')
                        #x_0 = x_0[:,:,::3,::3]
                    
                        # Compute loss
                        image = vae.decode(1 / 0.18215 * x_0).sample
                    
                    
                        image = (image / 2 + 0.5).clamp(0, 1)
                        pixels_gen = image.squeeze(0).reshape(3,-1)
            
                        gen_mean = torch.mean(pixels_gen, dim=1)
                        gen_cov = torch.cov(pixels_gen)

                        rand_idxes = np.random.randint(0, pixels_ref.shape[1], pixels_gen.shape[1])

                        if loss_type == 0:
                            loss = torch.mean(torch.square(gen_mean - ref_mean)) + torch.mean(torch.square(gen_cov - ref_cov))
                        # Compute Wasserstein loss and add to overall loss
                        if loss_type == 1:
                            loss  = sliced_wasserstein_distance(pixels_gen.T, pixels_ref[:,rand_idxes].T, 
                                                                device=device, num_projections=100, p=wsd_p)
                        if loss_type == 2:
                            loss  = distributional_sliced_wasserstein_distance(
                                                                pixels_gen.T,
                                                                pixels_ref[:,rand_idxes].T,
                                                                f = transform_net, f_op = op_trannet,
                                                                device=device, num_projections=100, p=wsd_p,
                                                                )
                        if loss_type == 3:
                            loss = gsw(pixels_gen.T,pixels_ref[:,rand_idxes].T,
                                    # ftype='poly',
                                    ftype=ftype,
                                    degree=degree,
                                    device=device, num_projections=100, p=wsd_p,)
                        if loss_type == 4:
                            loss = ISEBSW(pixels_gen.T,pixels_ref[:,rand_idxes].T,
                                    device=device, num_projections=100, p=wsd_p,)
                        #Change here! 
                        


                        # SARAH
                        u_t_grad = torch.autograd.grad(loss, u)[0]
                        with torch.no_grad():
                            u_t_grad = u_t_grad / u_t_grad.std()  # Normalize gradient
                            if use_sarah:
                                if i == 0:
                                    v.data = u_t_grad
                                else:
                                    v.data = v.data + u_t_grad - buf_grad  # SARAH update
                                buf_grad = u_t_grad.clone()  # Update buffer
                                u.data = u.data - u_lr * v.data  # Gradient step on u
                            else:
                                u.data = u.data - u_lr * u_t_grad.data
                        
                        #del noise_pred, sch_step, x_0, image, pixels_gen, gen_mean, gen_cov, loss, u_t_grad
                            
                        #gc.collect()
                        #torch.cuda.empty_cache()
            
                with torch.no_grad():
                    x_star_t = x_t.detach() + u.detach()
                    noise_pred_cond = unet(x_star_t, timesteps, encoder_hidden_states=prompt_embeddings).sample
                    noise_pred_uncond = unet(x_star_t, timesteps, encoder_hidden_states=uncond_embeddings).sample
                    guided_noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                    
                    timestep  = scheduler.timesteps[i]
                    at  = scheduler.alphas_cumprod[timesteps]
                    timesteps_next = scheduler.timesteps[j]
                    at_next = scheduler.alphas_cumprod[timesteps_next] if timesteps_next >= 0 else scheduler.final_alpha_cumprod
                    
                    x_0 = (x_star_t - (1-at)**(0.5) * guided_noise_pred) /  at ** (0.5)
            
                    x_t.data = at_next ** (0.5) * x_0 + (1 - at_next) ** (0.5) * guided_noise_pred

                x0_pred_last = x_0.to('cpu')
                if use_sarah:
                    buf_grad.data *= u_lr
                    v.data *= u_lr
            else:
                # backward step for repaint strategy 
                j = scheduler.num_inference_steps - j
                timesteps_next = scheduler.timesteps[j]
                at_next =  scheduler.alphas_cumprod[timesteps_next]
                x0_t = x0_pred_last.to(device)
                x_t = at_next.sqrt() * x0_t +  torch.randn_like(x0_t) * (1 - at_next).sqrt()

    decode_to_image(x0_pred_last.to(device)).save(f"{save_folder}/{idx}.png")