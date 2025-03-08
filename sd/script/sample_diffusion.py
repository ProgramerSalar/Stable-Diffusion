import argparse, os, sys, glob, datetime, yaml 
import torch 
import time 
import numpy as np 
from tqdm import trange

from omegaconf import OmegaConf

from PIL import Image 

from sd.models.ldm.diffusion.ddim import DDIMSampler
from sd.utils import instantiate_from_config

rescale = lambda x: (x + 1.) / 2.

def custom_to_pil(x):
    x = x.detach().cpu()
    x = torch.clamp(x, -1., 1.)
    x = (x + 1.) / 2.
    x = x.permute(1, 2, 0).numpy()
    x = (255 * x).astype(np.uint8)
    x = Image.fromarray(x)
    if not x.mode == "RGB":
        x = x.convert("RGB")

    return x 


def custom_to_np(x):
    sample = x.detach().cpu()
    sample = ((sample + 1) * 127.5).clamp(0, 255).to(torch.uint8)
    sample = sample.permute(0, 2, 3, 1)
    sample = sample.contiguous()
    return sample


def logs2pil(logs, keys=["sample"]):
    imgs = dict()
    for k in logs:
        try:
            if len(logs[k].shape) == 4:
                img = custom_to_pil(logs[k][0, ...])

            elif len(logs[k].shape) == 3:
                img = custom_to_pil(logs[k])

            else:
                print(f"Unknown format for key {k}.")
                img = None

        except:
            img = None

        imgs[k] = img 

    return imgs 


@torch.no_grad()
def convsample(model, 
               shape, 
               return_intermediates=True,
               verbose=True,
               make_prog_row=False):
    
    if not make_prog_row:
        return model.p_sample_loop(None, shape, 
                                   return_intermediates=return_intermediates, verbose=verbose)
    

    else:
        return model.progressive_denoising(
            None, shape, verbose=True
        )
    

@torch.no_grad()
def convsample_ddim(model, steps, shape, eta=1.0):

    ddim = DDIMSampler(model)
    bs = shape[0]
    shape = shape[1:]
    samples, intermediates = ddim.sample(steps, batch_size=bs, shape=shape, eta=eta, verbose=False)

    return samples, intermediates


@torch.no_grad()
def make_convolutional_sample(model, batch_size, vanilla=False, custom_steps=None, eta=1.0,):


    log = dict()

    shape = [batch_size,
             model.model.diffusion_model.in_channels,
             model.model.diffusion_model.image_size,
             model.model.diffusion_model.image_size]

    with model.ema_scope("Plotting"):
        t0 = time.time()
        if vanilla:
            sample, progrow = convsample(model, shape,
                                         make_prog_row=True)
        else:
            sample, intermediates = convsample_ddim(model,  steps=custom_steps, shape=shape,
                                                    eta=eta)

        t1 = time.time()

    x_sample = model.decode_first_stage(sample)

    log["sample"] = x_sample
    log["time"] = t1 - t0
    log['throughput'] = sample.shape[0] / (t1 - t0)
    print(f'Throughput for this batch: {log["throughput"]}')
    return log

def run(model, logdir, batch_size=50, vanilla=False, custom_steps=None, eta=None, n_samples=50000, nplog=None):
    if vanilla:
        print(f'Using Vanilla DDPM sampling with {model.num_timesteps} sampling steps.')
    else:
        print(f'Using DDIM sampling with {custom_steps} sampling steps and eta={eta}')


    tstart = time.time()
    n_saved = len(glob.glob(os.path.join(logdir,'*.png')))-1
    # path = logdir
    if model.cond_stage_model is None:
        all_images = []

        print(f"Running unconditional sampling for {n_samples} samples")
        for _ in trange(n_samples // batch_size, desc="Sampling Batches (unconditional)"):
            logs = make_convolutional_sample(model, batch_size=batch_size,
                                             vanilla=vanilla, custom_steps=custom_steps,
                                             eta=eta)
            n_saved = save_logs(logs, logdir, n_saved=n_saved, key="sample")
            all_images.extend([custom_to_np(logs["sample"])])
            if n_saved >= n_samples:
                print(f'Finish after generating {n_saved} samples')
                break
        all_img = np.concatenate(all_images, axis=0)
        all_img = all_img[:n_samples]
        shape_str = "x".join([str(x) for x in all_img.shape])
        nppath = os.path.join(nplog, f"{shape_str}-samples.npz")
        np.savez(nppath, all_img)

    else:
       raise NotImplementedError('Currently only sampling for unconditional models supported.')

    print(f"sampling of {n_saved} images finished in {(time.time() - tstart) / 60.:.2f} minutes.")


def save_logs(logs, path, n_saved=0, key="sample", np_path=None):
    for k in logs:
        if k == key:
            batch = logs[key]
            if np_path is None:
                for x in batch:
                    img = custom_to_pil(x)
                    imgpath = os.path.join(path, f"{key}_{n_saved:06}.png")
                    img.save(imgpath)
                    n_saved += 1
            else:
                npbatch = custom_to_np(batch)
                shape_str = "x".join([str(x) for x in npbatch.shape])
                nppath = os.path.join(np_path, f"{n_saved}-{shape_str}-samples.npz")
                np.savez(nppath, npbatch)
                n_saved += npbatch.shape[0]
    return n_saved

def load_model_from_config(config, sd):
    model = instantiate_from_config(config)
    model.load_state_dict(sd,strict=False)
    model.cuda()
    model.eval()
    return model


def load_model(config, ckpt, gpu=True, eval_mode=True):
    if ckpt:
        print(f"Loading model from {ckpt}")
        pl_sd = torch.load(ckpt, map_location="cpu")
        global_step = pl_sd["global_step"]
    else:
        pl_sd = {"state_dict": None}
        global_step = None
    model = load_model_from_config(config.model,
                                   pl_sd["state_dict"])

    return model, global_step


if __name__ == "__main__":

    # Load the state dictionary from a checkpoint file
    checkpoint_path = "E:/stable_diffusion/ckpt/sd-v1-4.ckpt"
    pl_sd = torch.load(checkpoint_path, map_location="cuda")
    state_dict = pl_sd["state_dict"]

    config = OmegaConf.load("E:/stable_diffusion/sd/config/celebhq.yaml")
    # print(config)

    # Load the configuration file
    # config = load_model_from_config(config=config,
    #                                 sd=state_dict)

    model, global_step = load_model(config=config,
                                    ckpt=checkpoint_path,
                                    gpu=True,
                                    eval_mode=True
                                    )
    
    # print(model)
    # Set up the log directory
    logdir = "E:/stable_diffusion/samples"
    os.makedirs(logdir, exist_ok=True)

    # Run sampling
    run(
        model=model,
        logdir=logdir,
        batch_size=16,  # Batch size for sampling
        vanilla=False,  # Use DDIM sampling (set to True for vanilla DDPM)
        custom_steps=50,  # Number of DDIM steps
        eta=1.0,  # DDIM eta parameter
        n_samples=100,  # Total number of samples to generate
        nplog=logdir  # Directory to save NumPy arrays
    )

    print("----------->", state_dict.keys())
    

    

    



# python -m sd.script.sample_diffusion