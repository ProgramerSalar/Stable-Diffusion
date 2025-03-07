import torch 
import pytorch_lightning as pl 
import torch.nn.functional as F 
from contextlib import contextmanager
import numpy as np 
from torch.optim.lr_scheduler import LambdaLR

from sd.models.vae.unet import Encoder, Decoder
from sd.utils import instantiate_from_config
from taming.modules.vqvae.quantize import VectorQuantizer2 as VectorQuantizer
from sd.models.vae.ema import LitEma



class VQModel(pl.LightningDataModule):

    def __init__(self,
                 ddconfig,
                 lossconfig,
                 n_embed,
                 embed_dim,
                 ckpt_path=None,
                 ignore_keys=[],
                 image_key="image",
                 colorize_nlabels=None,
                 monitor=None,
                 batch_resize_range=None,
                 scheduler_config=None,
                 lr_g_factor=1.0,
                 remap=None,
                 sane_index_shape=False,
                 use_ema=False):
        
        super().__init__()
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**ddconfig)
        self.decoder = Decoder(**ddconfig)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        
        self.quant_conv = torch.nn.Conv2d(ddconfig["z_channels"], embed_dim, 1)
        self.post_quant_conv = torch.nn.Conv2d(embed_dim, ddconfig["z_channels"], 1)

        


if __name__ == "__main__":

    # Example cofig for the encoder and decoder (ddconfig) 
    dd_config = {
    "double_z": True,
      "z_channels": 3,
      "resolution": 256,
      "in_channels": 3,
      "out_ch": 3,
      "ch": 128,
      "ch_mult": [ 1,2,4 ],  # num_down = len(ch_mult)-1
      "num_res_blocks": 2,
      "attn_resolutions": [ ],
      "dropout": 0.0
    }

    loss_config = {
        "target": "torch.nn.MSELoss"
    }

    # other hyperparameters 
    n_embed = 512 
    embed_dim = 64 
    image_key = "image"

    model = VQModel(
        ddconfig = dd_config,
        lossconfig = loss_config,
        n_embed=n_embed,
        embed_dim=embed_dim,
        use_ema=True
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)

    # python -m sd.models.vae.autoencoder