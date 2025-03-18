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
from sd.models.vae.distribution import DiagonalGaussianDistribution


class VQModel(pl.LightningModule):

    def __init__(self,
                 dd_config,
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
                 learning_rate = 1e-4,
                 use_ema=False):
        
        super().__init__()
        self.automatic_optimization = False
        self.learning_rate = learning_rate
        self.embed_dim = embed_dim
        self.n_embed = n_embed
        self.image_key = image_key
        self.encoder = Encoder(**dd_config)
        self.decoder = Decoder(**dd_config)
        self.loss = instantiate_from_config(lossconfig)
        self.quantize = VectorQuantizer(n_embed, embed_dim, beta=0.25,
                                        remap=remap,
                                        sane_index_shape=sane_index_shape)
        
        self.quant_conv = torch.nn.Conv2d(in_channels=dd_config["z_channels"],
                                          out_channels=embed_dim,
                                          kernel_size=1)
        self.post_quant_conv = torch.nn.Conv2d(in_channels=embed_dim,
                                               out_channels=dd_config["z_channels"],
                                               kernel_size=1)
        
        if colorize_nlabels is not None:
            assert type(colorize_nlabels) == int 
            self.register_buffer(name="colorize",
                                 tensor=torch.randn(3, colorize_nlabels, 1, 1))
            

        if monitor is not None:
            self.monitor = monitor

        self.batch_resize_range = batch_resize_range
        if self.batch_resize_range is not None:
            print(f"{self.__class__.__name__}: Using per-batch resizing in range {batch_resize_range}. ")


        self.use_ema = use_ema
        if self.use_ema:
            self.model_ema = LitEma(self)
            print(f"Keeping EMAs of {len(list(self.model_ema.buffers()))}.")


        if ckpt_path is not None:
            self.init_from_ckpt(ckpt_path, ignore_keys=ignore_keys)

        self.scheduler_config = scheduler_config
        self.lr_g_factor = lr_g_factor



    @contextmanager
    def ema_scope(self, context=None):
        if self.use_ema:
            self.model_ema.store(self.parameters())
            self.model_ema.copy_to(self)
            if context is not None:
                print(f"{context}: Switched to EMA weights")
        try:
            yield None
        finally:
            if self.use_ema:
                self.model_ema.restore(self.parameters())
                if context is not None:
                    print(f"{context}: Restored training weights")        





    def init_from_ckpt(self, path, ignore_keys=list()):
        device = torch.device("cuda")
        sd = torch.load(path, map_location=device)["state_dict"]
        keys = list(sd.keys())
        for k in keys:
            for ik in ignore_keys:
                if k.startwith(ik):
                    print("Deleting key {} from state_dict.".format(k))
                    del sd[k]

        missing, unexpected = self.load_state_dict(sd, strict=False)
        print(f"Restored from {path} with {len(missing)} missing and {len(unexpected)} unexpected keys")

        if len(missing) > 0:
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")


        
    def on_train_batch_end(self, *args, **kwargs):
        if self.use_ema:
            self.model_ema(self)



    def encode(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        
        quant, emb_loss, info = self.quantize(h)
        return quant, emb_loss, info 
    
    def encode_to_perquant(self, x):
        h = self.encoder(x)
        h = self.quant_conv(h)
        return h 
    
    def decode(self, quant):
        quant = self.post_quant_conv(quant)
        dec = self.decoder(quant)
        return dec 
    

    def forward(self, input, return_pred_indices=False):
        quant, diff, (_, _, ind) = self.encode(input)
        dec = self.decode(quant)

        if return_pred_indices:
            return dec, diff, ind
        
        return dec, diff 
    

    def get_input(self, batch, k):
        x = batch[k]
        if len(x.shape) == 3:
            x = x[..., None]

        x = x.permute(0, 3, 1, 2).to(memory_format=torch.contiguous_format).float()

        if self.batch_resize_range is not None:
            lower_size = self.batch_resize_range[0]
            upper_size = self.batch_resize_range[1]

            if self.global_step <= 4:
                # do the first  few batches with max size  to avoid leter oom 
                new_resize = upper_size

            else:
                new_resize = np.random.choice(np.arange(lower_size, upper_size + 16, 16))

            if new_resize != x.shape[2]:
                x = F.interpolate(x, size=new_resize, mode="bicubic")

            x = x.detach()
            
        return x 
    


    def training_step(self, batch, batch_idx, optimizer_idx):

        x = self.get_input(batch, self.image_key)
        xrec, qloss, ind = self(x, return_pred_indices=True)

        if optimizer_idx == 0:
            # autoencoder 
            aeloss, log_dict_ae = self.loss(qloss,
                                            x, 
                                            xrec,
                                            optimizer_idx,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train",
                                            predicted_indices = ind)
            
            self.log_dict(log_dict_ae, 
                          prog_bar=False,
                          logger=True,
                          on_step=True,
                          on_epoch=True)
            
            return aeloss
        

        if optimizer_idx == 1:

            # discriminator 
            discloss, log_dict_dic = self.loss(qloss,
                                               x,
                                               xrec,
                                               optimizer_idx,
                                               self.global_step,
                                               last_layer=self.get_last_layer(),
                                               split="train")
            
            self.log_dict(log_dict_dic, prog_bar=False, logger=True, on_step=True, on_epoch=True)
            return discloss
        


    def validation_step(self, batch, batch_idx):
        log_dict = self._validation_step(batch, batch_idx)
        with self.ema_scope():
            log_dict_ema = self._validation_step(batch, batch_idx, suffix="_ema")
            log_dict.update(log_dict_ema)

        return log_dict
    
            

    def get_last_layer(self):
        return self.decoder.conv_out.weight


    def _validation_step(self, batch, batch_idx, suffix=""):
            x = self.get_input(batch, self.image_key)
            xrec, qloss, ind = self(x, return_pred_indices=True)
            aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="val"+suffix,
                                            predicted_indices=ind
                                            )
    
            discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                                self.global_step,
                                                last_layer=self.get_last_layer(),
                                                split="val"+suffix,
                                                predicted_indices=ind
                                                )
            rec_loss = log_dict_ae[f"val{suffix}/rec_loss"]
            self.log(f"val{suffix}/rec_loss", rec_loss,
                       prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            self.log(f"val{suffix}/aeloss", aeloss,
                       prog_bar=True, logger=True, on_step=False, on_epoch=True, sync_dist=True)
            # if version.parse(pl.__version__) >= version.parse('1.4.0'):
                # del log_dict_ae[f"val{suffix}/rec_loss"]
            self.log_dict(log_dict_ae)
            self.log_dict(log_dict_disc)
            return self.log_dict    
    


    def configure_optimizers(self):
        lr_d = self.learning_rate
        lr_g = self.lr_g_factor*self.learning_rate
        print("lr_d", lr_d)
        print("lr_g", lr_g)
        opt_ae = torch.optim.Adam(list(self.encoder.parameters())+
                                  list(self.decoder.parameters())+
                                  list(self.quantize.parameters())+
                                  list(self.quant_conv.parameters())+
                                  list(self.post_quant_conv.parameters()),
                                  lr=lr_g, betas=(0.5, 0.9))
        opt_disc = torch.optim.Adam(self.loss.discriminator.parameters(),
                                    lr=lr_d, betas=(0.5, 0.9))

        if self.scheduler_config is not None:
            scheduler = instantiate_from_config(self.scheduler_config)

            print("Setting up LambdaLR scheduler...")
            scheduler = [
                {
                    'scheduler': LambdaLR(opt_ae, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
                {
                    'scheduler': LambdaLR(opt_disc, lr_lambda=scheduler.schedule),
                    'interval': 'step',
                    'frequency': 1
                },
            ]
            return [opt_ae, opt_disc], scheduler
        return [opt_ae, opt_disc], []

    def get_last_layer(self):
        return self.decoder.conv_out.weight

    def log_images(self, batch, only_inputs=False, plot_ema=False, **kwargs):
        log = dict()
        x = self.get_input(batch, self.image_key)
        x = x.to(self.device)
        if only_inputs:
            log["inputs"] = x
            return log
        xrec, _ = self(x)
        if x.shape[1] > 3:
            # colorize with random projection
            assert xrec.shape[1] > 3
            x = self.to_rgb(x)
            xrec = self.to_rgb(xrec)
        log["inputs"] = x
        log["reconstructions"] = xrec
        if plot_ema:
            with self.ema_scope():
                xrec_ema, _ = self(x)
                if x.shape[1] > 3: xrec_ema = self.to_rgb(xrec_ema)
                log["reconstructions_ema"] = xrec_ema
        return log

    def to_rgb(self, x):
        assert self.image_key == "segmentation"
        if not hasattr(self, "colorize"):
            self.register_buffer("colorize", torch.randn(3, x.shape[1], 1, 1).to(x))
        x = F.conv2d(x, weight=self.colorize)
        x = 2.*(x-x.min())/(x.max()-x.min()) - 1.
        return x   



    def training_step(self, batch, batch_idx):
        # Get the optimizers
        opt_ae, opt_disc = self.optimizers()

        # Get the input data
        x = self.get_input(batch, self.image_key)

        # Forward pass: Autoencoder
        xrec, qloss, ind = self(x, return_pred_indices=True)

        # Autoencoder loss
        aeloss, log_dict_ae = self.loss(qloss, x, xrec, 0,
                                        self.global_step,
                                        last_layer=self.get_last_layer(),
                                        split="train",
                                        predicted_indices=ind)
        
        # Log autoencoder loss
        self.log_dict(log_dict_ae, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # Update autoencoder
        opt_ae.zero_grad()
        self.manual_backward(aeloss)
        opt_ae.step()

        # Discriminator loss
        discloss, log_dict_disc = self.loss(qloss, x, xrec, 1,
                                            self.global_step,
                                            last_layer=self.get_last_layer(),
                                            split="train")
        
        # Log discriminator loss
        self.log_dict(log_dict_disc, prog_bar=False, logger=True, on_step=True, on_epoch=True)

        # Update discriminator
        opt_disc.zero_grad()
        self.manual_backward(discloss)
        opt_disc.step()

    def train_dataloader(self):
        # Define your dataset and transformations
        transform = transforms.Compose([
            transforms.Resize((256, 256)),  # Resize to the input size expected by the model
            transforms.ToTensor(),          # Convert to tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # Normalize to [-1, 1]
        ])

        # Create dataset
        dataset = ImageDataset(image_folder="E:\\stable_diffusion\\Data\\10_images", transform=transform)

        # Create DataLoader
        batch_size = 32
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)
        return dataloader





# ------------------------------------------------------------------------------------------------------------------------------
from sd.models.vae.autoencoder import VQModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sd.dataset.custom_dataset import ImageDataset
# from taming.modules.losses.vqperceptual import VQLPIPSWithDiscriminator

if __name__ == "__main__":

    

    # Define the encoder/decoder configuration (ddconfig)
    dd_config = {
        "double_z": True,
        "z_channels": 3,
        "resolution": 256,
        "in_channels": 3,
        "out_ch": 3,
        "ch": 128,
        "ch_mult": [1, 2, 4],  # num_down = len(ch_mult)-1
        "num_res_blocks": 2,
        "attn_resolutions": [],
        "dropout": 0.0
    }

    # Define the loss configuration
    loss_config = {
        "target": "taming.modules.losses.vqperceptual.VQLPIPSWithDiscriminator",
        "params": {
            "disc_conditional": False,
            "disc_in_channels": 3,
            "disc_start": 0,
            "disc_weight": 0.75,
            "codebook_weight": 1.0
        }
    }

    # Other hyperparameters
    n_embed = 512  # Number of embeddings in the codebook
    embed_dim = 64  # Dimensionality of each embedding
    image_key = "image"  # Key for accessing images in the batch

    # Instantiate the model
    model = VQModel(
        dd_config=dd_config,
        lossconfig=loss_config,
        n_embed=n_embed,
        embed_dim=embed_dim,
        image_key=image_key,
        use_ema=True  # Enable EMA for better stability
    )


    from pytorch_lightning import Trainer
    from pytorch_lightning.callbacks import ModelCheckpoint
    import torch

    # Define a checkpoint callback to save the best model
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",  # Monitor validation loss
        dirpath="checkpoints/",  # Directory to save checkpoints
        filename="vqmodel-{epoch:02d}-{val_loss:.2f}",
        save_top_k=3,  # Save the top 3 models
        mode="min",  # Minimize validation loss
    )
    device = torch.device("cuda:0")
    # Define the trainer
    trainer = Trainer(
        max_epochs=100,  # Number of epochs
        accelerator="gpu",
        callbacks=[checkpoint_callback],  # Add checkpoint callback
        # progress_bar_refresh_rate=20,  # Update progress bar every 20 batches
        enable_progress_bar=True,
        logger=True,  # Enable logging (e.g., TensorBoard)
    )





   


    trainer.fit(model=model)


    # python -m sd.models.vae.autoencoder