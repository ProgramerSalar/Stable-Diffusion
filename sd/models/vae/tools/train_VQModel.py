from sd.models.vae.autoencoder import VQModel
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from sd.dataset.custom_dataset import ImageDataset

transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = ImageDataset(image_folder="E:\\stable_diffusion\\Data\\10_images", transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

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
    "target": "torch.nn.MSELoss"
}

# Other hyperparameters
n_embed = 512  # Number of embeddings in the codebook
embed_dim = 64  # Dimensionality of each embedding
image_key = "image"  # Key for accessing images in the batch

# Instantiate the model
model = VQModel(
    ddconfig=dd_config,
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

# Define the trainer
trainer = Trainer(
    max_epochs=100,  # Number of epochs
    gpus=1 if torch.cuda.is_available() else 0,  # Use GPU if available
    callbacks=[checkpoint_callback],  # Add checkpoint callback
    progress_bar_refresh_rate=20,  # Update progress bar every 20 batches
    logger=True,  # Enable logging (e.g., TensorBoard)
)


   

if __name__ == "__main__":
    trainer.fit(model=model, dataloader = dataloader)


# python -m sd.models.vae.tools.train_VQModel