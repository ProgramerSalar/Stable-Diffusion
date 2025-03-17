import torch 
import numpy as np
import random
import os
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets


from sd.dataset.celeb_dataset import CelebDataset
from sd.models.vae.autoencoder import AutoencoderKL
from sd.models.vae.LPIPSWithDiscriminator import LPIPSWithDiscriminator







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = v2.Compose([
    v2.Resize(size=(128, 128)),
    v2.ToTensor()
])


# train_dataset = datasets.ImageFolder(root="E:\\stable_diffusion\\Data",
#                                      transform=transform)


im_dataset = CelebDataset(im_path="E:\\stable_diffusion\\Data",  # for colab: /content/Stable-Diffusion/Data
                          im_size=128,
                          im_channels=3)


autoEncoder_data_loader = DataLoader(im_dataset,
                                     batch_size=2,
                                     shuffle=True)



def train():

    seed = 111 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    


    # create the model and dataset
    dd_config = {
      "double_z": True,
      "z_channels": 3,
      "resolution": 128,
      "in_channels": 3,
      "out_ch": 3,
      "ch": 64,
      "ch_mult": [ 1,2,4 ],  # num_down = len(ch_mult)-1
      "num_res_blocks": 2,
      "attn_resolutions": [ ],
      "dropout": 0.0
    }

    lpipwithDis = LPIPSWithDiscriminator(disc_start=50001,
                                         kl_weight=0.000001,
                                         disc_weight=0.5)

    model = AutoencoderKL(
        ddconfig=dd_config,
        lossconfig=lpipwithDis,
        embed_dim=3,
    ).to(device)

    optimizer = Adam(model.parameters(),
                     lr=1e-4)
    

    if not os.path.exists('celebhq'):
        os.mkdir('celebhq')

    num_epochs = 1 
    for epoch in range(num_epochs):
        model.train()
        for batch in tqdm(autoEncoder_data_loader):

            # Move data to device 
            inputs = batch.to(device)

            # forward pass 
            reconstructions, poseriors = model(inputs)

            # compute loss 
            loss, log = lpipwithDis(inputs,
                                    reconstructions,
                                    poseriors,
                                    optimizer_idx=0,
                                    global_step=epoch * len(autoEncoder_data_loader))
            
            # Backward pass 
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # Log losses 
            print(f"Epoch [{epoch + 1} / {num_epochs}], Loss: {loss.item()}")

             # Clear GPU memory
            torch.cuda.empty_cache()


    # save the model 
    torch.save(model.state_dict(), "celebhq/vae_model.pth")





if __name__ == "__main__":
    train()



# python -m sd.models.tools.train_vae