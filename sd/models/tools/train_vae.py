import torch 
import numpy as np
import random
import os
from torch.optim import Adam
from tqdm import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from torchvision import datasets
from dataset.celeb_dataset import CelebDataset

from sd.models.vae.autoencoder import VQModel







device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

transform = v2.Compose([
    v2.Resize(size=(128, 128)),
    v2.ToTensor()
])


train_dataset = datasets.ImageFolder(root="E:\\stable_diffusion\\Data",
                                     transform=transform)


im_dataset = CelebDataset(im_path="E:\\stable_diffusion\\Data",
                          im_size=128,
                          im_channels=3)


autoEncoder_data_loader = DataLoader(im_dataset,
                                     batch_size=4,
                                     shuffle=True)



def train():

    seed = 111 
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    if device == "cuda":
        torch.cuda.manual_seed_all(seed)


    # create the model and dataset 



if __name__ == "__main__":
    print(train_dataset)



# python -m sd.models.tools.train_vae