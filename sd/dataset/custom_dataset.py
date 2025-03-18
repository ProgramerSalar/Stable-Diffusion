import os 
from PIL import Image 
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class ImageDataset(Dataset):

    def __init__(
            self, 
            image_folder,
            transform=None
    ):
        
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, fname) for fname in os.listdir(image_folder)]
        self.transform = transform


    def __len__(self):
        return len(self.image_paths)
    

    def __getitem__(self, index):
        image_path = self.image_paths[index]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        return image 
    


transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])
   


if __name__ == "__main__":

    dataset = ImageDataset(image_folder="E:\\stable_diffusion\\Data\\10_images", transform=transform)
    
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    for i in dataloader:
        print(i.shape)


# python -m sd.dataset.custom_dataset