import os
import os
import numpy as np
import PIL
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import json


class LSUNBase(Dataset):
    def __init__(self, json_file, data_root, size=None, interpolation="bicubic", flip_p=0.5):
        self.data_paths = json_file
        self.data_root = data_root

        # Load JSON file
        with open(self.data_paths, "r") as f:
            data = json.load(f)
        
        # Extract paths from JSON
        self.labels = {
            "relative_file_path_": data["relative_file_path_"],
            "file_path_": [os.path.join(data_root, l.replace("/", "\\")) for l in data["relative_file_path_"]]
        }

        self._length = len(self.labels["relative_file_path_"])

        self.size = size
        self.interpolation = {
            # "linear": Image.Resampling.LINEAR,
            "bilinear": Image.Resampling.BILINEAR,
            "bicubic": Image.Resampling.BICUBIC,
            "lanczos": Image.Resampling.LANCZOS,
        }[interpolation]

        self.flip = transforms.RandomHorizontalFlip(p=flip_p)

    def __len__(self):
        return self._length
    
    def __getitem__(self, i):
        example = dict((k, self.labels[k][i]) for k in self.labels)
        image = Image.open(example["file_path_"])
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # Default to score-sde preprocessing
        img = np.array(image).astype(np.uint8)
        crop = min(img.shape[0], img.shape[1])
        h, w = img.shape[0], img.shape[1]
        img = img[(h - crop) // 2:(h + crop) // 2,
                  (w - crop) // 2:(w + crop) // 2]

        image = Image.fromarray(img)
        if self.size is not None:
            image = image.resize((self.size, self.size), resample=self.interpolation)

        image = self.flip(image)
        image = np.array(image).astype(np.uint8)
        example["image"] = (image / 127.5 - 1.0).astype(np.float32)
        return example
    

class LSUNChurchesTrain(LSUNBase):
    def __init__(self, **kwargs):
        super().__init__(json_file="E:\\stable_diffusion\\data\\archive\\data0\\lsun\\bedroom\\bedrooms_train.json",
                          data_root="E:\\stable_diffusion\\data\\archive\\data0\\lsun\\bedroom", **kwargs)


if __name__ == "__main__":

    dataset = LSUNBase(json_file="E:\\stable_diffusion\\data\\archive\\data0\\lsun\\bedroom\\bedrooms_train.json",
             data_root="E:\\stable_diffusion\\data\\archive\\data0\\lsun\\bedroom")
    
    dataset = DataLoader(dataset=dataset, batch_size=32)
    # print(dataset)



