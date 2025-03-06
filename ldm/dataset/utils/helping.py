import PIL.Image
import yaml, cv2, PIL
from torch.utils.data import Dataset, Subset
from omegaconf import OmegaConf
import os, pickle, shutil, tarfile, glob, tqdm
import numpy as np 
import torchvision.transforms.functional as TF
from PIL import Image

## tamming models
import taming.data.utils as tdu 
from taming.data.imagenet import str_to_indices, give_synsets_from_indices, download, retrieve
from taming.data.imagenet import ImagePaths

# ldm models 
from ldm.modules.image_degradation import degradation_fn_bsr, degradation_fn_bsr_light

import albumentations
from functools import partial

from dataset.imagenet import ImageNetTrain

class ImageNetSR(Dataset):

    """ 
    1. Load and preprocess High-resolution image from the ImageNet dataset.
    2. Crops and resize the High-resolution images.
    3. Degrades the High-resolution image to generate corresponding Low-resolution image using specified degradation methods.
    4. Returns normalized High-Resolution and Low-resolution for training super-resolution models.
    """

    def __init__(self,
                 size=None,          # The target size for resize the high resolution image after cropping
                 degradation=None,  # The degradation function to apply to generate the low resolution image (eg. bicubic interpolation or BSRGAN)
                 downscale_f=4,     # The downscaling factor to generate the low resolution image (default is 4)
                 min_crop_f=0.5,    # The mim crop factor to determine the crop size (default is 0.5)
                 max_crop_f=1.,     # the max crop factor to determine the crop size (default is 1.0)
                 random_crop=True):     # Whether to use random cropping or center cropping (default is True)
        

        # calls a method `get_base()` to load the base dataset. This method is expected to return a list of dictionaries, where each dictionary contains metadata about an image.
        self.base = ImageNetSRTrain.get_base() 
        assert size                                 # Ensure that the size parameters is provided. Raise an error if `size` is `None`
        assert (size / downscale_f).is_integer()    # Ensure that the target size is divisible by the downscaling factor. This ensures that the low resolution image size is an integer
        self.size = size                            # store the target size for the high-resolution image.
        self.LR_size = int(size / downscale_f)      # computes and store the size of the Low-resolution image by dividing the high-resolution size by the downscaling factor.
        # store the min and max crop factor.
        self.min_crop_f = min_crop_f                
        self.max_crop_f = max_crop_f
        assert (max_crop_f <= 1.)                   # Ensure that the max crop factor does not exceed 1.0(100% of the image size.)
        self.center_crop = not random_crop          # sets `self.center_crop` to `True` if `random_crop` is `False`, and vice versa

        # initializes an `albumentations.SmallestMaxSize` object to resize the cropped high-resolution image to the target size using area interpolation.
        self.image_rescaler = albumentations.SmallestMaxSize(max_size=size, interpolation=cv2.INTER_AREA)
        # initializes a flag `self.pil_interpolation` to `False`. This flag is later set to `True` if the degradation process uses pillow interpolation.
        self.pil_interpolation = False # gets reset later if incase interp_op is from pillow 

        # if the degradation method is `bsrgan` sets `self.degradation_process` to a partial function of `degradation_fn_bsr` with the downscaling factor
        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_fn_bsr, sf=downscale_f)

        # if the degradation method is `bsrgan_light`, sets `self.degradation_process` to a partial fun of `degradation_fn_bsr_light` with the downscaling factor
        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_fn_bsr_light, sf=downscale_f)

        # if the degradation method is not `bsrgan` or `bsrgan_light`, it selected the appropriate interpolation function from a dictory based on the degrdation parameter.
        else:
            interpolation_fn = {
            "cv_nearest": cv2.INTER_NEAREST,
            "cv_bilinear": cv2.INTER_LINEAR,
            "cv_bicubic": cv2.INTER_CUBIC,
            "cv_area": cv2.INTER_AREA,
            "cv_lanczos": cv2.INTER_LANCZOS4,
            "pil_nearest": PIL.Image.NEAREST,
            "pil_bilinear": PIL.Image.BILINEAR,
            "pil_bicubic": PIL.Image.BICUBIC,
            "pil_box": PIL.Image.BOX,
            "pil_hamming": PIL.Image.HAMMING,
            "pil_lanczos": PIL.Image.LANCZOS,
            }[degradation]

            # sets `self.pil_interpolation` to `True` if the degradation method start with `pil_` (indicating pillow interpolation)
            self.pil_interpolation = degradation.startswith("pil_")

            # if pillow interpolation is used, sets `self.degradation_process` to a partial fun to `Tf.resize` with target size and selected interpolation method
            if self.pil_interpolation:
                self.degradation_process = partial(TF.resize,
                                                   size=self.LR_size,
                                                   interpolation=interpolation_fn)
                
            # if opencv interpolation is used, set `self.degradation_process` to an `albumentations.SmallestMaxSize` object with the target size and the selected interpolation method.
            else:
                self.degradation_process = albumentations.SmallestMaxSize(max_size=self.LR_size,
                                                                          interpolation=interpolation_fn)
                




    def __len__(self):
        return len(self.base)
    


    def __getitem__(self, index):
        example = self.base[index]
        # opens the image file using the file path
        image = Image.open(example["file_path_"])

        # convert the image to RGB mode if it is not already in RGB
        if not image.mode == "RGB":
            image = image.convert("RGB")

        # converts the image to numpy array with dtype 
        image = np.array(image).astype(np.uint8)

        # computes the min side length (height or width) of the image
        min_side_len = min(image.shape[:2])
        # computes the crop size by multiply the min size length by a random factor sampled uniformly between `self.min_crop_f` and `self.max_crop_f`
        crop_side_len = min_side_len * np.random.uniform(self.min_crop_f, self.max_crop_f, size=None)
        crop_side_len = int(crop_side_len)

        # if `self.center_crop` is `True`, initializes an `albumentations.CenterCrop` object with the comptued crop size.
        if self.center_crop:
            self.cropper = albumentations.CenterCrop(height=crop_side_len,
                                                     width=crop_side_len)
            
        # if `self.center_crop` is `False`, initializes an `albumentations.RandomCrop` object with the computed crop size.
        else:
            self.cropper = albumentations.RandomCrop(height=crop_side_len,
                                                     width=crop_side_len)
            

        # Applies the cropping operation to the image and resize the croped image to the target hight-resolution size 
        image = self.cropper(image=image)["image"]
        image = self.image_rescaler(image)["image"]

        # if pillow interpolation is used. converts the image to pil image applies the degradation process and convert it back to Numpy array
        if self.pil_interpolation:
            image_pil = PIL.Image.fromarray(image)
            LR_image = self.degradation_process(image_pil)
            LR_image = np.array(LR_image).astype(np.uint8)

        # if opencv interpolation is used, applies the degradation process directly to the Numpy array.
        else:
            LR_image = self.degradation_process(image)["image"]

        # normalize the hight-resolution image to range `-1, 1`
        example["image"] = (image/127.5 - 1.0).astype(np.float32)
        # normalize the Low-resolution image to range `-1, 1`
        example["LR_image"] = (LR_image/127.5 - 1.0).astype(np.float32)

        # Return the processed example containing the Hight-reolution and low-resolution images.
        return example
    



class ImageNetSRTrain(ImageNetSR):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)


    def get_base(self):
        with open("data/imagenet_train_hr_indices.p", "rb") as f:
            indices = pickle.load(f)

        dset = ImageNetTrain(process_image=False)
        return Subset(dset, indices)
    



    

    # this method handles the preparation of the dataset
    def _prepare(self):

        # if `data_root` is provided sets `self.root` to the path combining `data_root` and the dataset name `self.NAME`
        if self.data_root:
            self.root = os.path.join(self.data_root, self.NAME)

        # if `data_root` is not provided, sets `self.root` to default cached directory. It uses the 
        # `XDG_CACHE_HOME` envirnoment variable if available, otherwise it default to `~/.cache` like this name: `C:\Users\username\.cache\autoencoders\data\ILSVRC2012_train`
        else:
            cachedir = os.environ.get("XDG_CACHE_HOME", os.path.expanduser("~/.cache"))
            self.root = os.path.join(cachedir, "autoencoders/data", self.NAME)

        # Sets `self.datadir` to the path where the dataset files will be stored (inside the `data` subdirectory of `self.root`)
        self.datadir = os.path.join(self.root, "data")
        # sets `self.txt_filelist` to the path of a text file that will store the list of all image file paths
        self.txt_filelist = os.path.join(self.root, "filelist.txt")
        # sets `self.expected_length` to the expected number of images in the dataset (1281167 for ImageNet 2012 training set.)
        self.expected_length = 1281167
        # Retrieves a configuration value for `random_crop` from self.config if not found it defaults to True
        self.random_crop = retrieve(self.config,
                                    "ImageNetTrain/random_crop",
                                    default=True)


        

        # check if the dataset is already prepared using a utility function `tdu.is_prepared`. if not, it proceeds with the preparation.
        if not tdu.is_prepared(self.root):
            print("Preparing data {} in {}".format(self.NAME, self.root))


            datadir = self.datadir

            # checks if the `datadir` directory does not exist.
            if not os.path.exists(datadir):
                # constrants the path to the dataset tar file `ILSVRC2012_img_train.tar`
                path = os.path.join(self.root, self.FILES[0])
                # check if the tar file does not exist or its size does not match the expected size 
                if not os.path.exists(path) or not os.path.getsize(path) == self.SIZES[0]:
                    # use the `academictorrents` library to downlaod the dataset using the provided hash `AT_HASH`. Ensure the downloaded file matched the expected path.
                    import academictorrents as at
                    atpath = at.get(self.AT_HASH, datastore=self.root)
                    assert atpath == path 

                # prints a message indicating that the tar file is being extracted.
                print("Extracting {} to {}".format(path, datadir))
                # create the `datadir` directory if it does not exist.
                os.makedirs(datadir, exist_ok=True)
                # Extract the contents of the tar file into datadir
                with tarfile.open(path, "r:") as tar:
                    tar.extractall(path=datadir)

                # print a message indicating the sub-tar files are bing extracted.
                print("Extracting sub-tars.")
                # finds all `.tar` file in datadir and sorts them
                subpaths = sorted(glob.glob(os.path.join(datadir, "*.tar")))
                # iterator over each sub-tar file using tqdm
                for subpath in tqdm(subpaths):
                    # constructs the directory name by removing the `.tar` extension from the sub-tar file name.
                    subdir = subpath[:-len(".tar")]
                    # creates the subdirectory if it does not exist
                    os.makedirs(subdir, exist_ok=True)

                    # Extracts the contents of the sub-tar file into the corresponding subdirectory
                    with tarfile.open(subpath, 'r:') as tar:
                        tar.extractall(path=subdir)


        # finds all `.jpeg` file in `datadir` and its subdirectory
        filelist = glob.glob(os.path.join(datadir, "**", "*.JPEG"))
        # converts the file paths to relative paths (relative to `datadir`)
        filelist = [os.path.relpath(p, start=datadir) for p in filelist]
        # sorts the list of file paths
        filelist = sorted(filelist)
        # join the file paths into a single string, with each path on a new line.
        filelist = "\n".join(filelist)+"\n"

        # write the file list to `self.txt_filelist`
        with open(self.txt_filelist, "w") as f:
            f.write(filelist)

        # marks the dataset as prepared using the utility function `tdu.mark_prepared`
        tdu.mark_prepared(self.root)

