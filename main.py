import argparse, torch
import numpy as np
from torch.utils.data import Dataset, DataLoader, Subset, random_split
from ldm.dataset.base import Txt2ImgIterableBaseDataset

from pytorch_lightning.trainer import Trainer
from pytorch_lightning.utilities.argparse import add_argparse_args

def get_parser(**parser_kwargs):

    def str2bool(v):
        if isinstance(v, bool):
            return v 
        
        if v.lower() in ("yes", "true", "t", "y", "1"):
            return True
        
        elif v.lower() in ("no", "false", "f", "n", "0"):
            return False
        
        else:
            raise argparse.ArgumentTypeError("Boolean value expected.")
        

    parser = argparse.ArgumentParser(**parser_kwargs)
    parser.add_argument(
        "-n",
        "--name",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="postfix for logdir"
    )
    parser.add_argument(
        "-r",
        "--resume",
        type=str,
        const=True,
        default="",
        nargs="?",
        help="resume from logdir or checkpoint in logdir",
    )
    parser.add_argument(
        "-b",
        "--base",
        nargs="*",
        metavar="config.yaml",
        help="paths to base configs. Loaded from left-to-right. "
             "Parameters can be overwritten or added with command-line options of the form `--key value`.",
        default=list(),
    )
    parser.add_argument(
        "-t",
        "--train",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="train",
    )
    parser.add_argument(
        "--no-test",
        type=str2bool,
        const=True,
        default=False,
        nargs="?",
        help="disable test",
    )
    parser.add_argument(
        "-p",
        "--project",
        help="name of new or path to existing project"
    )
    parser.add_argument(
        "-d",
        "--debug",
        type=str2bool,
        nargs="?",
        const=True,
        default=False,
        help="enable post-mortem debugging",
    )
    parser.add_argument(
        "-s",
        "--seed",
        type=int,
        default=23,
        help="seed for seed_everything",
    )
    parser.add_argument(
        "-f",
        "--postfix",
        type=str,
        default="",
        help="post-postfix for default name",
    )
    parser.add_argument(
        "-l",
        "--logdir",
        type=str,
        default="logs",
        help="directory for logging dat shit",
    )
    parser.add_argument(
        "--scale_lr",
        type=str2bool,
        nargs="?",
        const=True,
        default=True,
        help="scale base-lr by ngpu * batch_size * n_accumulate",
    )
    return parser




class WrappedDataset(Dataset):

    """ Wraps an arbitrary object with __len__ and __getitem__ into a pytorch dataset."""

    def __init__(self, dataset):
        self.data = dataset

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx]
    

def worker_init_fn(_):

    worker_info = torch.utils.data.get_worker_info()

    dataset = worker_info.dataset
    worker_id = worker_info.id 

    if isinstance(dataset, Txt2ImgIterableBaseDataset):
        split_size = dataset.num_records // worker_info.num_workers
        # reset num_records to the true number to retain reliable length information
        dataset.sample_ids = dataset.valid_ids[worker_id * split_size:(worker_id + 1) * split_size]
        current_id = np.random.choice(len(np.random.get_state()[1]), 1)
        return np.random.seed(np.random.get_state()[1][current_id] + worker_id)
    else:
        return np.random.seed(np.random.get_state()[1][0] + worker_id)

import pytorch_lightning as pl
from functools import partial
from ldm.modules.utils import instantiate_from_config

class DataModuleFromConfig(pl.LightningDataModule):

    def __init__(self, 
                 batch_size, 
                 train=None,
                 validation=None,
                 test=None,
                 predict=None,
                 wrap=False,
                 num_workers=None,
                 shuffle_test_loader=False,
                 use_worker_init_fn=False,
                 shuffle_val_dataloader=False):
        
        super().__init__()
        self.batch_size = batch_size
        self.dataset_configs = dict()
        self.num_workers = num_workers if num_workers is not None else batch_size * 2 
        self.use_worker_init_fn = use_worker_init_fn

        if train is not None:
            self.dataset_configs["train"] = train 
            self.train_dataloader = self._train_dataloader

        if validation is not None:
            self.dataset_configs["validation"] = validation
            self.val_dataloader = partial(self._val_dataloader, shuffle=shuffle_val_dataloader)

        if test is not None:
            self.dataset_configs["test"] = test 
            self.test_dataloader = partial(self._test_dataloader, shuffle=shuffle_test_loader)

        if predict is not None:
            self.dataset_configs["predict"] = predict
            self.predict_dataloader = self._predict_dataloader

        self.wrap = wrap 


    def prepare_data(self):
        for data_cfg in self.dataset_configs.values():
            instantiate_from_config(data_cfg)


    def setup(self, stage = None):
        self.datasets = dict(
            (k, instantiate_from_config(self.dataset_configs[k]))
            for k in self.dataset_configs
        )

        if self.wrap:
            for k in self.datasets:
                self.datasets[k] = WrappedDataset(self.datasets[k])



        


    def _train_dataloader(self):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["train"], batch_size=self.batch_size,
                          num_workers=self.num_workers, shuffle=False if is_iterable_dataset else True,
                          worker_init_fn=init_fn)

    def _val_dataloader(self, shuffle=False):
        if isinstance(self.datasets['validation'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["validation"],
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          worker_init_fn=init_fn,
                          shuffle=shuffle)

    def _test_dataloader(self, shuffle=False):
        is_iterable_dataset = isinstance(self.datasets['train'], Txt2ImgIterableBaseDataset)
        if is_iterable_dataset or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None

        # do not shuffle dataloader for iterable dataset
        shuffle = shuffle and (not is_iterable_dataset)

        return DataLoader(self.datasets["test"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn, shuffle=shuffle)

    def _predict_dataloader(self, shuffle=False):
        if isinstance(self.datasets['predict'], Txt2ImgIterableBaseDataset) or self.use_worker_init_fn:
            init_fn = worker_init_fn
        else:
            init_fn = None
        return DataLoader(self.datasets["predict"], batch_size=self.batch_size,
                          num_workers=self.num_workers, worker_init_fn=init_fn)
    



def nondefault_trainer_args(opt):
    # create an argumentparser
    parser = argparse.ArgumentParser()
    # Add Trainer Arguments
    parser = add_argparse_args(Trainer, parser)
    # Parse Empty Arguments
    args = parser.parse_args([])
    # Compare opt with default args
    # return sorted(k for k in vars(args) if getattr(opt, k) != getattr(args, k))
    return sorted(k for k in vars(args) if hasattr(opt, k) and getattr(opt, k) != getattr(args,k))



if __name__ == "__main__":
    # python -m main --base config/config.yaml
    
    import datetime, sys, os, glob
    from pytorch_lightning import seed_everything
    from omegaconf import OmegaConf



    now = datetime.datetime.now().strftime("%Y-%m-%dT%H-%M-%S")
    sys.path.append(os.getcwd())

    parser = get_parser()
    parser = add_argparse_args(Trainer, parser)
    # print(parser)
    

    # opt, unknown = parser.parse_known_args()
    # # print("opt", opt)
    # # print("unknown", unknown)

    
    #     # init and save config 
    # configs = [OmegaConf.load(cfg) for cfg in opt.base]
    #             # print("configs", configs)
    # cli = OmegaConf.from_dotlist(unknown)
    #     # print("cli: ---->", cli)

    # config = OmegaConf.merge(*configs, cli)
    # print("configs: ------> ", configs)

    # lightning_config = config.pop("lightning", OmegaConf.create())
    # print("lightning_config ----->", lightning_config)

    #     # merge trainer cli with config 
    # trainer_config = lightning_config.get("trainer", OmegaConf.create())
    # print("trainer_config: ---->", trainer_config)

    # trainer_config["accelerator"] = "ddp"
    # for k in nondefault_trainer_args(opt):
    #     trainer_config[k] = getattr(opt, k)

    # if not "gpus" in trainer_config:
    #     del trainer_config["accelerator"]
    #     cpu = True 

    # else:
    #     gpuinfo = trainer_config["gpus"]
    #     print(f"Running on GPUs {gpuinfo}")
    #     cpu = False

    # trainer_opt = argparse.Namespace(**trainer_config)
    # lightning_config.trainer = trainer_config

    #################
    config = OmegaConf.load("E:\\stable_diffusion\\config\\config.yaml")
    print("config: --->", config.model)

        # model 
    model = instantiate_from_config(config)
    print("model: --->", model)


        

        








        
        

    
