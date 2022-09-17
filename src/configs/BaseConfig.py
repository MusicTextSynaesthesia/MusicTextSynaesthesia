import argparse
import platform
import random
import os
import sys
import torch
import time
import numpy as np


IN_DEV = (platform.system() == "Linux")


class _Config(argparse.Namespace):
    def hierarchical_print(self):
        viewed, s = set(), ""
        for cur_class in self.__class__.__mro__:
            if cur_class is _Config:
                break
            s += f"\n\n____{cur_class.__name__}____"
            parent = super(cur_class, self)
            for attr in dir(self):
                if attr not in viewed and attr[:1] != "_" and getattr(self, attr) != getattr(parent, attr, None):
                    viewed.add(attr)
                    attr_value = getattr(self, attr)
                    s_attr = str(attr_value) if not callable(attr_value) else attr_value.__name__
                    if issubclass(attr_value.__class__, _Config):
                        s_attr = s_attr[1:].replace("\n", "\n    ")
                    s += f"\n{attr}: {s_attr}"
        return s
    __str__ = __repr__ = hierarchical_print


class GenerateConfig(_Config):
    max_len = 200
    do_sample = True
    num_beams = 3
    top_k = 50
    num_return_sequences = 3


class BaseConfig(_Config):
    # paths
    device = "cuda:0"
    model_name = ""
    dataset_dir = "../data"
    tokenizer = "bert-base-uncased"
    logfile = "stdout.log"

    # basic setting
    train_split = [slice(0, 1564)]
    valid_split = [slice(1564, None)]
    tags = ["mode", "instrument", "tempo", "ensemble"]
    batch_size = 8
    gradient_checkpoint = True
    load_pretrained = True
    load_frozen_pretrained = False
    pool_size = 20
    music_dim = 64
    latent_dim = 768
    optimizer = torch.optim.AdamW
    lr = 5e-5
    weight_decay = 5
    epoch = 5
    sample_rate = 16000
    seed = 42

    save_last_checkpoint = False
    valid_in_train = True

    generate_config = GenerateConfig()

    def __init__(self, train=True, train_val_split=None, **kwargs):
        super(BaseConfig, self).__init__()
        for k in kwargs:
            setattr(self, k, kwargs[k])
        if train_val_split:
            self.train_split, self.valid_split = train_val_split

        # output
        self.model_save_dir = f"./checkpoints/{self.model_name}/"
        self.output_dir = f"{self.model_save_dir}/output"
        self.logfile = f"./logs/{self.model_name}_{time.strftime('%Y-%m-%d-%H-%M-%S%z')}_{self.logfile}"

        if train:
            os.makedirs("./logs", exist_ok=True)
            os.makedirs(self.model_save_dir, exist_ok=True)
            # hook logger
            std_logger = open(self.logfile, "w")
            hook_std_log(sys.stdout, std_logger)
            # hook_std_log(sys.stderr, std_logger)
        else:
            os.makedirs(self.output_dir, exist_ok=True)

        # set seed
        torch.cuda.set_device(self.device)
        torch.cuda.init()
        self.set_seed()

        print(self)

    def set_seed(self):
        torch.manual_seed(self.seed)
        np.random.seed(self.seed)
        random.seed(self.seed)


class HookF:
    def __init__(self, src_func, dst_func):
        self.src_func = src_func
        self.dst_func = dst_func

    def __call__(self, *args, **kwargs):
        self.dst_func(*args, **kwargs)
        self.src_func(*args, **kwargs)


def hook_std_log(src, dst):
    for func_name in ("write", "writelines", "writable", "seek", "tell", "flush"):
        hooked = getattr(src, func_name)
        if not isinstance(hooked, HookF):
            hooked = HookF(getattr(src, func_name), getattr(dst, func_name))
        hooked.dst_func = getattr(dst, func_name)
        setattr(src, func_name, hooked)
