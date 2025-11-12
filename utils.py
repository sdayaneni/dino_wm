import os
import torch
from torch import nn
import random
import argparse
import numpy as np
from PIL import Image
from omegaconf import OmegaConf
from typing import Callable, Dict
import psutil

def get_ram_usage():
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024 * 1024)  # Memory usage in MB

def get_available_ram():
    mem = psutil.virtual_memory()
    return mem.available / (1024 * 1024 * 1024)  # Available memory in MB

def dict_to_namespace(cfg_dict):
    args = argparse.Namespace()
    for key in cfg_dict:
        setattr(args, key, cfg_dict[key])
    return args

def move_to_device(dct, device):
    for key, value in dct.items():
        if isinstance(value, torch.Tensor):
            dct[key] = value.to(device)
    return dct

def slice_trajdict_with_t(data_dict, start_idx=0, end_idx=None, step=1):
    if end_idx is None:
        end_idx = max(arr.shape[1] for arr in data_dict.values())
    return {key: arr[:, start_idx:end_idx:step, ...] for key, arr in data_dict.items()}

def concat_trajdict(dcts):
    full_dct = {}
    for k in dcts[0].keys():
        if isinstance(dcts[0][k], np.ndarray):
            full_dct[k] = np.concatenate([dct[k] for dct in dcts], axis=1)
        elif isinstance(dcts[0][k], torch.Tensor):
            full_dct[k] = torch.cat([dct[k] for dct in dcts], dim=1)
        else:
            raise TypeError(f"Unsupported data type: {type(dcts[0][k])}")
    return full_dct

def aggregate_dct(dcts):
    full_dct = {}
    for dct in dcts:
        for key, value in dct.items():
            if key not in full_dct:
                full_dct[key] = []
            full_dct[key].append(value)
    for key, value in full_dct.items():
        if isinstance(value[0], torch.Tensor):
            full_dct[key] = torch.stack(value)
        else:
            full_dct[key] = np.stack(value)
    return full_dct

def sample_tensors(tensors, n, indices=None):
    if indices is None:
        b = tensors[0].shape[0]
        indices = torch.randperm(b)[:n]
    indices = torch.tensor(indices)
    for i, tensor in enumerate(tensors):
        if tensor is not None:
            tensors[i] = tensor[indices]
    return tensors


def cfg_to_dict(cfg):
    cfg_dict = OmegaConf.to_container(cfg)
    for key in cfg_dict:
        if isinstance(cfg_dict[key], list):
            cfg_dict[key] = ",".join(cfg_dict[key])
    return cfg_dict

def reduce_dict(f: Callable, d: Dict):
    return {k: reduce_dict(f, v) if isinstance(v, dict) else f(v) for k, v in d.items()}

def seed(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_loader(path):
    with open(path, "rb") as f:
        with Image.open(f) as img:
            return img.convert("RGB")

def clear_cache():
    torch.cuda.empty_cache()

def has_layer_norm(model):
    return any(isinstance(module, nn.LayerNorm) for _, module in model.named_modules())

def init_wandb_watch(wandb_logger, model_trainer, wandb_watch_log_freq):
    if not has_layer_norm(model_trainer.model):
        wandb_logger.watch(model_trainer.model, log="all", log_freq = wandb_watch_log_freq)
    
    else: # all of complex below code is to get around the issue where wandb watch with layer norm has 'AttributeError: 'NoneType' object has no attribute 'data'' when logging gradients...
        non_layernorm_container = nn.Module()
        layernorm_container = nn.Module()

        non_ln_modules = {}
        ln_modules = {}

        for name, module in model_trainer.model.named_modules():
            if name == "": # skips top level model
                continue
            safe_name = name.replace(".", "_") # model cant contain '.' in name

            if isinstance(module, nn.LayerNorm):
                ln_modules[safe_name] = module
            else:
                # Only add modules that don't contain LayerNorm as submodules
                has_ln_child = any(isinstance(child, nn.LayerNorm) 
                                for child in module.modules())
                if not has_ln_child:
                    non_ln_modules[safe_name] = module

        for name, module in non_ln_modules.items():
            non_layernorm_container.add_module(name, module)

        for name, module in ln_modules.items():
            layernorm_container.add_module(name, module)

        # print("\nNon-LayerNorm modules:")
        # for name, _ in non_layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        # print("\nLayerNorm modules:")
        # for name, _ in layernorm_container.named_modules():
        #     if name != "":  # Skip the container itself
        #         print(f"  - {name}")

        wandb_logger.watch(non_layernorm_container, log="all", log_freq=wandb_watch_log_freq)
        wandb_logger.watch(layernorm_container, log="parameters", log_freq=wandb_watch_log_freq)