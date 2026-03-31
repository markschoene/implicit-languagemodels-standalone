import json
import os
from typing import Optional

import numpy as np
import torch
import torch.nn as nn


def load_checkpoint(pretrained_model_name_or_path: str) -> dict:
    """
    Load the checkpoint from the specified path. This function handles both sharded and non-sharded checkpoints.
    """
    # Determine whether we have a sharded checkpoint (using an index file)
    index_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin.index.json")
    if os.path.isfile(index_path):
        # Load the index file
        with open(index_path, "r") as f:
            index = json.load(f)
        weight_map = index.get("weight_map", {})
        state_dict = {}

        # Collect unique shard filenames from the index
        shard_files = set(weight_map.values())
        for shard_file in shard_files:
            shard_path = os.path.join(pretrained_model_name_or_path, shard_file)
            if not os.path.isfile(shard_path):
                raise FileNotFoundError(f"Expected shard file {shard_path} not found.")
            # Load each shard and update our state_dict
            shard_state_dict = torch.load(shard_path, map_location="cpu")
            state_dict.update(shard_state_dict)
    else:
        # Otherwise, load a single checkpoint file.
        checkpoint_path = os.path.join(pretrained_model_name_or_path, "pytorch_model.bin")
        safetensors_path = os.path.join(pretrained_model_name_or_path, "model.safetensors")
        if os.path.isfile(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        elif os.path.isfile(safetensors_path):
            from safetensors.torch import load_file
            state_dict = load_file(safetensors_path, device="cpu")
        else:
            raise FileNotFoundError(
                f"No checkpoint found at {pretrained_model_name_or_path}. "
                f"Expected pytorch_model.bin or model.safetensors."
            )
    return state_dict


class CosineWarmupScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_iterations, lr_decay_iters=None, min_lr=None, decay_lr=True):
        self.warmup = warmup_steps
        self.max_num_iters = max_iterations
        self.decay_lr = decay_lr
        self.min_lr_coeff = 0.1  # per Chinchilla
        # use lr_decay_iters< max_iterations to start constant lr phase before max_iterations
        self.lr_decay_iters = lr_decay_iters if lr_decay_iters is not None else max_iterations
        self.min_lr = min_lr  # if None uses  min_lr_coeff of peak lr

        super().__init__(optimizer)

    def get_lr(self):

        def get_min_lr(base_lr):
            return self.min_lr if self.min_lr is not None else base_lr * self.min_lr_coeff

        if not self.decay_lr:
            return list(self.base_lrs)

        if self.last_epoch < self.warmup:
            # Linear warmup phase
            return [base_lr * self.last_epoch / self.warmup for base_lr in self.base_lrs]
        elif self.last_epoch > self.lr_decay_iters:
            # Constant learning rate phase
            return [get_min_lr(base_lr) for base_lr in self.base_lrs]
        else:
            # Cosine decay phase
            decay_ratio = (self.last_epoch - self.warmup) / (self.lr_decay_iters - self.warmup)
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1 + np.cos(np.pi * decay_ratio))
            return [(get_min_lr(base_lr)) + coeff * (base_lr - get_min_lr(base_lr)) for base_lr in self.base_lrs]


class LinearWarmupConstantDecayScheduler(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, warmup_steps, max_iterations, lr_decay_iters=None):
        self.warmup = warmup_steps
        self.max_num_iters = max_iterations
        self.cooldown_ratio = 0.2  # as per https://arxiv.org/pdf/2405.18392
        self.lr_decay_iters = (
            lr_decay_iters if lr_decay_iters is not None else int(max_iterations * (1 - self.cooldown_ratio))
        )
        super().__init__(optimizer)

    def get_lr(self):
        if self.last_epoch < self.warmup:
            # Linear warmup phase
            return [base_lr * self.last_epoch / self.warmup for base_lr in self.base_lrs]
        elif self.last_epoch <= self.lr_decay_iters:
            # Constant learning rate phase
            return list(self.base_lrs)
        else:
            # 1 - sqrt decay phase
            decay_ratio = (self.last_epoch - self.lr_decay_iters) / (self.max_num_iters - self.lr_decay_iters)
            assert 0 <= decay_ratio <= 1
            coeff = 1 - np.sqrt(decay_ratio)
            return [base_lr * coeff for base_lr in self.base_lrs]


