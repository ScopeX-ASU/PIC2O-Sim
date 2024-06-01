from typing import Dict, Tuple

import torch
import torch.nn as nn
from neuralop.models import FNO
from pyutils.config import configs
from pyutils.datasets import get_dataset
from pyutils.lr_scheduler.warmup_cosine_restart import CosineAnnealingWarmupRestarts
from pyutils.optimizer.sam import SAM
from pyutils.typing import DataLoader, Optimizer, Scheduler
from torch.types import Device
from torch.utils.data import Sampler
import random

from core.datasets import (
    FDTDDataset,
)
from core.models import *

from .utils import (
    DAdaptAdam,
    NormalizedMSELoss,
    maskedNMSELoss,
    NL2NormLoss,
    maskedNL2NormLoss,
)

__all__ = [
    "make_dataloader",
    "make_model",
    "make_weight_optimizer",
    "make_arch_optimizer",
    "make_optimizer",
    "make_scheduler",
    "make_criterion",
]


def collate_fn_keep_spatial_res(batch):
    data, targets = zip(*batch)
    new_size = []
    for item in data:
        if item["device_type"].item() == 1 or item["device_type"].item() == 5: # which means it is a mmi and resolution = 20
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item()*0.75)),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item()*0.75)),)
        else:
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item())),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item())),)
        new_size.append(newSize)
    Hight = [int(round(item["Ez"].shape[-2]*item["scaling_factor"].item())) for item in data]
    Width = [int(round(item["Ez"].shape[-1]*item["scaling_factor"].item())) for item in data]
    maxHight = max(Hight)
    maxWidth = max(Width)
    if maxWidth % 2 == 1:
        maxWidth += 1 ## make sure the width is even so that there won't be any mismatch between fourier and inverse fourier
    # Pad all items to the max length and max width using zero padding
    # eps should use background value to padding
    # fields should use zero padding
    # weight masks should use zero padding
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0) # dummy batch dim and then remove it
        item["eps"] = torch.nn.functional.interpolate(item["eps"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["source"] = torch.nn.functional.interpolate(item["source"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["mseWeight"] = torch.nn.functional.interpolate(item["mseWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["src_mask"] = torch.nn.functional.interpolate(item["src_mask"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["epsWeight"] = torch.nn.functional.interpolate(item["epsWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["padding_mask"] = torch.ones_like(item["eps"], device=item["eps"].device)
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)

    hightPatchSize_bot = [(maxHight-item["Ez"].shape[-2])//2 for item in data]
    hightPatchSize_top = [maxHight-item["Ez"].shape[-2]-(maxHight-item["Ez"].shape[-2])//2 for item in data]
    widthPatchSize_left = [(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    widthPatchSize_right = [maxWidth-item["Ez"].shape[-1]-(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["eps"] = torch.nn.functional.pad(item["eps"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=item["eps_bg"].item())
        item["padding_mask"] = torch.nn.functional.pad(item["padding_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["source"] = torch.nn.functional.pad(item["source"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["mseWeight"] = torch.nn.functional.pad(item["mseWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["src_mask"] = torch.nn.functional.pad(item["src_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["epsWeight"] = torch.nn.functional.pad(item["epsWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)

    Ez_data = torch.stack([item["Ez"] for item in data], dim=0)
    source_data = torch.stack([item["source"] for item in data], dim=0)
    eps_data = torch.stack([item["eps"] for item in data], dim=0)
    padding_mask_data = torch.stack([item["padding_mask"] for item in data], dim=0)
    mseWeight_data = torch.stack([item["mseWeight"] for item in data], dim=0)
    src_mask_data = torch.stack([item["src_mask"] for item in data], dim=0)
    epsWeight_data = torch.stack([item["epsWeight"] for item in data], dim=0)
    device_type_data = torch.stack([item["device_type"] for item in data], dim=0)
    eps_bg_data = torch.stack([item["eps_bg"] for item in data], dim=0)
    grid_step_data = torch.stack([item["grid_step"] for item in data], dim=0)

    Ez_target = torch.stack([item["Ez"] for item in targets], dim=0)
    std_target = torch.stack([item["std"] for item in targets], dim=0)
    mean_target = torch.stack([item["mean"] for item in targets], dim=0)
    stdDacayRate = torch.stack([item["stdDacayRate"] for item in targets], dim=0)

    raw_data = {
        "Ez": Ez_data,
        "source": source_data,
        "eps": eps_data,
        "padding_mask": padding_mask_data,
        "mseWeight": mseWeight_data,
        "src_mask": src_mask_data,
        "epsWeight": epsWeight_data,
        "device_type": device_type_data,
        "eps_bg": eps_bg_data,
        "grid_step": grid_step_data,
    }
    raw_targets = {
        "Ez": Ez_target,
        "std": std_target,
        "mean": mean_target,
        "stdDacayRate": stdDacayRate,
    }

    return raw_data, raw_targets

def collate_fn_keep_spatial_res_pad_to_256(batch):
    # Extract all items for each key and compute the max length
    data, targets = zip(*batch)
    new_size = []
    for item in data:
        if item["device_type"].item() == 1 or item["device_type"].item == 5: # train seperately for metaline, so no need to consider the resolution mismatch
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item()*0.75)),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item()*0.75)),)
        else:
            newSize = (int(round(item["Ez"].shape[-2]*item["scaling_factor"].item())),
                    int(round(item["Ez"].shape[-1]*item["scaling_factor"].item())),)
        new_size.append(newSize)
    maxHight = 256
    maxWidth = 256
    if item["device_type"].item() == 5: # which means it is a metaline
        maxHight = 168
        maxWidth = 168
    # Pad all items to the max length and max width using zero padding
    # eps should use background value to padding
    # fields should use zero padding
    # weight masks should use zero padding
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0) # dummy batch dim and then remove it
        item["eps"] = torch.nn.functional.interpolate(item["eps"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["source"] = torch.nn.functional.interpolate(item["source"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["mseWeight"] = torch.nn.functional.interpolate(item["mseWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["src_mask"] = torch.nn.functional.interpolate(item["src_mask"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["epsWeight"] = torch.nn.functional.interpolate(item["epsWeight"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)
        item["padding_mask"] = torch.ones_like(item["eps"], device=item["eps"].device)
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.interpolate(item["Ez"].unsqueeze(0), size=new_size[idx], mode="bilinear").squeeze(0)

    hightPatchSize_bot = [(maxHight-item["Ez"].shape[-2])//2 for item in data]
    hightPatchSize_top = [maxHight-item["Ez"].shape[-2]-(maxHight-item["Ez"].shape[-2])//2 for item in data]
    widthPatchSize_left = [(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    widthPatchSize_right = [maxWidth-item["Ez"].shape[-1]-(maxWidth-item["Ez"].shape[-1])//2 for item in data]
    for idx, item in enumerate(data):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["eps"] = torch.nn.functional.pad(item["eps"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=item["eps_bg"].item())
        item["padding_mask"] = torch.nn.functional.pad(item["padding_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["source"] = torch.nn.functional.pad(item["source"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["mseWeight"] = torch.nn.functional.pad(item["mseWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["src_mask"] = torch.nn.functional.pad(item["src_mask"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        item["epsWeight"] = torch.nn.functional.pad(item["epsWeight"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)
        
    
    for idx, item in enumerate(targets):
        item["Ez"] = torch.nn.functional.pad(item["Ez"], (widthPatchSize_left[idx], widthPatchSize_right[idx], hightPatchSize_bot[idx], hightPatchSize_top[idx]), mode='constant', value=0)

    Ez_data = torch.stack([item["Ez"] for item in data], dim=0)
    source_data = torch.stack([item["source"] for item in data], dim=0)
    eps_data = torch.stack([item["eps"] for item in data], dim=0)
    padding_mask_data = torch.stack([item["padding_mask"] for item in data], dim=0)
    mseWeight_data = torch.stack([item["mseWeight"] for item in data], dim=0)
    src_mask_data = torch.stack([item["src_mask"] for item in data], dim=0)
    epsWeight_data = torch.stack([item["epsWeight"] for item in data], dim=0)
    device_type_data = torch.stack([item["device_type"] for item in data], dim=0)
    eps_bg_data = torch.stack([item["eps_bg"] for item in data], dim=0)
    grid_step_data = torch.stack([item["grid_step"] for item in data], dim=0)

    Ez_target = torch.stack([item["Ez"] for item in targets], dim=0)
    std_target = torch.stack([item["std"] for item in targets], dim=0)
    mean_target = torch.stack([item["mean"] for item in targets], dim=0)
    stdDacayRate = torch.stack([item["stdDacayRate"] for item in targets], dim=0)

    raw_data = {
        "Ez": Ez_data,
        "source": source_data,
        "eps": eps_data,
        "padding_mask": padding_mask_data,
        "mseWeight": mseWeight_data,
        "src_mask": src_mask_data,
        "epsWeight": epsWeight_data,
        "device_type": device_type_data,
        "eps_bg": eps_bg_data,
        "grid_step": grid_step_data,
    }
    raw_targets = {
        "Ez": Ez_target,
        "std": std_target,
        "mean": mean_target,
        "stdDacayRate": stdDacayRate,
    }

    return raw_data, raw_targets

class MySampler(Sampler):
    def __init__(self, data_source, shuffle=False):
        self.indices = sorted(range(len(data_source)), key=lambda x: data_source[x][0]["area"].item())
        self.shuffle = shuffle
        # self.indices is a list of indices sorted by the area of the devices

    def __iter__(self):
        if self.shuffle:
            group_size = 2500
            group_num = len(self.indices) // group_size
            for i in range(group_num+1):
                group_indices = self.indices[i*group_size:(i+1)*group_size] if i != group_num else self.indices[i*group_size:]
                random.shuffle(group_indices)
                if i != group_num:
                    self.indices[i*group_size:(i+1)*group_size] = group_indices
                else:
                    self.indices[i*group_size:] = group_indices
        else:
            pass
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)

def make_dataloader(
    name: str = None,
    splits=["train", "valid", "test"],
    train_noise_cfg=None,
    out_frames=None,
) -> Tuple[DataLoader, DataLoader]:
    name = (name or configs.dataset.name).lower()
    if name == "fdtd":
        train_dataset, validation_dataset, test_dataset = (
            (
                FDTDDataset(
                    root=configs.dataset.root,
                    device_list=configs.dataset.device_list,
                    split=split,
                    test_ratio=configs.dataset.test_ratio,
                    train_valid_split_ratio=configs.dataset.train_valid_split_ratio,
                    processed_dir=configs.dataset.processed_dir,
                    in_frames=configs.dataset.in_frames,
                    out_frames=configs.dataset.out_frames,
                    offset_frames=configs.dataset.offset_frames,
                    img_size=configs.dataset.img_height,
                    mixup_cfg=configs.dataset.augment if split == "train" else None,
                    batch_strategy=configs.dataset.batch_strategy,
                )
                if split in splits
                else None
            )
            for split in ["train", "valid", "test"]
        )
    else:
        train_dataset, test_dataset = get_dataset(
            name,
            configs.dataset.img_height,
            configs.dataset.img_width,
            dataset_dir=configs.dataset.root,
            transform=configs.dataset.transform,
        )
        validation_dataset = None

    if train_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res":
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res,
                            sampler = MySampler(train_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif train_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res_pad_to_256":
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res_pad_to_256,
                            sampler = MySampler(train_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif train_dataset is not None and configs.dataset.batch_strategy == "resize_and_padding_to_square":
        train_loader = torch.utils.data.DataLoader(
                            dataset=train_dataset,
                            batch_size=configs.run.batch_size,
                            shuffle=int(configs.dataset.shuffle),
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                        )
    else:
        train_loader = None

    if validation_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res":
        validation_loader = torch.utils.data.DataLoader(
                            dataset=validation_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res,
                            sampler = MySampler(validation_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif validation_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res_pad_to_256":
        validation_loader = torch.utils.data.DataLoader(
                            dataset=validation_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res_pad_to_256,
                            sampler = MySampler(validation_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif validation_dataset is not None and configs.dataset.batch_strategy == "resize_and_padding_to_square":
        validation_loader = torch.utils.data.DataLoader(
                            dataset=validation_dataset,
                            batch_size=configs.run.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                        )
    else:
        validation_loader = None

    if test_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res":
        test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res,
                            sampler = MySampler(test_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif test_dataset is not None and configs.dataset.batch_strategy == "keep_spatial_res_pad_to_256":
        test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset,
                            batch_size=configs.run.batch_size,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                            collate_fn=collate_fn_keep_spatial_res_pad_to_256,
                            sampler = MySampler(test_dataset, shuffle=int(configs.dataset.shuffle)),
                        )
    elif test_dataset is not None and configs.dataset.batch_strategy == "resize_and_padding_to_square":
        test_loader = torch.utils.data.DataLoader(
                            dataset=test_dataset,
                            batch_size=configs.run.batch_size,
                            shuffle=False,
                            pin_memory=True,
                            num_workers=configs.dataset.num_workers,
                            prefetch_factor=8,
                            persistent_workers=True,
                        )
    else:
        test_loader = None

    return train_loader, validation_loader, test_loader


def make_model(device: Device, random_state: int = None, **kwargs) -> nn.Module:
    if "cnnfdtd" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_size=configs.dataset.img_height,
            in_channels=1 + configs.dataset.in_frames + configs.model.out_channels,
            out_channels=configs.model.out_channels,
            in_frames=configs.dataset.in_frames,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            stride_list=configs.model.stride_list,
            padding_list=configs.model.padding_list,
            hidden_list=configs.model.hidden_list,
            act_func=configs.model.act_func,
            norm=configs.model.norm,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            device=device,
            aux_head=configs.model.aux_head,
            aux_stide=configs.model.aux_stride,
            aux_padding=configs.model.aux_padding,
            aux_kernel_size_list=configs.model.aux_kernel_size_list,
            field_norm_mode=configs.model.field_norm_mode,
            **kwargs,
        ).to(device)
    elif "fouriercnn" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_size=configs.dataset.img_height,
            in_channels=1 + configs.dataset.offset_frames + configs.model.out_channels,
            out_channels=configs.model.out_channels,
            in_frames=configs.dataset.in_frames,
            offset_frames=configs.dataset.offset_frames,
            input_cfg=configs.model.input_cfg,
            guidance_generator_cfg = configs.model.guidance_generator_cfg,
            encoder_cfg=configs.model.encoder_cfg,
            backbone_cfg=configs.model.backbone_cfg,
            decoder_cfg=configs.model.decoder_cfg,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            device=device,
            aux_head=configs.model.aux_head,
            aux_stide=configs.model.aux_stride,
            aux_padding=configs.model.aux_padding,
            aux_kernel_size_list=configs.model.aux_kernel_size_list,
            field_norm_mode=configs.model.field_norm_mode,
            num_iters = configs.model.num_iters,
            eps_lap = configs.model.input_cfg.eps_lap,
            pac = configs.model.encoder_cfg.pac or configs.model.decoder_cfg.pac or configs.model.backbone_cfg.pac,
            max_propagating_filter = configs.model.max_propagating_filter,
            **kwargs,
        ).to(device)
        model.reset_parameters(random_state)
    elif "multistepdynamiccnn" in configs.model.name.lower():
        model = eval(configs.model.name)(
            img_size=configs.dataset.img_height,
            in_channels=1 + configs.dataset.offset_frames + configs.model.out_channels,
            out_channels=configs.model.out_channels,
            in_frames=configs.dataset.in_frames,
            offset_frames=configs.dataset.offset_frames,
            input_cfg=configs.model.input_cfg,
            history_encoder_cfg=configs.model.history_encoder_cfg,
            guidance_generator_cfg = configs.model.guidance_generator_cfg,
            encoder_cfg=configs.model.encoder_cfg,
            backbone_cfg=configs.model.backbone_cfg,
            decoder_cfg=configs.model.decoder_cfg,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            device=device,
            aux_head=configs.model.aux_head,
            aux_stide=configs.model.aux_stride,
            aux_padding=configs.model.aux_padding,
            aux_kernel_size_list=configs.model.aux_kernel_size_list,
            field_norm_mode=configs.model.field_norm_mode,
            num_iters = configs.model.num_iters,
            eps_lap = configs.model.input_cfg.eps_lap,
            pac = configs.model.encoder_cfg.pac or configs.model.decoder_cfg.pac or configs.model.backbone_cfg.pac,
            share_encoder = configs.model.share_encoder,
            share_decoder = configs.model.share_decoder,
            share_backbone = configs.model.share_backbone,
            share_history_encoder = configs.model.share_history_encoder,
            if_pass_history = configs.model.if_pass_history,
            if_pass_grad = configs.model.if_pass_grad,
            **kwargs,
        ).to(device)
        model.reset_parameters(random_state)
    elif "neuralop_fno" in configs.model.name.lower():
        model = FNO(
            n_modes=configs.model.mode_list,
            in_channels=configs.dataset.in_channels,
            out_channels=configs.dataset.out_channels,
            hidden_channels=configs.model.hidden_dim,
            projection_channels=configs.model.proj_dim,
            n_layers=configs.model.n_layers,
        ).to(device)
    elif "ffno" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            in_frames = configs.model.in_frames,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            domain_size=configs.model.domain_size,
            grid_step=configs.model.grid_step,
            buffer_width=configs.model.buffer_width,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            with_cp=configs.model.with_cp,
            device=device,
            conv_stem=configs.model.conv_stem,
            aug_path=configs.model.aug_path,
            ffn=configs.model.ffn,
            ffn_dwconv=configs.model.ffn_dwconv,
            encoder_cfg = configs.model.encoder_cfg,
            backbone_cfg = configs.model.backbone_cfg,
            decoder_cfg = configs.model.decoder_cfg,
            **kwargs,
        ).to(device)
    elif "fno3d" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_frames=configs.dataset.in_frames,
            img_size=configs.dataset.img_height,
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding="none",
            with_cp=configs.model.with_cp,
            device=device,
            **kwargs,
        ).to(device)
    elif "kno2d" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_frames=configs.model.in_frames,
            img_size=configs.dataset.img_height,
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            with_cp=configs.model.with_cp,
            kno_alg=configs.model.kno_alg,
            kno_r=configs.model.kno_r,
            # T1=configs.model.T1,
            transform=configs.model.transform,
            # num_iters=configs.model.num_iters,
            device=device,
            encoder_cfg=configs.model.encoder_cfg,
            backbone_cfg=configs.model.backbone_cfg,
            decoder_cfg=configs.model.decoder_cfg,
            **kwargs,
        ).to(device)
        # model.reset_parameters(random_state)
    elif "neurolight" in configs.model.name.lower():
        model = eval(configs.model.name)(
            in_frames=configs.dataset.in_frames,
            in_channels=configs.model.in_channels,
            out_channels=configs.model.out_channels,
            dim=configs.model.dim,
            kernel_list=configs.model.kernel_list,
            kernel_size_list=configs.model.kernel_size_list,
            padding_list=configs.model.padding_list,
            mode_list=configs.model.mode_list,
            act_func=configs.model.act_func,
            domain_size=configs.model.domain_size,
            grid_step=configs.model.grid_step,
            buffer_width=configs.model.buffer_width,
            dropout_rate=configs.model.dropout_rate,
            drop_path_rate=configs.model.drop_path_rate,
            aux_head=configs.model.aux_head,
            aux_head_idx=configs.model.aux_head_idx,
            pos_encoding=configs.model.pos_encoding,
            with_cp=configs.model.with_cp,
            device=device,
            conv_stem=configs.model.conv_stem,
            aug_path=configs.model.aug_path,
            ffn=configs.model.ffn,
            ffn_dwconv=configs.model.ffn_dwconv,
            encoder_cfg=configs.model.encoder_cfg,
            backbone_cfg=configs.model.backbone_cfg,
            decoder_cfg=configs.model.decoder_cfg,
            **kwargs,
        ).to(device)
    elif "sinenet" in configs.model.name.lower():
        model = eval(configs.model.name)(
        n_input_scalar_components=configs.model.n_input_scalar_components,
        n_input_vector_components=configs.model.n_input_vector_components,
        n_output_scalar_components=configs.model.n_output_scalar_components,
        n_output_vector_components=configs.model.n_output_vector_components,
        time_history=configs.model.time_history,
        in_frames=configs.model.in_frames,
        time_future=configs.model.time_future,
        hidden_channels=configs.model.hidden_channels,
        padding_mode=configs.model.padding_mode,
        activation="gelu",
        num_layers=configs.model.num_layers,
        num_waves=configs.model.num_waves,
        num_blocks=configs.model.num_blocks,
        norm=True,
        mult=2,
        residual=True,
        wave_residual=True,
        disentangle=True,
        down_pool=True,
        avg_pool=True,
        up_interpolation=True,
        interpolation_mode='bicubic',
        par1=None
        ).to(device)
    else:
        model = None
        raise NotImplementedError(f"Not supported model name: {configs.model.name}")
    return model


def make_optimizer(params, name: str = None, configs=None) -> Optimizer:
    if name == "sgd":
        optimizer = torch.optim.SGD(
            params,
            lr=configs.lr,
            momentum=configs.momentum,
            weight_decay=configs.weight_decay,
            nesterov=True,
        )
    elif name == "adam":
        optimizer = torch.optim.Adam(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            betas=getattr(configs, "betas", (0.9, 0.999)),
        )
    elif name == "adamw":
        optimizer = torch.optim.AdamW(
            params,
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    elif name == "dadaptadam":
        optimizer = DAdaptAdam(
            params,
            lr=configs.lr,
            betas=getattr(configs, "betas", (0.9, 0.999)),
            weight_decay=configs.weight_decay,
        )
    elif name == "sam_sgd":
        base_optimizer = torch.optim.SGD
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.5),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
            momenum=0.9,
        )
    elif name == "sam_adam":
        base_optimizer = torch.optim.Adam
        optimizer = SAM(
            params,
            base_optimizer=base_optimizer,
            rho=getattr(configs, "rho", 0.001),
            adaptive=getattr(configs, "adaptive", True),
            lr=configs.lr,
            weight_decay=configs.weight_decay,
        )
    else:
        raise NotImplementedError(name)

    return optimizer


def make_scheduler(optimizer: Optimizer, name: str = None) -> Scheduler:
    name = (name or configs.scheduler.name).lower()
    if name == "constant":
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lambda epoch: 1
        )
    elif name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=int(configs.run.n_epochs),
            eta_min=float(configs.scheduler.lr_min),
        )
    elif name == "cosine_warmup":
        scheduler = CosineAnnealingWarmupRestarts(
            optimizer,
            first_cycle_steps=configs.run.n_epochs,
            max_lr=configs.optimizer.lr,
            min_lr=configs.scheduler.lr_min,
            warmup_steps=int(configs.scheduler.warmup_steps),
        )
    elif name == "exp":
        scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=configs.scheduler.lr_gamma
        )
    else:
        raise NotImplementedError(name)

    return scheduler


def make_criterion(name: str = None, cfg=None) -> nn.Module:
    name = (name or configs.criterion.name).lower()
    cfg = cfg or configs.criterion
    if name == "mse":
        criterion = nn.MSELoss()
    elif name == "nmse":
        criterion = NormalizedMSELoss()
    elif name == "nl2norm":
        criterion = NL2NormLoss()
    elif name == "masknl2norm":
        criterion = maskedNL2NormLoss(weighted_frames=cfg.weighted_frames, weight=cfg.weight, if_spatial_mask=cfg.if_spatial_mask)
    elif name == "masknmse":
        criterion = maskedNMSELoss(weighted_frames=cfg.weighted_frames, weight=cfg.weight, if_spatial_mask=cfg.if_spatial_mask)
    else:
        raise NotImplementedError(name)
    return criterion
