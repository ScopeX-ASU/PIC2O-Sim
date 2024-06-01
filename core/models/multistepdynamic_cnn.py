import copy
from collections import OrderedDict
from typing import List, Optional, Tuple

import torch
import torch.nn as nn
from torch import nn
from torch.functional import Tensor
from torch.types import Device

from .constant import *
from .pde_base import PDE_NN_BASE, ConvBlock, LaplacianBlock

__all__ = ["MultiStepDynamicCNN"]


class MultiStepDynamicCNN(PDE_NN_BASE):
    def __init__(
        self,
        img_size: int = 256,
        in_channels: int = 1,
        out_channels: int = 2,
        in_frames: int = 8,
        offset_frames: int = 8,
        input_cfg: dict = {},
        history_encoder_cfg: dict = {},
        guidance_generator_cfg: dict = {},
        encoder_cfg: dict = {},
        backbone_cfg: dict = {},
        decoder_cfg: dict = {},
        dropout_rate: float = 0.0,
        drop_path_rate: float = 0.0,
        device: Device = torch.device("cuda:0"),
        aux_head: bool = False,
        aux_stide: List[int] = [2, 2, 2],
        aux_padding: List[int] = [1, 1, 1],
        aux_kernel_size_list: List[int] = [3, 3, 3],
        field_norm_mode: str = "max",
        num_iters: int = 1,
        share_encoder: bool = False,
        share_backbone: bool = False,
        share_decoder: bool = False,
        share_history_encoder: bool = False,
        eps_lap: bool = False,
        pac: bool = False,
        if_pass_history: bool = False,
        if_pass_grad: bool = False,
        **kwargs,
    ):

        super().__init__(
            encoder_cfg=encoder_cfg,
            backbone_cfg=backbone_cfg,
            decoder_cfg=decoder_cfg,
            **kwargs,
        )

        """
        input: the solution of the coefficient function and locations (a(x, y), x, y)
        input shape: (batchsize, f = 1+8+50, x, y)
        output: the solution of next 50 frames (bs, f = 50, x ,y)
        """
        self.img_size = img_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_frames = in_frames
        self.offset_frames = offset_frames
        self.history_encoder_cfg = history_encoder_cfg
        self.guidance_generator_cfg = guidance_generator_cfg
        self.encoder_cfg = encoder_cfg
        self.backbone_cfg = backbone_cfg
        self.decoder_cfg = decoder_cfg
        self.dropout_rate = dropout_rate
        self.drop_path_rate = drop_path_rate

        self.aux_head = aux_head
        self.aux_stide = aux_stide
        self.aux_padding = aux_padding
        self.aux_kernel_size_list = aux_kernel_size_list

        self.field_norm_mode = field_norm_mode
        self.num_iters = num_iters
        self.eps_lap = eps_lap
        self.pac = pac
        self.share_encoder = share_encoder
        self.share_backbone = share_backbone
        self.share_decoder = share_decoder
        self.share_history_encoder = share_history_encoder
        self.if_pass_history = if_pass_history
        self.if_pass_grad = if_pass_grad
        if self.if_pass_history:
            assert (
                self.num_iters > 1
            ), "if pass hidden state, num_iters must be larger than 1"

        self.input_cfg = input_cfg
        if self.input_cfg.input_mode == "eps_E0_Ji":
            self.in_channels = 1 + self.offset_frames + self.out_channels
            if not self.input_cfg.include_src:
                self.in_channels -= 1
        elif self.input_cfg.input_mode == "E0_Ji":
            self.in_channels = self.in_frames + self.out_channels
        elif self.input_cfg.input_mode == "eps_E0_lap_Ji":
            self.in_channels = 1 + self.offset_frames + 1 + self.out_channels
            if not self.input_cfg.include_src:
                self.in_channels -= 1

        self.device = device

        if self.backbone_cfg.share_weight:
            assert self.backbone_cfg.num_shared_layers > 1

        self.build_layers()
        self.set_max_trainable_iter(self.num_iters)

    def build_modules(self, name, cfg, in_channels):
        features = OrderedDict()
        for idx, out_channels in enumerate(cfg.kernel_list, 0):
            layer_name = "conv" + str(idx + 1)
            in_channels = in_channels if (idx == 0) else cfg.kernel_list[idx - 1]
            features[layer_name] = ConvBlock(
                in_channels,
                out_channels,
                cfg.kernel_size_list[idx],
                cfg.stride_list[idx],
                cfg.padding_list[idx],
                cfg.dilation_list[idx],
                cfg.groups_list[idx],
                bias=True,
                conv_cfg=cfg.conv_cfg,
                norm_cfg=(cfg.norm_cfg if cfg.norm_list[idx] else None),
                act_cfg=(cfg.act_cfg if cfg.act_list[idx] else None),
                residual=cfg.residual[idx],
                se=cfg.se[idx],
                pac=cfg.pac,
                with_cp=cfg.with_cp,
                device=self.device,
                if_pre_dwconv=cfg.if_pre_dwconv,
            )

        self.register_module(name, nn.Sequential(features))
        return getattr(self, name), out_channels

    def build_layers(self):
        if self.pac:
            assert (
                self.input_cfg.input_mode == "E0_Ji"
            ), "when pac, input_mode must be E0_Ji"
            self.guidance_generator, guidance_channels = self.build_modules(
                "guidance_generator", self.guidance_generator_cfg, 1
            )

        if self.if_pass_history and (not self.share_history_encoder):
            history_encoder = [
                self.build_modules(
                    "history_encoder",
                    self.history_encoder_cfg,
                    self.backbone_cfg.kernel_list[-1]
                    + self.encoder_cfg.kernel_list[-1],
                )
                for _ in range(self.num_iters - 1)
            ]
        elif self.if_pass_history and self.share_history_encoder:
            history_encoder = [
                self.build_modules(
                    "history_encoder",
                    self.history_encoder_cfg,
                    self.backbone_cfg.kernel_list[-1]
                    + self.encoder_cfg.kernel_list[-1],
                )
            ] * (self.num_iters - 1)
        if self.if_pass_history:
            self.history_encoder = nn.ModuleList([m[0] for m in history_encoder])

        if not self.share_encoder:
            encoder = [
                self.build_modules("encoder", self.encoder_cfg, self.in_channels)
                for _ in range(self.num_iters)
            ]
            object_list = set([id(item) for item in encoder])
            assert self.num_iters == len(
                object_list
            ), "the encoder must be different with each other"
        else:
            encoder = [
                self.build_modules("encoder", self.encoder_cfg, self.in_channels)
            ] * self.num_iters
            object_list = set([id(item) for item in encoder])
            assert len(object_list) == 1, "the encoder must be the same one"
        self.encoder = nn.ModuleList([m[0] for m in encoder])
        out_channels = encoder[0][1]

        if "3d" in self.backbone_cfg.conv_cfg["type"]:
            new_out_channels = 1
        else:
            new_out_channels = out_channels

        if not self.share_backbone:
            backbone = [
                self.build_modules("backbone", self.backbone_cfg, new_out_channels)
                for _ in range(self.num_iters)
            ]
            object_list = set([id(item) for item in backbone])
            assert self.num_iters == len(
                object_list
            ), "the backbone must be different with each other"
        else:
            backbone = [
                self.build_modules("backbone", self.backbone_cfg, new_out_channels)
            ] * self.num_iters
            object_list = set([id(item) for item in backbone])
            assert len(object_list) == 1, "the backbone must be the same one"
        self.backbone = nn.ModuleList([m[0] for m in backbone])
        ret_out_channels = backbone[0][1]

        if "3d" in self.backbone_cfg.conv_cfg["type"]:
            new_out_channels = out_channels
        else:
            new_out_channels = ret_out_channels

        if (
            self.decoder_cfg.kernel_list[-1] != self.out_channels
        ):  # make sure the last layer is the out_channels
            self.decoder_cfg.kernel_list[-1] = self.out_channels

        if not self.share_decoder:
            decoder = [
                self.build_modules("decoder", self.decoder_cfg, new_out_channels)[0]
                for _ in range(self.num_iters)
            ]
            object_list = set([id(item) for item in decoder])
            assert self.num_iters == len(
                object_list
            ), "the decoder must be different with each other"
        else:
            decoder = [
                self.build_modules("decoder", self.decoder_cfg, new_out_channels)[0]
            ] * self.num_iters
            object_list = set([id(item) for item in decoder])
            assert len(object_list) == 1, "the decoder must be the same one"
        self.decoder = nn.ModuleList(decoder)

        if "lap" in self.input_cfg.input_mode:
            self.laplacian = LaplacianBlock(device=self.device)

    def get_scaling_factor(self, input_fields, src):
        if self.field_norm_mode == "max":
            abs_values = torch.abs(input_fields)
            scaling_factor = abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
        elif self.field_norm_mode == "max99":
            p99 = input_fields.flatten(1).quantile(0.99995, dim=-1, keepdim=True)
            scaling_factor = p99.unsqueeze(-1).unsqueeze(-1)
        elif self.field_norm_mode == "std":
            scaling_factor = 15 * input_fields.std(dim=(1, 2, 3), keepdim=True)
        elif self.field_norm_mode == "none":
            scaling_factor = torch.ones(
                (input_fields.size(0), 1, 1, 1), device=input_fields.device
            )
        elif self.field_norm_mode == "max_w_src":
            E0_src = torch.cat([input_fields, src], dim=1)
            abs_values = torch.abs(E0_src)
            scaling_factor = abs_values.amax(dim=(1, 2, 3), keepdim=True) + 1e-6
        else:
            raise NotImplementedError

        return scaling_factor  # [bs, 1, 1, 1]

    def preprocess(self, x: Tensor) -> Tuple[Tensor, ...]:
        ## obtain the input fields and the source fields from input data
        eps = 1 / x[:, 0:1].square()
        input_fields = x[:, 1 : 1 + self.in_frames]
        srcs = x[:, 1 + self.in_frames :, ...].chunk(self.num_iters, dim=1)
        return eps, input_fields, srcs

    def set_max_trainable_iter(self, max_trainable_iter: int):
        self.max_trainable_iter = max_trainable_iter
        assert 1 <= self.max_trainable_iter <= self.num_iters

    def load_parameters(self, src: int, dst: int, optimizer) -> None:
        # with multiple iterations, load the parameters from previous to next one
        if not self.share_encoder:
            if hasattr(optimizer, "state"):
                for p_src, p_dst in zip(
                    self.encoder[src].parameters(), self.encoder[dst].parameters()
                ):
                    optimizer.state[p_dst] = copy.deepcopy(optimizer.state[p_src])
            self.encoder[dst].load_state_dict(self.encoder[src].state_dict())
        if not self.share_backbone:
            if hasattr(optimizer, "state"):
                for p_src, p_dst in zip(
                    self.backbone[src].parameters(), self.backbone[dst].parameters()
                ):
                    optimizer.state[p_dst] = copy.deepcopy(optimizer.state[p_src])
            self.backbone[dst].load_state_dict(self.backbone[src].state_dict())
        if not self.share_decoder:
            if hasattr(optimizer, "state"):
                for p_src, p_dst in zip(
                    self.decoder[src].parameters(), self.decoder[dst].parameters()
                ):
                    optimizer.state[p_dst] = copy.deepcopy(optimizer.state[p_src])
            self.decoder[dst].load_state_dict(self.decoder[src].state_dict())
        if not self.share_history_encoder and src > 1:
            if hasattr(optimizer, "state"):
                for p_src, p_dst in zip(
                    self.history_encoder[src - 1].parameters(),
                    self.history_encoder[dst - 1].parameters(),
                ):
                    optimizer.state[p_dst] = copy.deepcopy(optimizer.state[p_src])
            self.history_encoder[dst - 1].load_state_dict(
                self.history_encoder[src - 1].state_dict()
            )

    def forward(
        self,
        x,
        src_mask: Optional[Tensor] = None,
        padding_mask: Optional[Tensor] = None,
    ):
        eps, input_fields, srcs = self.preprocess(x)
        normalization_factor = []
        outs = []
        guidance = None
        for iter, src in enumerate(srcs):
            if iter >= self.max_trainable_iter:
                break
            scaling_factor = self.get_scaling_factor(input_fields, src)
            if self.input_cfg.input_mode == "eps_E0_Ji":
                x = torch.cat(
                    [eps, input_fields / scaling_factor, src / scaling_factor], dim=1
                )
                if not self.input_cfg.include_src:
                    x = x[:, :-1, ...]
            elif self.input_cfg.input_mode == "E0_Ji":
                assert self.pac, "PAC must be true"
                assert (
                    self.guidance_generator is not None
                ), "guidance_generator must be provided"
                if guidance is None:
                    guidance, _ = self.guidance_generator((eps, None))
                else:
                    guidance = guidance.detach()
                x = torch.cat([input_fields, src], dim=1)
                if self.field_norm_mode != "none":
                    x.div_(scaling_factor)
            elif self.input_cfg.input_mode == "eps_E0_lap_Ji":
                divergence = self.laplacian(input_fields[:, -1:, ...] / scaling_factor)
                if self.eps_lap:
                    divergence = divergence * eps
                x = torch.cat(
                    [
                        eps,
                        input_fields / scaling_factor,
                        divergence,
                        src / scaling_factor,
                    ],
                    dim=1,
                )
                if not self.input_cfg.include_src:
                    x = x[:, :-1, ...]  # drop the last src
            x, _ = self.encoder[iter]((x, None))
            if self.if_pass_history and iter > 0:
                x = torch.cat([x, hidden_state], dim=1)
                x, _ = self.history_encoder[iter - 1]((x, None))
            if self.backbone_cfg.pac:
                x, _ = self.backbone[iter]((x, guidance))
                if self.if_pass_history:
                    hidden_state = x.clone().detach()
            else:
                x, _ = self.backbone[iter]((x, None))
                if self.if_pass_history:
                    hidden_state = x.clone().detach()
            if self.decoder_cfg.pac:
                x, _ = self.decoder[iter]((x, guidance))
            else:
                x, _ = self.decoder[iter]((x, None))
            src_mask = src_mask >= 0.5
            src = src[:, -self.out_channels :, ...]
            if self.field_norm_mode != "none":
                src = src / scaling_factor
            out = torch.where(src_mask, src, x)
            out = out * padding_mask

            normalization_factor.append(scaling_factor)
            outs.append(out)
            ## update input fields for next iteration
            input_fields = (
                out[:, -self.in_frames :, ...]
                if self.if_pass_grad
                else out[:, -self.in_frames :, ...].detach()
            )
            if self.field_norm_mode != "none":
                input_fields = input_fields * scaling_factor

        outs = torch.cat(outs, dim=1)

        normalization_factor = (
            torch.stack(normalization_factor, 1)
            .expand(-1, -1, self.out_channels, -1, -1)
            .flatten(1, 2)
        )

        return outs, normalization_factor
