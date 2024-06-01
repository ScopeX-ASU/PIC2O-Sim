import torch
from torch import Tensor
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
import torch.fft
import random

#################################################
#
# Utilities
#
#################################################

from typing import TYPE_CHECKING, Any

import torch.optim
import logging
import torch.distributed as dist

if TYPE_CHECKING:
    from torch.optim.optimizer import _params_t
else:
    _params_t = Any
class DAdaptAdam(torch.optim.Optimizer):
    r"""
    Implements Adam with D-Adaptation automatic step-sizes.
    Leave LR set to 1 unless you encounter instability.

    To scale the learning rate differently for each layer, set the 'layer_scale'
    for each parameter group. Increase (or decrease) from its default value of 1.0
    to increase (or decrease) the learning rate for that layer relative to the
    other layers.

    Arguments:
        params (iterable):
            Iterable of parameters to optimize or dicts defining parameter groups.
        lr (float):
            Learning rate adjustment parameter. Increases or decreases the D-adapted learning rate.
        betas (Tuple[float, float], optional): coefficients used for computing
            running averages of gradient and its square (default: (0.9, 0.999))
        eps (float):
            Term added to the denominator outside of the root operation to improve numerical stability. (default: 1e-8).
        weight_decay (float):
            Weight decay, i.e. a L2 penalty (default: 0).
        log_every (int):
            Log using print every k steps, default 0 (no logging).
        decouple (boolean):
            Use AdamW style decoupled weight decay
        use_bias_correction (boolean):
            Turn on Adam's bias correction. Off by default.
        d0 (float):
            Initial D estimate for D-adaptation (default 1e-6). Rarely needs changing.
        growth_rate (float):
            prevent the D estimate from growing faster than this multiplicative rate.
            Default is inf, for unrestricted. Values like 1.02 give a kind of learning
            rate warmup effect.
        fsdp_in_use (bool):
            If you're using sharded parameters, this should be set to True. The optimizer
            will attempt to auto-detect this, but if you're using an implementation other
            than PyTorch's builtin version, the auto-detection won't work.
    """

    def __init__(
        self,
        params,
        lr=1.0,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        log_every=0,
        decouple=False,
        use_bias_correction=False,
        d0=1e-6,
        growth_rate=float("inf"),
        fsdp_in_use=False,
    ):
        if not 0.0 < d0:
            raise ValueError("Invalid d0 value: {}".format(d0))
        if not 0.0 < lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 < eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))

        if decouple:
            print(f"Using decoupled weight decay")

        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            d=d0,
            k=0,
            layer_scale=1.0,
            numerator_weighted=0.0,
            log_every=log_every,
            growth_rate=growth_rate,
            use_bias_correction=use_bias_correction,
            decouple=decouple,
            fsdp_in_use=fsdp_in_use,
        )
        self.d0 = d0
        super().__init__(params, defaults)

    @property
    def supports_memory_efficient_fp16(self):
        return False

    @property
    def supports_flat_params(self):
        return True

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        sk_l1 = 0.0

        group = self.param_groups[0]
        use_bias_correction = group["use_bias_correction"]
        numerator_weighted = group["numerator_weighted"]
        beta1, beta2 = group["betas"]
        k = group["k"]

        d = group["d"]
        lr = max(group["lr"] for group in self.param_groups)

        if use_bias_correction:
            bias_correction = ((1 - beta2 ** (k + 1)) ** 0.5) / (1 - beta1 ** (k + 1))
        else:
            bias_correction = 1

        dlr = d * lr * bias_correction

        growth_rate = group["growth_rate"]
        decouple = group["decouple"]
        log_every = group["log_every"]
        fsdp_in_use = group["fsdp_in_use"]

        sqrt_beta2 = beta2 ** (0.5)

        numerator_acum = 0.0

        for group in self.param_groups:
            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]
            group_lr = group["lr"]
            r = group["layer_scale"]

            if group_lr not in [lr, 0.0]:
                raise RuntimeError(
                    f"Setting different lr values in different parameter groups "
                    "is only supported for values of 0. To scale the learning "
                    "rate differently for each layer, set the 'layer_scale' value instead."
                )

            for p in group["params"]:
                if p.grad is None:
                    continue
                if hasattr(p, "_fsdp_flattened"):
                    fsdp_in_use = True

                grad = p.grad.data

                # Apply weight decay (coupled variant)
                if decay != 0 and not decouple:
                    grad.add_(p.data, alpha=decay)

                state = self.state[p]

                # State initialization
                if "step" not in state:
                    state["step"] = 0
                    state["s"] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of gradient values
                    state["exp_avg"] = torch.zeros_like(p.data).detach()
                    # Exponential moving average of squared gradient values
                    state["exp_avg_sq"] = torch.zeros_like(p.data).detach()

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                s = state["s"]

                if group_lr > 0.0:
                    denom = exp_avg_sq.sqrt().add_(eps)
                    numerator_acum += (
                        r
                        * dlr
                        * torch.dot(grad.flatten(), s.div(denom).flatten()).item()
                    )

                    # Adam EMA updates
                    exp_avg.mul_(beta1).add_(grad, alpha=r * dlr * (1 - beta1))
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    s.mul_(sqrt_beta2).add_(grad, alpha=dlr * (1 - sqrt_beta2))
                    sk_l1 += r * s.abs().sum().item()

            ######

        numerator_weighted = (
            sqrt_beta2 * numerator_weighted + (1 - sqrt_beta2) * numerator_acum
        )
        d_hat = d

        # if we have not done any progres, return
        # if we have any gradients available, will have sk_l1 > 0 (unless \|g\|=0)
        if sk_l1 == 0:
            return loss

        if lr > 0.0:
            if fsdp_in_use:
                dist_tensor = torch.zeros(2).cuda()
                dist_tensor[0] = numerator_weighted
                dist_tensor[1] = sk_l1
                dist.all_reduce(dist_tensor, op=dist.ReduceOp.SUM)
                global_numerator_weighted = dist_tensor[0]
                global_sk_l1 = dist_tensor[1]
            else:
                global_numerator_weighted = numerator_weighted
                global_sk_l1 = sk_l1

            d_hat = global_numerator_weighted / ((1 - sqrt_beta2) * global_sk_l1)
            d = max(d, min(d_hat, d * growth_rate))

        if log_every > 0 and k % log_every == 0:
            logging.info(
                f"lr: {lr} dlr: {dlr} d_hat: {d_hat}, d: {d}. sk_l1={global_sk_l1:1.1e} numerator_weighted={global_numerator_weighted:1.1e}"
            )

        for group in self.param_groups:
            group["numerator_weighted"] = numerator_weighted
            group["d"] = d

            decay = group["weight_decay"]
            k = group["k"]
            eps = group["eps"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data

                state = self.state[p]

                exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

                state["step"] += 1

                denom = exp_avg_sq.sqrt().add_(eps)

                # Apply weight decay (decoupled variant)
                if decay != 0 and decouple:
                    p.data.add_(p.data, alpha=-decay * dlr)

                ### Take step
                p.data.addcdiv_(exp_avg, denom, value=-1)

            group["k"] = k + 1

        return loss
class DeterministicCtx:
    def __init__(self, seed: int | None = None) -> None:
        self.seed = seed
        self.random_state = None
        self.numpy_random_state = None
        self.torch_random_state = None
        self.torch_cuda_random_state = None

    def __enter__(self):
        # Save the current states
        self.random_state = random.getstate()
        self.numpy_random_state = np.random.get_state()
        self.torch_random_state = torch.random.get_rng_state()
        if torch.cuda.is_available():
            self.torch_cuda_random_state = torch.cuda.get_rng_state()
        
        # Set deterministic behavior based on the seed
        set_torch_deterministic(self.seed)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Restore the saved states
        random.setstate(self.random_state)
        np.random.set_state(self.numpy_random_state)
        torch.random.set_rng_state(self.torch_random_state)
        if torch.cuda.is_available():
            torch.cuda.set_rng_state(self.torch_cuda_random_state)

def set_torch_deterministic(seed: int = 0) -> None:
    seed = int(seed) % (2**32)
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.cuda.manual_seed_all(seed)

class NormalizedMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean"):
        super(NormalizedMSELoss, self).__init__()

        self.reduce = reduce

    def forward(self, x, y, mask):
        one_mask = mask != 0

        error_energy = torch.norm((x - y)*one_mask, p=2, dim=(-1, -2))
        field_energy = torch.norm(y*one_mask, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy/field_energy).mean()
    
class NL2NormLoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean"):
        super(NL2NormLoss, self).__init__()

        self.reduce = reduce

    def forward(self, x, y, mask):
        one_mask = mask != 0

        error_energy = torch.norm((x - y)*one_mask, p=2, dim=(-1, -2))
        field_energy = torch.norm(y*one_mask, p=2, dim=(-1, -2)) + 1e-6
        return (error_energy/field_energy).mean()
class maskedNMSELoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean", weighted_frames = 10, weight = 1, if_spatial_mask = True):
        super(maskedNMSELoss, self).__init__()

        self.reduce = reduce
        self.weighted_frames = weighted_frames
        self.weight = weight
        self.if_spatial_mask = if_spatial_mask

    def forward(self, x, y, mask, num_iters = 1):
        frame_mask = None
        ones_mask = mask != 0
        if self.weighted_frames != 0:
            assert x.shape[-3] // num_iters >= 10
            assert x.shape[-3] % num_iters == 0
            frame_mask = torch.ones((1, x.shape[1], 1, 1)).to(x.device)
            single_prediction_len = x.shape[-3] // num_iters
            for i in range(num_iters):
                frame_mask[:, -single_prediction_len*(i-1)-self.weighted_frames:-single_prediction_len*(i-1), :, :] = 3
        assert (frame_mask is not None) or self.if_spatial_mask, "if mask NMSE, either frame_mask or spatial_mask should be True"
        if self.if_spatial_mask:
            error_energy = torch.norm((x - y)*mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y*mask, p=2, dim=(-1, -2))
        else:
            error_energy = torch.norm((x - y)*ones_mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y*ones_mask, p=2, dim=(-1, -2))
        if frame_mask is not None:
            return (((error_energy/field_energy)*frame_mask).mean(dim=(-1))).mean()
        else:
            return ((error_energy/field_energy).mean(dim=(-1))).mean()

class maskedNL2NormLoss(torch.nn.modules.loss._Loss):
    def __init__(self, reduce="mean", weighted_frames = 10, weight = 1, if_spatial_mask = True):
        super(maskedNL2NormLoss, self).__init__()

        self.reduce = reduce
        self.weighted_frames = weighted_frames
        self.weight = weight
        self.if_spatial_mask = if_spatial_mask

    def forward(self, x, y, mask, num_iters = 1):
        frame_mask = None
        ones_mask = mask != 0
        if self.weighted_frames != 0:
            assert x.shape[-3] // num_iters >= 10
            assert x.shape[-3] % num_iters == 0
            frame_mask = torch.ones((1, x.shape[1], 1, 1)).to(x.device)
            single_prediction_len = x.shape[-3] // num_iters
            for i in range(num_iters):
                frame_mask[:, -single_prediction_len*(i-1)-self.weighted_frames:-single_prediction_len*(i-1), :, :] = 3
        assert (frame_mask is not None) or self.if_spatial_mask, "if mask nl2norm, either frame_mask or spatial_mask should be True"
        if self.if_spatial_mask:
            error_energy = torch.norm((x - y)*mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y*mask, p=2, dim=(-1, -2))
        else:
            error_energy = torch.norm((x - y)*ones_mask, p=2, dim=(-1, -2))
            field_energy = torch.norm(y*ones_mask, p=2, dim=(-1, -2))
        if frame_mask is not None:
            return (((error_energy/field_energy)*frame_mask).mean(dim=(-1))).mean()
        else:
            return ((error_energy/field_energy).mean(dim=(-1))).mean()

def normalize(x):
    if isinstance(x, np.ndarray):
        x_min, x_max = np.percentile(x, 5), np.percentile(x, 95)
        x = np.clip(x, x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    else:
        x_min, x_max = torch.quantile(x, 0.05), torch.quantile(x, 0.95)
        x = x.clamp(x_min, x_max)
        x = (x - x_min) / (x_max - x_min)
    return x

def print_stat(x, dist=False):
    total_number = None
    distribution = None
    if dist:
        total_number = x.numel()
        distribution = torch.histc(x, bins=10, min=float(x.min()), max=float(x.max()))
    if isinstance(x, torch.Tensor):
        print(
            f"min = {x.min().data.item():-15f} max = {x.max().data.item():-15f} mean = {x.mean().data.item():-15f} std = {x.std().data.item():-15f}\n total num = {total_number} distribution = {distribution}"
        )
    elif isinstance(x, np.ndarray):
        print(
            f"min = {np.min(x):-15f} max = {np.max(x):-15f} mean = {np.mean(x):-15f} std = {np.std(x):-15f}"
        )

def plot_compare_vis(
    epsilon: Tensor,
    input_fields: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    pol: str = "Ez",
    norm: bool = True,
    max_fields=32,
    if_colorbar = False,
    field_range = 0.25,
    error_range = 0.1,
) -> None:

    field_val = pred_fields.data.transpose(-1, -2).cpu().numpy()
    target_field_val = target_fields.data.transpose(-1, -2).cpu().numpy()
    input_field_val = input_fields.data.transpose(-1, -2).cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.transpose(-1, -2).cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(eps_r)

    # vmax = field_val.max()
    vmin = 0.0
    b = min(max(field_val.shape[0], input_field_val.shape[0]), max_fields)
    target_start = max(-field_val.shape[0], -max_fields)
    input_start = max(-input_field_val.shape[0], -max_fields)

    field_val = field_val[target_start:]
    target_field_val = target_field_val[target_start:]
    err_field_val = err_field_val[target_start:]
    input_field_val = input_field_val[input_start:]
    fig, axes = plt.subplots(4, b, constrained_layout=True, figsize=(3 * b, 8.1))
    if b == 1:
        axes = axes[..., np.newaxis]
    # cmap = "magma"
    cmap = "RdBu_r"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    zeros = torch.zeros_like(target_fields[0]).cpu().numpy()
    for i in range(b):
        vmax = np.max(np.abs(target_field_val[i])) if i < target_field_val.shape[0] else 0.1
        h0 = axes[0, i].imshow(input_field_val[i] if i < input_field_val.shape[0] else zeros, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
        if norm:
            h1 = axes[2, i].imshow(normalize(field_val[i]) if i < field_val.shape[0] else zeros, cmap=cmap, origin="lower")
            h2 = axes[1, i].imshow(normalize(target_field_val[i]) if i < target_field_val.shape[0] else zeros, cmap=cmap, origin="lower")
        else:
            h1 = axes[2, i].imshow(field_val[i] if i < field_val.shape[0] else zeros, cmap=cmap, vmin=-field_range, vmax=field_range, origin="lower")
            h2 = axes[1, i].imshow(target_field_val[i] if i < target_field_val.shape[0] else zeros, cmap=cmap, vmin=-field_range, vmax=field_range, origin="lower")
        h3 = axes[3, i].imshow(err_field_val[i] if i < err_field_val.shape[0] else zeros, cmap=cmap, vmin=-error_range, vmax=error_range, origin="lower")
        if if_colorbar:
            for j in range(4):
                divider = make_axes_locatable(axes[j, i])
                cax = divider.append_axes("right", size="5%", pad=0.05)

                fig.colorbar([h0, h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)

    # Do black and white so we can see on both magma and RdBu
    for ax in axes.flatten():
        ax.contour(outline_val[0], levels=2, linewidths=1.0, colors="w")
        ax.contour(outline_val[0], levels=2, linewidths=0.5, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    # set_ms()
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_compare(
    epsilon: Tensor,
    input_fields: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    pol: str = "Ez",
    norm: bool = True,
    max_fields=32,
) -> None:

    field_val = pred_fields.data.transpose(-1, -2).cpu().numpy()
    target_field_val = target_fields.data.transpose(-1, -2).cpu().numpy()
    input_field_val = input_fields.data.transpose(-1, -2).cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.transpose(-1, -2).cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(eps_r)

    # vmax = field_val.max()
    vmin = 0.0
    b = min(max(field_val.shape[0], input_field_val.shape[0]), max_fields)
    target_start = max(-field_val.shape[0], -max_fields)
    input_start = max(-input_field_val.shape[0], -max_fields)

    field_val = field_val[target_start:]
    target_field_val = target_field_val[target_start:]
    err_field_val = err_field_val[target_start:]
    input_field_val = input_field_val[input_start:]
    fig, axes = plt.subplots(4, b, constrained_layout=True, figsize=(3 * b, 8.1))
    if b == 1:
        axes = axes[..., np.newaxis]
    # cmap = "magma"
    cmap = "RdBu_r"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    zeros = torch.zeros_like(target_fields[0]).cpu().numpy()
    for i in range(b):
        vmax = np.max(np.abs(target_field_val[i])) if i < target_field_val.shape[0] else 0.1
        h0 = axes[0, i].imshow(input_field_val[i] if i < input_field_val.shape[0] else zeros, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
        if norm:
            h1 = axes[2, i].imshow(normalize(field_val[i]) if i < field_val.shape[0] else zeros, cmap=cmap, origin="lower")
            h2 = axes[1, i].imshow(normalize(target_field_val[i]) if i < target_field_val.shape[0] else zeros, cmap=cmap, origin="lower")
        else:
            h1 = axes[2, i].imshow(field_val[i] if i < field_val.shape[0] else zeros, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
            h2 = axes[1, i].imshow(target_field_val[i] if i < target_field_val.shape[0] else zeros, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
        h3 = axes[3, i].imshow(err_field_val[i] if i < err_field_val.shape[0] else zeros, cmap=cmap, vmin=-vmax, vmax=vmax, origin="lower")
        for j in range(4):
            divider = make_axes_locatable(axes[j, i])
            cax = divider.append_axes("right", size="5%", pad=0.05)

            fig.colorbar([h0, h1, h2, h3][j], label=pol, ax=axes[j, i], cax=cax)
 
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])

    # Do black and white so we can see on both magma and RdBu
    for ax in axes.flatten():
        ax.contour(outline_val[0], levels=2, linewidths=1.0, colors="w")
        ax.contour(outline_val[0], levels=2, linewidths=0.5, colors="k")
        ax.set_xticks([])
        ax.set_yticks([])
    # set_ms()
    plt.savefig(filepath, dpi=150)
    plt.close()

def plot_dynamics(
    wavelength: Tensor,
    grid_step: Tensor,
    epsilon: Tensor,
    pred_fields: Tensor,
    target_fields: Tensor,
    filepath: str,
    eps_list,
    eps_text_loc_list,
    region_list,
    box_id,
    ref_eps,
    norm: bool = True,
    wl_text_pos=None,
    time=None,
    fps=None,
) -> None:
    if epsilon.is_complex():
        epsilon = epsilon.real
        ref_eps = ref_eps.real
    # simulation = Simulation(omega, eps_r=epsilon.data.cpu().numpy(), dl=grid_step, NPML=[1,1], pol='Hz')
    field_val = pred_fields.data.cpu().numpy()
    target_field_val = target_fields.data.cpu().numpy()
    # eps_r = simulation.eps_r
    eps_r = epsilon.data.cpu().numpy()
    err_field_val = field_val - target_field_val
    # field_val = np.abs(field_val)
    field_val = field_val.real
    # target_field_val = np.abs(target_field_val)
    target_field_val = target_field_val.real
    err_field_val = np.abs(err_field_val)
    outline_val = np.abs(ref_eps.data.cpu().numpy())

    # vmax = field_val.max()
    vmin = 0.0
    b = field_val.shape[0]
    fig, ax = plt.subplots(1, b, constrained_layout=True, figsize=(5.1, 1.15))
    # fig, ax = plt.subplots(1, b, constrained_layout=True, figsize=(5.2, 1))
    cmap = "magma"
    # print(field_val.shape, target_field_val.shape, outline_val.shape)
    i = 0
    vmax = np.max(target_field_val[i])
    if norm:
        h1 = ax.imshow(normalize(field_val[i]), cmap=cmap, origin="lower")
    else:
        h1 = ax.imshow(field_val[i], cmap=cmap, vmin=0, vmax=vmax, origin="lower")

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)

    fig.colorbar(h1, label="Mag.", ax=ax, cax=cax)
    for i, (eps, eps_pos, box) in enumerate(zip(eps_list, eps_text_loc_list, region_list)):
        xl, xh, yl, yh = box
        if i == box_id:
            color = "yellow"
        else:
            color = "white"
        ax.annotate(r"$\epsilon_r$"+f" = {eps:.3f}", xy=eps_pos, xytext=eps_pos, color=color)
        ax.plot((xl, xh), (yl, yl), linewidth=0.5, color=color)
        ax.plot((xl, xh), (yh, yh), linewidth=0.5, color=color)
        ax.plot((xl, xl), (yl, yh), linewidth=0.5, color=color)
        ax.plot((xh, xh), (yl, yh), linewidth=0.5, color=color)
    if wl_text_pos is not None:
        if box_id == len(region_list):
            color = "yellow"
        else:
            color = "white"
        ax.annotate(r"$\lambda$" + f" = {wavelength.item():.3f}", xy=wl_text_pos, xytext=wl_text_pos, color=color)
        # fig.colorbar(h2, label=pol, ax=axes[1,i])
        # fig.colorbar(h3, label=pol, ax=axes[2,i])
    # ax.annotate(f"Runtime = {time:.3f} s", xy=(field_val.shape[-1]//2-30, field_val.shape[-2]), xytext=(field_val.shape[-1]//2-40, field_val.shape[-2]+1), color="black", annotation_clip=False)
    if time is not None:
        ax.annotate(f"Runtime = {time:.3f} s, FPS = {fps:.1f}", xy=(field_val.shape[-1]//2-110, field_val.shape[-2]+3), xytext=(field_val.shape[-1]//2-110, field_val.shape[-2]+3), color="black", annotation_clip=False)

    if box_id == len(region_list) + 1:
        color = "blue"
    else:
        color = "black"
    ax.annotate(r"$l_z$" + f" = {grid_step[..., 0].item()*field_val.shape[-1]:.2f} " + r"$\mu m$", xy=(field_val.shape[-1]//2-30, -15), xytext=(field_val.shape[-1]//2-30, -15), color=color, annotation_clip=False)
    ax.annotate(r"$l_x$" + f" = {grid_step[..., 1].item()*field_val.shape[-2]:.2f} " + r"$\mu m$", xy=(-18, field_val.shape[-2]//2-44), xytext=(-18, field_val.shape[-2]//2-44), color=color, annotation_clip=False, rotation=90)


    # Do black and white so we can see on both magma and RdBu

    ax.contour(outline_val[0], levels=1, linewidths=1.0, colors="w")
    ax.contour(outline_val[0], levels=1, linewidths=0.5, colors="k")
    ax.set_xticks([])
    ax.set_yticks([])
    # plt.tight_layout()
    plt.savefig(filepath, dpi=200)
    plt.close()
