#!/usr/bin/env python
# coding=UTF-8
import argparse
import datetime
import os
from typing import Callable, Dict, Iterable, List, Tuple

import mlflow
import numpy as np
import torch
import torch.cuda.amp as amp
import torch.fft
import torch.nn as nn
import torch.nn.functional as F
from pyutils.config import configs
from pyutils.general import AverageMeter
from pyutils.general import logger as lg
from pyutils.torch_train import (
    BestKModelSaver,
    count_parameters,
    get_learning_rate,
    load_model,
    set_torch_deterministic,
)
from pyutils.typing import Criterion, DataLoader, Optimizer, Scheduler

import wandb
from core import builder
from core.utils import DeterministicCtx, plot_compare


def train(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: Optimizer,
    scheduler: Scheduler,
    epoch: int,
    criterion: Criterion,
    aux_criterions: Dict,
    mixup_fn: Callable = None,
    device: torch.device = torch.device("cuda:0"),
    plot: bool = False,
    grad_scaler=None,
    maskloss: bool = False,
) -> None:
    model.train()
    configs.run.multi_train_schedule = configs.run.multi_train_schedule[
        : model.num_iters - 1
    ]
    step = epoch * len(train_loader)
    if epoch == 1:
        model.set_max_trainable_iter(1)
        configs.run.multi_train_milestone = 0
    mse_meter = AverageMeter("mse")
    aux_meters = {name: AverageMeter(name) for name in aux_criterions}
    aux_output_weight = getattr(configs.criterion, "aux_output_weight", 0)
    accum_iter = getattr(configs.run, "grad_accum_step", 1)

    data_counter = 0
    total_data = len(train_loader.dataset)
    if epoch in configs.run.multi_train_schedule:
        milestone = configs.run.multi_train_schedule.index(epoch)
        configs.run.multi_train_milestone += 1
        model.set_max_trainable_iter(milestone + 2)
        model.load_parameters(milestone, milestone + 1, optimizer)
        lg.info(f"trigger multi_train_schedule at epoch {epoch}")
        lg.info(f"Load parameters from {milestone} and {milestone+1}")
    for batch_idx, (raw_data, raw_target) in enumerate(train_loader):
        for key, d in raw_data.items():
            raw_data[key] = d.to(device, non_blocking=True)
        for key, t in raw_target.items():
            raw_target[key] = t.to(device, non_blocking=True)

        data = torch.cat([raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1)
        target = raw_target["Ez"]
        target = target[
            :, : model.out_channels * (configs.run.multi_train_milestone + 1)
        ]
        data_counter += data.shape[0]
        target = target.to(device, non_blocking=True)
        if mixup_fn is not None:
            data, target = mixup_fn(data, target)

        with amp.autocast(enabled=grad_scaler._enabled):
            output, normalization_factor = model(
                data,
                src_mask=raw_data["src_mask"],
                padding_mask=raw_data["padding_mask"],
            )
            if type(output) == tuple:
                output, aux_output = output
            else:
                aux_output = None
            if maskloss:
                if model.num_iters == 1:
                    regression_loss = criterion(
                        output, target / normalization_factor, raw_data["mseWeight"]
                    )
                else:
                    regression_loss = criterion(
                        output,
                        target / normalization_factor,
                        raw_data["mseWeight"],
                        model.num_iters,
                    )
            else:
                regression_loss = criterion(
                    output, target / normalization_factor, raw_data["mseWeight"]
                )
            mse_meter.update(regression_loss.item())
            loss = regression_loss
            for name, config in aux_criterions.items():
                aux_criterion, weight = config
                if name == "curl_loss":
                    fields = torch.cat([target[:, 0:1]], output, target[:, 2:3], dim=1)
                    aux_loss = weight * aux_criterion(fields, data[:, 0:1])
                elif name == "tv_loss":
                    aux_loss = weight * aux_criterion(output, target)
                elif name == "poynting_loss":
                    aux_loss = weight * aux_criterion(output, target)
                elif name == "rtv_loss":
                    aux_loss = weight * aux_criterion(output, target)
                loss = loss + aux_loss
                aux_meters[name].update(aux_loss.item())
            # TODO aux output loss
            if aux_output is not None and aux_output_weight > 0:
                aux_output_loss = aux_output_weight * F.mse_loss(
                    aux_output, target.abs()
                )  # field magnitude learning
                loss = loss + aux_output_loss
            else:
                aux_output_loss = None

            loss = loss / accum_iter

        grad_scaler.scale(loss).backward()

        if ((batch_idx + 1) % accum_iter == 0) or (batch_idx + 1 == len(train_loader)):
            grad_scaler.unscale_(optimizer)
            grad_scaler.step(optimizer)
            grad_scaler.update()
            optimizer.zero_grad()

        step += 1

        if batch_idx % int(configs.run.log_interval) == 0:
            log = "Train Epoch: {} [{:7d}/{:7d} ({:3.0f}%)] Loss: {:.4e} Regression Loss: {:.4e}".format(
                epoch,
                data_counter,
                total_data,
                100.0 * data_counter / total_data,
                loss.data.item(),
                regression_loss.data.item(),
            )
            for name, aux_meter in aux_meters.items():
                log += f" {name}: {aux_meter.val:.4e}"
            if aux_output_loss is not None:
                log += f" aux_output_loss: {aux_output_loss.item()}"
            lg.info(log)

            mlflow.log_metrics({"train_loss": loss.item()}, step=step)
            wandb.log(
                {
                    "train_running_loss": loss.item(),
                    "global_step": step,
                },
            )
    scheduler.step()
    avg_regression_loss = mse_meter.avg
    lg.info(f"Train Regression Loss: {avg_regression_loss:.4e}")
    mlflow.log_metrics(
        {"train_regression": avg_regression_loss, "lr": get_learning_rate(optimizer)},
        step=epoch,
    )
    wandb.log(
        {
            "train_loss": avg_regression_loss,
            "epoch": epoch,
            "lr": get_learning_rate(optimizer),
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_train.png")
        plot_compare(
            epsilon=data[0, 0:1],
            input_fields=data[0, 1 : -target.shape[1]],
            pred_fields=output[0] * normalization_factor[0],
            target_fields=target[0],
            filepath=filepath,
            pol="Ez",
            norm=False,
        )


def validate(
    model: nn.Module,
    validation_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = True,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    with torch.no_grad(), DeterministicCtx(42):
        for i, (raw_data, raw_target) in enumerate(validation_loader):
            for key, d in raw_data.items():
                raw_data[key] = d.to(device, non_blocking=True)
            for key, t in raw_target.items():
                raw_target[key] = t.to(device, non_blocking=True)

            data = torch.cat(
                [raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1
            )
            target = raw_target["Ez"]
            target = target[
                :, : model.out_channels * (configs.run.multi_train_milestone + 1)
            ]
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target)

            with amp.autocast(enabled=False):
                output, normalization_factor = model(
                    data,
                    src_mask=raw_data["src_mask"],
                    padding_mask=raw_data["padding_mask"],
                )
                if type(output) == tuple:
                    output, aux_output = output
                else:
                    aux_output = None
                val_loss = criterion(
                    output, target / normalization_factor, raw_data["mseWeight"]
                )
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nValidation set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"val_loss": mse_meter.avg}, step=epoch)
    wandb.log(
        {
            "val_loss": mse_meter.avg,
            "epoch": epoch,
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_valid.png")
        plot_compare(
            epsilon=data[1, 0:1] if data.shape[0] > 1 else data[0, 0:1],
            input_fields=(
                data[1, 1 : -target.shape[1]]
                if data.shape[0] > 1
                else data[0, 1 : -target.shape[1]]
            ),
            pred_fields=(
                output[1] * normalization_factor[1]
                if data.shape[0] > 1
                else output[0] * normalization_factor[0]
            ),
            target_fields=target[1] if data.shape[0] > 1 else target[0],
            filepath=filepath,
            pol="Ez",
            norm=False,
        )


def test(
    model: nn.Module,
    test_loader: DataLoader,
    epoch: int,
    criterion: Criterion,
    loss_vector: Iterable,
    accuracy_vector: Iterable,
    device: torch.device,
    mixup_fn: Callable = None,
    plot: bool = False,
    test_loaded: bool = False,
) -> None:
    model.eval()
    val_loss = 0
    mse_meter = AverageMeter("mse")
    if test_loaded:
        model.set_max_trainable_iter(model.num_iters)
        configs.run.multi_train_milestone = 1
    with torch.no_grad(), DeterministicCtx(42):
        for i, (raw_data, raw_target) in enumerate(test_loader):
            for key, d in raw_data.items():
                raw_data[key] = d.to(device, non_blocking=True)
            for key, t in raw_target.items():
                raw_target[key] = t.to(device, non_blocking=True)

            data = torch.cat(
                [raw_data["eps"], raw_data["Ez"], raw_data["source"]], dim=1
            )
            target = raw_target["Ez"]
            target = target[
                :, : model.out_channels * (configs.run.multi_train_milestone + 1)
            ]
            target = target.to(device, non_blocking=True)
            if mixup_fn is not None:
                data, target = mixup_fn(data, target)

            with amp.autocast(enabled=False):
                output, normalization_factor = model(
                    data,
                    src_mask=raw_data["src_mask"],
                    padding_mask=raw_data["padding_mask"],
                )
                if type(output) == tuple:
                    output, aux_output = output
                else:
                    aux_output = None
                val_loss = criterion(
                    output, target / normalization_factor, raw_data["mseWeight"]
                )
            mse_meter.update(val_loss.item())

    loss_vector.append(mse_meter.avg)

    lg.info("\nTest set: Average loss: {:.4e}\n".format(mse_meter.avg))
    mlflow.log_metrics({"test_loss": mse_meter.avg}, step=epoch)
    wandb.log(
        {
            "test_loss": mse_meter.avg,
            "epoch": epoch,
        },
    )

    if plot and (
        epoch % configs.plot.interval == 0 or epoch == configs.run.n_epochs - 1
    ):
        dir_path = os.path.join(configs.plot.root, configs.plot.dir_name)
        os.makedirs(dir_path, exist_ok=True)
        filepath = os.path.join(dir_path, f"epoch_{epoch}_test.png")
        plot_compare(
            epsilon=data[0, 0:1],
            input_fields=data[0, 1 : -target.shape[1]],
            pred_fields=output[0] * normalization_factor[0],
            target_fields=target[0],
            filepath=filepath,
            pol="Ez",
            norm=False,
        )
    if test_loaded:
        model.set_max_trainable_iter(1)
        configs.run.multi_train_milestone = 0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("config", metavar="FILE", help="config file")
    args, opts = parser.parse_known_args()

    configs.load(args.config, recursive=True)
    configs.update(opts)
    if torch.cuda.is_available() and int(configs.run.use_cuda):
        torch.cuda.set_device(configs.run.gpu_id)
        device = torch.device("cuda:" + str(configs.run.gpu_id))
        torch.backends.cudnn.benchmark = True
    else:
        device = torch.device("cpu")
        torch.backends.cudnn.benchmark = False

    if int(configs.run.deterministic) == True:
        set_torch_deterministic(int(configs.run.random_state))

    if "backbone_cfg" in configs.model.keys():
        if (
            configs.model.backbone_cfg.conv_cfg.type == "Conv2d"
            or configs.model.backbone_cfg.conv_cfg.type == "LargeKernelConv2d"
        ):
            if "r" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["r"]
            if "is_causal" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["is_causal"]
            if "mask_shape" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["mask_shape"]
            if "enable_padding" in configs.model.backbone_cfg.conv_cfg.keys():
                del configs.model.backbone_cfg.conv_cfg["enable_padding"]

    train_loader, validation_loader, test_loader = builder.make_dataloader()
    model = builder.make_model(
        device,
        int(configs.run.random_state) if int(configs.run.deterministic) else None,
    )
    lg.info(model)

    optimizer = builder.make_optimizer(
        [p for p in model.parameters() if p.requires_grad],
        name=configs.optimizer.name,
        configs=configs.optimizer,
    )
    scheduler = builder.make_scheduler(optimizer)
    criterion = builder.make_criterion(configs.criterion.name, configs.criterion).to(
        device
    )
    test_criterion = builder.make_criterion(
        configs.test_criterion.name, configs.test_criterion
    ).to(device)
    aux_criterions = {
        name: [builder.make_criterion(name, cfg=config), float(config.weight)]
        for name, config in configs.aux_criterion.items()
        if float(config.weight) > 0
    }
    print(aux_criterions)
    mixup_config = configs.dataset.augment
    # mixup_fn = MixupAll(**mixup_config)
    # test_mixup_fn = MixupAll(**configs.dataset.test_augment)
    # no mixup in this project
    mixup_fn = None
    test_mixup_fn = None
    saver = BestKModelSaver(
        k=int(configs.checkpoint.save_best_model_k),
        descend=False,
        truncate=10,
        metric_name="err",
        format="{:.4f}",
    )

    grad_scaler = amp.GradScaler(enabled=getattr(configs.run, "fp16", False))
    lg.info(f"Number of parameters: {count_parameters(model)}")

    model_name = f"{configs.model.name}"
    checkpoint = f"./checkpoint/{configs.checkpoint.checkpoint_dir}/{model_name}_{configs.checkpoint.model_comment}.pt"

    lg.info(f"Current checkpoint: {checkpoint}")

    wandb.login()
    tag = wandb.util.generate_id()
    group = f"{datetime.date.today()}"
    name = f"{configs.run.wandb.name}-{datetime.datetime.now().hour:02d}{datetime.datetime.now().minute:02d}{datetime.datetime.now().second:02d}-{tag}"
    configs.run.pid = os.getpid()
    run = wandb.init(
        project=configs.run.wandb.project,
        group=group,
        name=name,
        id=tag,
        config=configs,
    )

    lossv, accv = [0], [0]
    epoch = 0
    try:
        lg.info(
            f"Experiment {name} starts. Group: {group}, Run ID: ({run.id}). PID: ({os.getpid()}). PPID: ({os.getppid()}). Host: ({os.uname()[1]})"
        )
        lg.info(configs)
        if (
            int(configs.checkpoint.resume)
            and len(configs.checkpoint.restore_checkpoint) > 0
        ):
            load_model(
                model,
                configs.checkpoint.restore_checkpoint,
                ignore_size_mismatch=int(configs.checkpoint.no_linear),
            )

            lg.info("Validate resumed model...")
            test(
                model,
                test_loader,
                0,
                test_criterion,
                [],
                [],
                device,
                mixup_fn=test_mixup_fn,
                plot=configs.plot.test,
                test_loaded=True,
            )

        for epoch in range(1, int(configs.run.n_epochs) + 1):
            train(
                model,
                train_loader,
                optimizer,
                scheduler,
                epoch,
                criterion,
                aux_criterions,
                mixup_fn,
                device,
                plot=configs.plot.train,
                grad_scaler=grad_scaler,
                maskloss=("mask" in configs.criterion.name),
            )

            if validation_loader is not None:
                validate(
                    model,
                    validation_loader,
                    epoch,
                    test_criterion,
                    lossv,
                    accv,
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.valid,
                )
            if epoch > int(configs.run.n_epochs) - 21:
                test(
                    model,
                    test_loader,
                    epoch,
                    test_criterion,
                    [],
                    [],
                    device,
                    mixup_fn=test_mixup_fn,
                    plot=configs.plot.test,
                )
                saver.save_model(
                    model,
                    lossv[-1],
                    epoch=epoch,
                    path=checkpoint,
                    save_model=False,
                    print_msg=True,
                )
        wandb.finish()
    except KeyboardInterrupt:
        lg.warning("Ctrl-C Stopped")


if __name__ == "__main__":
    main()
