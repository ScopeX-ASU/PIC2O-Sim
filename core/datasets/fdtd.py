"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2022-01-24 23:27:31
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2022-02-01 16:22:32
"""

"""
Description:
Author: Jiaqi Gu (jqgu@utexas.edu)
Date: 2021-04-04 13:38:43
LastEditors: Jiaqi Gu (jqgu@utexas.edu)
LastEditTime: 2021-04-04 15:16:55
"""
import glob
import os
import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import h5py

# from core.utils import plot_compare
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from pyutils.compute import add_gaussian_noise, gen_gaussian_noise
from pyutils.general import print_stat

# from pyutils.torch_train import DeterministicCtx
from torch import Tensor
from torchpack.datasets.dataset import Dataset
from torchvision import transforms
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import download_url
from torchvision.transforms import InterpolationMode, functional

resize_modes = {
    "bilinear": InterpolationMode.BILINEAR,
    "bicubic": InterpolationMode.BICUBIC,
    "nearest": InterpolationMode.NEAREST,
}

__all__ = ["FDTD", "FDTDDataset"]

# random.seed(42)

# def calculate_time_step(courant_const: float = 0.5, resolution: int = 20) -> float:
#     c = 3e8
#     time_step = 6 * (courant_const * (1 / resolution) / c) * 1e9
#     # 6 is because we get result every 6 delta_t in meep (0.3 unit of time), 1e9 is to convert to fs
#     return time_step


def resize_and_pad_to_square(
    image: Tensor, image_size: int = 512, resolution: float = 20.0, is_eps: bool = False, is_mseWeight: bool = False
):
    """Resizes an image to a square with constant padding, maintaining aspect ratio.

    Args:
        image: A PyTorch tensor of shape (C, H, W) representing the image.
        image_size: The desired size of the square image.

    Returns:
        A PyTorch tensor of shape (C, image_size, image_size) representing the resized and padded image.
    """

    # Determine the longer dimension of the original image
    max_dim = max(image.size(-1), image.size(-2))

    # Calculate the scaling ratio
    scaling_factor = image_size / max_dim

    # Resize the image based on the scaling factor
    new_size = (
        int(round(image.size(-2) * scaling_factor)),
        int(round(image.size(-1) * scaling_factor)),
    )
    # print(image.size(), new_size)
    image = image.unsqueeze(0)
    resized_image = torch.nn.functional.interpolate(
        image, size=new_size, mode="bilinear"
    )
    # print(resized_image.shape)
    # Calculate the padding needed to make the image square
    padding = (image_size - resized_image.size(-2), image_size - resized_image.size(-1))
    lower, upper = padding[0] // 2, padding[0] - padding[0] // 2
    left, right = padding[1] // 2, padding[1] - padding[1] // 2
    # print(padding)

    # Pad the image with constant padding
    if not is_mseWeight:
        padded_image = torch.nn.functional.pad(
            resized_image,
            (left, right, lower, upper),
            value=image[0, 0, 1, 1],
            mode="constant",
        )[0]
    else:
        padded_image = torch.nn.functional.pad(
            resized_image,
            (left, right, lower, upper),
            value=0,
            mode="constant",
        )[0]
    grid_step = 1 / resolution / scaling_factor
    # assert padded_image.shape[1] == padded_image.shape[2] == image_size, f"{padded_image.shape[1]} != {padded_image.shape[2]} != {image_size}"

    if is_eps:
        padded_image = padded_image / scaling_factor

    return padded_image, torch.tensor([grid_step], device=padded_image.device)


# here is the data augmentation functions
class FlipEField(object):
    def __call__(self, sample):
        data = sample["data"]
        target = sample["target"]
        marker = 0
        if random.random() < 0.1:
            data_input_field = -1 * (data["Ez"].clone())
            data_source = -1 * (data["source"].clone())
            data_eps = data["eps"].clone()
            data_grid_step = data["grid_step"].clone()
            data_mseWeight = data["mseWeight"].clone()
            data_src_mask = data["src_mask"].clone()

            target_Ez = -1 * target["Ez"].clone()
            marker = 1
        if marker == 1:
            return_sample = {
                "data": {
                    "Ez": data_input_field,
                    "source": data_source,
                    "eps": data_eps,
                    "grid_step": data_grid_step,
                    "mseWeight": data_mseWeight,
                    "src_mask": data_src_mask,
                },
                "target": {
                    "Ez": target_Ez,
                },
            }
            return return_sample
        else:
            return sample


class ScalePeak(object):
    def __call__(self, sample):
        data = sample["data"]
        target = sample["target"]
        marker = 0
        if random.random() < 0.1:
            scale_factor = random.uniform(-5, 5)
            data_input_field = scale_factor * (data["Ez"].clone())
            data_source = scale_factor * (data["source"].clone())
            data_eps = data["eps"].clone()
            data_mseWeight = data["mseWeight"].clone()
            data_src_mask = data["src_mask"].clone()
            data_grid_step = data["grid_step"].clone()

            target_Ez = scale_factor * (target["Ez"].clone())
            marker = 1
        if marker == 1:
            return_sample = {
                "data": {
                    "Ez": data_input_field,
                    "source": data_source,
                    "eps": data_eps,
                    "mseWeight": data_mseWeight,
                    "src_mask": data_src_mask,
                    "grid_step": data_grid_step,
                },
                "target": {
                    "Ez": target_Ez,
                },
            }
            return return_sample
        else:
            return sample


class RotateVideo(object):
    def __call__(self, sample):
        data = sample["data"]
        target = sample["target"]
        marker = 0
        if random.random() < 0.1:
            angle = random.choice([90, 180, 270])
            data_input_field = functional.rotate(data["Ez"], angle)
            data_source = functional.rotate(data["source"], angle)
            data_eps = functional.rotate(data["eps"], angle)
            data_mseWeight = functional.rotate(data["mseWeight"], angle)
            data_grid_step = data["grid_step"]
            data_src_mask = functional.rotate(data["src_mask"], angle)

            target_Ez = functional.rotate(target["Ez"], angle)

            marker = 1
        if marker == 1:
            return_sample = {
                "data": {
                    "Ez": data_input_field,
                    "source": data_source,
                    "eps": data_eps,
                    "mseWeight": data_mseWeight,
                    "grid_step": data_grid_step,
                    "src_mask": data_src_mask,
                },
                "target": {
                    "Ez": target_Ez,
                },
            }
            return return_sample
        else:
            return sample


class FlipHorizontal(object):
    def __call__(self, sample):
        data = sample["data"]
        target = sample["target"]
        marker = 0
        if random.random() < 0.1:
            data_input_field = torch.flip(data["Ez"], [-1])
            data_source = torch.flip(data["source"], [-1])
            data_eps = torch.flip(data["eps"], [-1])
            data_mseWeight = torch.flip(data["mseWeight"], [-1])
            data_grid_step = data["grid_step"]
            data_src_mask = torch.flip(data["src_mask"], [-1])

            target_Ez = torch.flip(target["Ez"], [-1])
            marker = 1
        if marker == 1:
            return_sample = {
                "data": {
                    "Ez": data_input_field,
                    "source": data_source,
                    "eps": data_eps,
                    "mseWeight": data_mseWeight,
                    "grid_step": data_grid_step,
                    "src_mask": data_src_mask,
                },
                "target": {
                    "Ez": target_Ez,
                },
            }
            return return_sample
        else:
            return sample


class FlipVertical(object):
    def __call__(self, sample):
        data = sample["data"]
        target = sample["target"]
        marker = 0
        if random.random() < 0.1:
            data_input_field = torch.flip(data["Ez"], [-2])
            data_source = torch.flip(data["source"], [-2])
            data_eps = torch.flip(data["eps"], [-2])
            data_mseWeight = torch.flip(data["mseWeight"], [-2])
            data_grid_step = data["grid_step"]
            data_src_mask = torch.flip(data["src_mask"], [-2])

            target_Ez = torch.flip(target["Ez"], [-2])
            marker = 1
        if marker == 1:
            return_sample = {
                "data": {
                    "Ez": data_input_field,
                    "source": data_source,
                    "eps": data_eps,
                    "mseWeight": data_mseWeight,
                    "grid_step": data_grid_step,
                    "src_mask": data_src_mask,
                },
                "target": {
                    "Ez": target_Ez,
                },
            }
            return return_sample
        else:
            return sample


class FDTD(VisionDataset):
    url = None
    train_filename = "training"
    test_filename = "test"
    device_pool_filename = "device_pool"
    meta_filename = "meta"
    folder = "fdtd"

    def __init__(
        self,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        train_ratio: float = 0.7,
        device_list: List[str] = ["mmi_3x3_L_random"],
        processed_dir: str = "processed",
        download: bool = False,
        mixup_cfg=None,
        in_frames: int = 2,
        out_frames: int = 100,
        offset_frames: int = 2,
        img_size: int = 512,
        batch_strategy: str = "resize_and_padding_to_square",
    ) -> None:
        self.processed_dir = processed_dir
        root = os.path.join(os.path.expanduser(root), self.folder)
        is_totensor = (
            isinstance(transform, transforms.Compose)
            and len(transform.transforms) == 1
            and isinstance(transform.transforms[0], transforms.ToTensor)
        )
        if is_totensor:
            self.transform = transforms.Compose([transforms.ToTensor()])
            self.ToTensorMarker = True
        else:
            self.transform = transform
            self.ToTensorMarker = False
        super().__init__(root, transform=transform, target_transform=target_transform)

        self.train = train  # training set or test set
        self.train_ratio = train_ratio
        self.device_list = sorted(device_list)
        self.device_pool = dict()  # this is for the mixup
        self.filenames = (
            []
        )  # filenames will be a list of list, each list contains all the ports for a device
        self.mixup_cfg = mixup_cfg
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.offset_frames = offset_frames

        self.train_filename = self.train_filename
        self.test_filename = self.test_filename
        self.device_pool_filename = self.device_pool_filename
        self.meta_filename = self.meta_filename

        self.img_size = img_size

        self.batch_strategy = batch_strategy

        if download:
            self.download()

        if not self._check_integrity():
            raise RuntimeError(
                "Dataset not found or corrupted."
                + " You can use download=True to download it"
            )

        self.process_raw_data()
        self.data = self.load(train=train)

    def process_raw_data(self) -> None:
        processed_dir = os.path.join(self.root, self.processed_dir)
        # no matter the preprocessed file exists or not, we will always process the data, won't take too much time
        processed_training_file = os.path.join(
            processed_dir, f"{self.train_filename}.yml"
        )
        processed_test_file = os.path.join(processed_dir, f"{self.test_filename}.yml")
        processed_device_pool_dile = os.path.join(
            processed_dir, f"{self.device_pool_filename}.yml"
        )
        processed_meta_file = os.path.join(processed_dir, f"{self.meta_filename}.yml") # record the max hight and width of the data
        if (
            os.path.exists(processed_training_file)
            and os.path.exists(processed_test_file)
            and os.path.exists(processed_device_pool_dile)
            and os.path.exists(processed_meta_file)
        ):
            print("Data already processed")
            return

        filenames = (
            self._load_dataset()
        )  # only load filenames for different devices with different ports
        (
            filenames_train,
            filenames_test,
        ) = self._split_dataset(
            filenames
        )  # split device files to make sure no overlapping device_id between train and test
        data_train, data_test = self._preprocess_dataset(
            filenames_train, filenames_test
        )
        # data train and data test looks like this [[device_files, start_index], [device_files, start_index], [device_files, start_index], ...]
        for device_files, start in data_train:
            device_name = (
                device_files[0].rstrip(".h5").split("-")[0]
                + "-"
                + device_files[0].rstrip(".h5").split("-")[1]
            )  # looks like mmi_3x3_L_random-0046
            if device_name not in self.device_pool.keys():
                self.device_pool[device_name] = []
                self.device_pool[device_name].append([device_files, start])
            else:
                self.device_pool[device_name].append([device_files, start])
        # looks like this [['mmi_3x3_L_random-0029-p0.h5', 'mmi_3x3_L_random-0029-p1.h5', 'mmi_3x3_L_random-0029-p2.h5'], 100], ...
        # now the device_pool is a dictionary, key is the device name, value is a list of [device_files, start_index]
        # every list in the dictionary corresponds to a single device with different ports and starting index
        meta_dict = dict()
        meta_dict["max_height"] = 0
        meta_dict["max_width"] = 0
        for device_files, _ in data_train:
            with open(os.path.join(self.root, "raw", device_files[0]).replace(".h5", ".yml"), "r") as f:
                yaml_file = yaml.safe_load(f)
                device_type = yaml_file["device"]["type"]
            with h5py.File(os.path.join(self.root, "raw", device_files[0]), "r") as f:
                height = f["Ez"].shape[-2]
                width = f["Ez"].shape[-1]
                if 'mmi' or 'metaline' in device_type.lower(): # so that they have the resolution of 15
                    height = height*0.75
                    width = width*0.75
                meta_dict["max_height"] = max(meta_dict["max_height"], height)
                meta_dict["max_width"] = max(meta_dict["max_width"], width)
        # the max height and width of the training data is recorded in the meta_dict
        self._save_dataset(
            data_train,
            data_test,
            meta_dict,
            self.device_pool,
            processed_dir,
            self.train_filename,
            self.test_filename,
            self.device_pool_filename,
            self.meta_filename
        )

    def _load_dataset(self) -> List:
        ## do not load actual data here, too slow. Just load the filenames
        for device in self.device_list:
            all_samples = [
                os.path.basename(i)
                for i in glob.glob(os.path.join(self.root, "raw", f"{device}-*.h5"))
            ]
            selected_samples = []
            for sample in all_samples:
                device_id = sample.rstrip(".h5").split("-")[1]
                device_id = device_id.lstrip("0")
                device_id = int(device_id) if device_id != "" else 0
                if self.device_list[0].startswith("metaline"):
                    if device_id < 40:
                        selected_samples.append(sample)
                else:
                    if device_id < 25:
                        selected_samples.append(sample)
            all_samples = selected_samples
            device_dict = dict()
            for sample in all_samples:
                device_id = sample.rstrip(".h5").split("-")[1]
                if device_id not in device_dict:
                    device_dict[device_id] = []
                device_dict[device_id].append(sample)

            all_ports_files = list(device_dict.values())
            self.filenames.extend(all_ports_files)
        return self.filenames

    def _split_dataset(self, filenames) -> Tuple[List, ...]:
        ## just split device filenames, not the actual tensor
        from sklearn.model_selection import train_test_split
        print("this is the train ratio: ", self.train_ratio, flush=True)
        print("this is the length of the filenames: ", len(filenames), flush=True)
        (
            filenames_train,
            filenames_test,
        ) = train_test_split(
            filenames,
            train_size=int(self.train_ratio * len(filenames)),
            random_state=42,
        )
        print(
            f"training: {len(filenames_train)} device examples, "
            f"test: {len(filenames_test)} device examples"
        )
        return (
            filenames_train,
            filenames_test,
        )

    def _preprocess_dataset(
        self, filenames_train, filenames_test
    ) -> Tuple[Tensor, Tensor]:
        data_train = []
        # sample_frames_train = []
        # sample_frames_test = []
        for device_files in filenames_train:
            if device_files[0].startswith("mmi") or device_files[0].startswith("dc") or device_files[0].startswith("metaline"):
                with h5py.File(
                    os.path.join(self.root, "raw", device_files[0]), "r"
                ) as f:
                    num_frames = f["Ez"].shape[0]  # [833, 650, 243]
                min_index = 8
                # self.offset_frames = random.randint(8, 200)
                # sample_frames_train.append(self.offset_frames + self.out_frames)
                # time_anchor = self.offset_frames - 8 + self.out_frames # this is the number we are going to query the model, 8 means the input frames
                max_index = num_frames - self.out_frames - self.offset_frames + 1 - 7
                start_index = min_index
                while start_index < max_index:
                    if start_index <= 560:
                        data_train.append(
                            [device_files, start_index]
                        )  # each data point include all ports for this device, and a starting slice index
                        data_train.append([device_files, start_index])
                    else:
                        data_train.append([device_files, start_index])
                    # data_train.append([device_files, start_index, time_anchor, self.offset_frames])
                    # start_index += self.in_frames
                    start_index += 16
            else:
                with h5py.File(
                    os.path.join(self.root, "raw", device_files[0]), "r"
                ) as f:
                    num_frames = f["Ez"].shape[0]  # [833, 650, 243]
                min_index = 8
                # self.offset_frames = random.randint(8, 200)
                # sample_frames_train.append(self.offset_frames + self.out_frames)
                # time_anchor = self.offset_frames - 8 + self.out_frames # this is the number we are going to query the model, 8 means the input frames
                max_index = num_frames - self.out_frames - self.offset_frames + 1 - 7
                start_index = min_index
                while start_index < max_index:
                    if start_index <= 560 and start_index >= 100:
                        data_train.append([device_files, start_index])
                        data_train.append([device_files, start_index])
                        data_train.append([device_files, start_index])
                        data_train.append([device_files, start_index])
                        data_train.append([device_files, start_index])
                        data_train.append([device_files, start_index])
                    else:
                        data_train.append(
                            [device_files, start_index]
                        )  # each data point include all ports for this device, and a starting slice index
                    # data_train.append([device_files, start_index, time_anchor, self.offset_frames])
                    # start_index += self.in_frames
                    start_index += 16

        data_test = []
        for device_files in filenames_test:
            with h5py.File(os.path.join(self.root, "raw", device_files[0]), "r") as f:
                num_frames = f["Ez"].shape[0]  # [833, 650, 243]
            min_index = 8
            # self.offset_frames = random.randint(8, num_frames - self.out_frames - 10)
            # sample_frames_test.append(self.offset_frames + self.out_frames)
            # time_anchor = self.offset_frames - 8 + self.out_frames # this is the number we are going to query the model, 8 means the input frames
            max_index = num_frames - self.out_frames - self.offset_frames + 1 - 7
            start_index = min_index
            while start_index < max_index:
                # each data point include all ports for this device, and a starting slice index
                data_test.append([device_files, start_index])
                # data_test.append([device_files, start_index, time_anchor, self.offset_frames])
                # start_index += self.in_frames
                start_index += 16
                # start_index += 2 # every two slices, we have a data point
        # max_sample_frames_train = max(sample_frames_train)
        # max_sample_frames_test = max(sample_frames_test)
        # meta_file = os.path.join(self.root, self.processed_dir, "meta.yml")
        # with open(meta_file, "w") as f:
        #     yaml.dump(max_sample_frames_train, f)
        #     yaml.dump(max_sample_frames_test, f)
        print(
            f"training: {len(data_train)} slice examples, "
            f"test: {len(data_test)} slice examples"
            # f"max sample frames in training set: {max_sample_frames_train}"
            # f"max sample frames in test set: {max_sample_frames_test}"
        )
        return data_train, data_test

    @staticmethod
    def _save_dataset(
        data_train: List,
        data_test: List,
        meta_dict: Dict,
        device_pool: Dict,
        processed_dir: str,
        train_filename: str = "training",
        test_filename: str = "test",
        device_pool_filename: str = "device_pool",
        meta_filename: str = "meta",
    ) -> None:
        os.makedirs(processed_dir, exist_ok=True)
        processed_training_file = os.path.join(processed_dir, f"{train_filename}.yml")
        processed_test_file = os.path.join(processed_dir, f"{test_filename}.yml")
        processed_device_pool_file = os.path.join(
            processed_dir, f"{device_pool_filename}.yml"
        )
        processed_meta_file = os.path.join(processed_dir, f"{meta_filename}.yml")
        with open(processed_training_file, "w") as f:
            yaml.dump(data_train, f)

        with open(processed_test_file, "w") as f:
            yaml.dump(data_test, f)

        with open(processed_device_pool_file, "w") as f:
            yaml.dump(device_pool, f)

        with open(processed_meta_file, "w") as f:
            yaml.dump(meta_dict, f)
        print(f"Processed data filenames + start_index saved")

    def load(self, train: bool = True):
        filename = (
            f"{self.train_filename}.yml" if train else f"{self.test_filename}.yml"
        )
        device_pool_name = f"{self.device_pool_filename}.yml"
        with open(os.path.join(self.root, self.processed_dir, filename), "r") as f:

            data = yaml.safe_load(f)
            if train:
                files = []
                for item in data:
                    files.append(item[0][0])
                files = set(files)
                files = list(files)
                files = sorted(files)
                mmi = []
                mrr = []
                dc = []
                metaline = []
                for item in files:
                    if item.startswith("mmi"):
                        mmi.append(item)
                    elif item.startswith("mrr"):
                        mrr.append(item)
                    elif item.startswith("dc"):
                        dc.append(item)
                    elif item.startswith("metaline"):
                        metaline.append(item)
                    else:
                        raise ValueError("unknown device type")
                mmi = sorted(mmi)
                mrr = sorted(mrr)
                dc = sorted(dc)
                metaline = sorted(metaline)
                denominator = 0
                if len(mmi) > 0:
                    denominator += 1
                elif len(mrr) > 0:
                    denominator += 4
                elif len(dc) > 0:
                    denominator += 1
                elif len(metaline) > 0:
                    denominator += 1
                num_instance_mmi = len(mmi)//denominator + 1
                num_instance_dc = len(dc)//denominator + 1
                num_instance_mrr = len(mrr)//denominator + 1
                num_instance_metaline = len(metaline)//denominator + 1
                files = mmi[:num_instance_mmi] + dc[:num_instance_dc] + mrr[:num_instance_mrr] + metaline[:num_instance_metaline]
                files = set(files)
                data = [item for item in data if item[0][0] in files]

        with open(
            os.path.join(self.root, self.processed_dir, device_pool_name), "r"
        ) as f:
            self.device_pool = yaml.safe_load(f)

        return data

    def download(self) -> None:
        if self._check_integrity():
            print("Files already downloaded and verified")
            return

    def _check_integrity(self) -> bool:
        return all(
            [
                os.path.exists(os.path.join(self.root, "raw", file))
                for filename in self.filenames
                for file in filename
            ]
        )

    def __len__(self):
        return len(self.data)

    def _extract_source(self, sources, Ez, resolution):
        source = torch.zeros_like(Ez)
        for src in sources:
            center = src["src_center"]  # um [-1, 0.1, 0]
            size = src["src_size"]  # um [0, 2, 0]
            # print(center, size)
            center = [int(round(i * resolution)) for i in center]  # pixel [-20, 2, 0]
            size = [int(round(i * resolution)) for i in size]  # [0, 40, 0]
            # print(center, size)
            left, right = (
                max(0, source.shape[1] // 2 + center[0] - 5 - size[0] // 2),
                source.shape[1] // 2 + center[0] + 5 + size[0] // 2,
            )
            lower, upper = (
                max(0, source.shape[2] // 2 + center[1] - 5 - size[1] // 2),
                source.shape[2] // 2 + center[1] + 5 + size[1] // 2,
            )
            # print(left, right, lower, upper)
            # print()
            source[:, left:right, lower:upper] = Ez[:, left:right, lower:upper]
            if right < 5 or lower < 5 or upper < 5:
                raise ValueError(
                    f"source is too close to the boundary, left: {left}, right: {right}, lower: {lower}, upper: {upper}"
                )

        src_mask = torch.zeros(Ez.shape[-2:])[None, ...]
        # this is to tell the model that where is the source and need to pay extra attention
        src_mask[:, left:right, lower:upper] = 1
        return source, src_mask, left, right, lower, upper

    def _gen_mseWeight(
        self,
        device_file,
        device_cfg,
        epsWeight,
        resolution,
        eps,
        eps_r,
        left,
        right,
        lower,
        upper,
    ):
        if device_file.startswith("mmi"):  # source
            mseWeight = torch.ones_like(eps)
            # we add whole row to the mask
            mseWeight[:, left : right + 50, lower - 2 : upper + 2] = np.sqrt(5)
        elif device_file.startswith("dc_N"):  # source
            mseWeight = torch.ones_like(eps)
            # we add whole row to the mask
            mseWeight[:, left : right + 50, lower - 2 : upper + 2] = np.sqrt(5)
        elif device_file.startswith("etchedmmi") or device_file.startswith("metaline"):  # source + etch
            mseWeight = torch.ones_like(eps)
            # we add whole row to the mask
            mseWeight[:, left : right + 50, lower - 2 : upper + 2] = np.sqrt(5)
            for slot in eval(device_cfg["slots"]):
                center_x = slot[0]
                center_y = slot[1]
                size_x = slot[2]
                size_y = slot[3]

                center_x_point = [
                    int(center_x * resolution) + eps.shape[1] // 2,
                    int(center_x * resolution) + eps.shape[1] // 2 + 1,
                ]
                center_y_point = [
                    int(center_y * resolution) + eps.shape[2] // 2,
                    int(center_y * resolution) + eps.shape[2] // 2 + 1,
                ]
                etch_detect_1 = (
                    eps[0, center_x_point[0], center_y_point[0]] <= 0.8 * eps_r
                    or eps[0, center_x_point[0], center_y_point[0]] >= 1.2 * eps_r
                )
                etch_detect_2 = (
                    eps[0, center_x_point[1], center_y_point[1]] <= 0.8 * eps_r
                    or eps[0, center_x_point[1], center_y_point[1]] >= 1.2 * eps_r
                )
                etch_detect_3 = (
                    eps[0, center_x_point[0], center_y_point[1]] <= 0.8 * eps_r
                    or eps[0, center_x_point[0], center_y_point[1]] >= 1.2 * eps_r
                )
                etch_detect_4 = (
                    eps[0, center_x_point[1], center_y_point[0]] <= 0.8 * eps_r
                    or eps[0, center_x_point[1], center_y_point[0]] >= 1.2 * eps_r
                )
                etch_detect = (
                    etch_detect_1 or etch_detect_2 or etch_detect_3 or etch_detect_4
                )

                top_edge = round(10000 * (center_y + size_y / 2) * resolution) // 10000
                bottom_edge = (
                    round(10000 * (center_y - size_y / 2) * resolution) // 10000
                )
                left_edge = round(10000 * (center_x - size_x / 2) * resolution) // 10000
                right_edge = (
                    round(10000 * (center_x + size_x / 2) * resolution) // 10000
                )

                if not etch_detect:  # which means that the slot center is not eps_bg
                    continue

                top_edge += eps.shape[2] // 2
                bottom_edge += eps.shape[2] // 2
                left_edge += eps.shape[1] // 2
                right_edge += eps.shape[1] // 2

                top_edge += 2
                bottom_edge -= 2
                left_edge -= 2
                right_edge += 2

                top_edge = min(top_edge, eps.shape[2])
                bottom_edge = max(bottom_edge, 0)
                left_edge = max(left_edge, 0)
                right_edge = min(right_edge, eps.shape[1])

                mseWeight[:, left_edge:right_edge, bottom_edge:top_edge] = np.sqrt(5)
        else:  # source + coupling
            mseWeight = torch.ones_like(eps)

            radius, ring_wg_width, bus_wg_widths, coupling_gaps = (
                device_cfg["radius"],
                device_cfg["ring_wg_width"],
                device_cfg["bus_wg_widths"],
                device_cfg["coupling_gaps"],
            )
            topBoxTop = (
                round(
                    10000
                    * (
                        radius
                        + (1 / 2) * ring_wg_width
                        + coupling_gaps[1]
                        + (3 / 2) * bus_wg_widths[1]
                    )
                    * resolution
                )
                // 10000
            )
            topBoxBottom = (
                round((radius - (1 / 2) * ring_wg_width) / np.sqrt(2)) * resolution
            )
            topBoxLeft = (
                round(-1 * (radius - (1 / 2) * ring_wg_width) / np.sqrt(2)) * resolution
            )
            topBoxRight = (
                round((radius - (1 / 2) * ring_wg_width) / np.sqrt(2)) * resolution
            )

            bottomBoxTop = (
                -1 * round((radius - (1 / 2) * ring_wg_width) / np.sqrt(2)) * resolution
            )
            bottomBoxBottom = (
                round(
                    -10000
                    * (
                        radius
                        + (1 / 2) * ring_wg_width
                        + coupling_gaps[0]
                        + (3 / 2) * bus_wg_widths[0]
                    )
                    * resolution
                )
                // 10000
            )
            bottomBoxLeft = (
                round(-1 * (radius - (1 / 2) * ring_wg_width) / np.sqrt(2)) * resolution
            )
            bottomBoxRight = (
                round((radius - (1 / 2) * ring_wg_width) / np.sqrt(2)) * resolution
            )

            topBoxTop += eps.shape[2] // 2
            topBoxBottom += eps.shape[2] // 2
            topBoxLeft += eps.shape[1] // 2
            topBoxRight += eps.shape[1] // 2

            bottomBoxTop += eps.shape[2] // 2
            bottomBoxBottom += eps.shape[2] // 2
            bottomBoxLeft += eps.shape[1] // 2
            bottomBoxRight += eps.shape[1] // 2

            mseWeight[:, topBoxLeft:topBoxRight, topBoxBottom:topBoxTop] = np.sqrt(5)
            mseWeight[:, bottomBoxLeft:bottomBoxRight, bottomBoxBottom:bottomBoxTop] = (
                np.sqrt(5)
            )

            mseWeight[:, left : right + 50, lower - 2 : upper + 2] = np.sqrt(5)
        # originally, the weight assign to bg is 1 and region of interest is 2, now we add extra 1 of weight to the waveguide region
        mseWeight.add_(epsWeight)  # + wg region
        return mseWeight

    def _load_one_data(self, device_file, start, multiplier=None, batch_strategy="resize_and_padding_to_square"):
        with open(os.path.join(self.root, self.processed_dir, f"{self.meta_filename}.yml"), "r") as f:
            max_size = yaml.safe_load(f)
            max_height = max_size["max_height"]
            max_width = max_size["max_width"]
            scaling_factor = self.img_size / max(max_height, max_width)
            scaling_factor = self.img_size / 534.75
        with open(
            os.path.join(self.root, "raw", device_file).replace(".h5", ".yml"),
            "r",
        ) as f:
            meta = yaml.safe_load(f)
            sources = meta["sources"][-1:]
            device_cfg = meta["device"]["cfg"]
            device_type = meta["device"]["type"]
            resolution = meta["simulation"]["resolution"]  # e.g., 20 pixels per um
            PML = meta["simulation"]["PML"]
            eps_bg = meta["device"]["cfg"]["eps_bg"]
            eps_r = meta["device"]["cfg"]["eps_r"]
        with h5py.File(os.path.join(self.root, "raw", device_file), "r") as f:
            eps = torch.from_numpy(f["eps"][()]).float()[None,]  # [1, 650, 243]
            Ez = torch.from_numpy(
                f["Ez"][start : start + self.offset_frames + self.out_frames][()]
            ).float()  # [833, 650, 243] -> [offset+out, 650, 243]
            area = eps.shape[-2] * eps.shape[-1]
            if multiplier is not None:
                Ez.mul_(multiplier)

        source, src_mask, left, right, lower, upper = self._extract_source(
            sources, Ez[self.in_frames :], resolution
        )

        epsWeight = ((eps > 1.1 * eps_bg) | (eps < 0.9 * eps_bg)).float()
        mseWeight = self._gen_mseWeight(
            device_file,
            device_cfg,
            epsWeight,
            resolution,
            eps,
            eps_r,
            left,
            right,
            lower,
            upper,
        )

        if "mmi" in device_type.lower():
            device_type = 1
        elif "mrr" in device_type.lower():
            device_type = 2
        elif "dc" in device_type.lower():
            device_type = 3
        elif "etchedmmi" in device_type.lower():
            device_type = 4
        elif "metaline" in device_type.lower():
            device_type = 5
        if "keep_spatial_res" in batch_strategy:
            data = {
                "eps": eps.sqrt(),
                "source": source,
                "Ez": Ez[: self.in_frames],
                "mseWeight": mseWeight,
                "src_mask": src_mask,
                "epsWeight": epsWeight,
                "device_type": torch.tensor([device_type]),
                "eps_bg": torch.tensor([eps_bg]),
                "scaling_factor": torch.tensor([scaling_factor]),
                "area": torch.tensor([area]),
            }
        elif batch_strategy == "resize_and_padding_to_square":
            data = {
                "eps": eps.sqrt(),  # we want to use index here, not permittivity, index is smaller
                "source": source,
                "Ez": Ez[: self.in_frames],
                "mseWeight": mseWeight,
                "src_mask": src_mask,
                "epsWeight": epsWeight,
                "device_type": device_type,
                "eps_bg": eps_bg,
                "scaling_factor": scaling_factor,
                "area": area,
            }
        else:
            raise ValueError(f"batch strategy {batch_strategy} not supported")

        if batch_strategy == "resize_and_padding_to_square":
            # resize it to the same image size
            data["eps"], grid_step = resize_and_pad_to_square(
                data["eps"], self.img_size, resolution
            )
            data["source"], _ = resize_and_pad_to_square(
                data["source"], self.img_size, resolution
            )
            data["Ez"], _ = resize_and_pad_to_square(data["Ez"], self.img_size, resolution)
            data["mseWeight"], _ = resize_and_pad_to_square(
                data["mseWeight"], self.img_size, resolution, is_mseWeight=True
            )
            data["src_mask"], _ = resize_and_pad_to_square(
                data["src_mask"], self.img_size, resolution
            )
            data["grid_step"] = grid_step
            data["epsWeight"], _ = resize_and_pad_to_square(
                data["epsWeight"], self.img_size, resolution
            )

            target = {
                "Ez": Ez[self.offset_frames : self.offset_frames + self.out_frames],
            }
            target["Ez"], _ = resize_and_pad_to_square(
                target["Ez"], self.img_size, resolution
            )

            std_in = torch.std(data["Ez"], dim=(-3, -2, -1))

            target["std"], target["mean"] = torch.std_mean(target["Ez"], dim=(-3, -2, -1))
            target["std"] = target["std"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target["stdDacayRate"] = target["std"] / std_in
        
        elif "keep_spatial_res" in batch_strategy:
            data["grid_step"] = torch.tensor([1/resolution/scaling_factor])
            target = {
                "Ez": Ez[self.offset_frames : self.offset_frames + self.out_frames],
            }

            std_in = torch.std(data["Ez"], dim=(-3, -2, -1))

            target["std"], target["mean"] = torch.std_mean(target["Ez"], dim=(-3, -2, -1))
            target["std"] = target["std"].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            target["stdDacayRate"] = target["std"] / std_in
        
        else:
            raise ValueError(f"batch strategy {batch_strategy} not supported")

        return data, target

    def __getitem__(self, item):
        # the output data has the following component:
        # the following are the keys of the data dictionary
        # eps [bs, 1, h, w] real, represents the permittivity.sqrt() which is the refractive index
        # source [bs, out_frames, h, w] real, represents the source, this is masked
        # Ez [bs, in_frames, h, w] real, represents the input field
        # mseWeight [bs, 1, h, w] real, represents the spatial weight of the loss function, redistributed the model capability to the important region such as coupling region
        # src_mask [bs, 1, h, w] real, represents the source region, used ground truth src in the predciton in the corresponding region
        # padding_mask [bs, 1, h, w] real, represents the padding region, used to mask the padding region in the loss calculation
        # device_type [bs] int, represents the device type, 1 for mmi, 2 for mrr, 3 for dc, 4 for etchedmmi, 5 for metaline
        # eps_bg [bs] real, represents the background permittivity
        # scaling_factor [bs] real, represents the scaling factor of the image
        # grid_step [bs] useless, can be just ignored in this project

        # the following are the keys of the target dictionary
        # Ez [bs, out_frames, h, w] real, represents the ground truth field to be predicted
        device_files, start = self.data[item]
        if self.train:
            start += random.randint(-8, 7)
        device_file = np.random.choice(device_files, size=1, replace=False)[0]
        if (
            self.mixup_cfg is not None
            and self.mixup_cfg.prob > 0
            and random.random() < self.mixup_cfg.prob
        ):
            device_name = (
                device_file.rstrip(".h5").split("-")[0]
                + "-"
                + device_file.rstrip(".h5").split("-")[1]
            )
            num_samples_to_add = random.randint(1, 3)
            sample_picked_list = random.sample(
                self.device_pool[device_name], num_samples_to_add
            )

            multiplier = torch.empty([1 + num_samples_to_add]).uniform_(-1, 1)
            multiplier /= multiplier.norm(2)
            sample_picked = [
                [np.random.choice(sample[0], size=1, replace=False)[0], sample[1]]
                for sample in sample_picked_list
            ]
            sample_to_mix = [[device_file, start]] + sample_picked
            multiplied_samples = []
            for i in range(1 + num_samples_to_add):
                # list of filenames with all ports and start slice index
                device_file, start = sample_to_mix[i]
                data, target = self._load_one_data(
                    device_file, start, multiplier=multiplier[i], batch_strategy=self.batch_strategy
                )

                sample = {
                    "data": data,
                    "target": target,
                }
                if self.ToTensorMarker:
                    multiplied_samples.append([sample["data"], sample["target"]])
                else:
                    sample = self.transform(sample)
                    multiplied_samples.append([sample["data"], sample["target"]])
            # now the multiplied_samples is a list of [data, target] pairs
            out_data = multiplied_samples[0][0]
            out_data["source"] = torch.stack(
                [sample[0]["source"] for sample in multiplied_samples], -1
            ).sum(-1)
            out_data["Ez"] = torch.stack(
                [sample[0]["Ez"] for sample in multiplied_samples], -1
            ).sum(-1)
            # mask is of shape [1, 256, 256]
            out_data["mseWeight"] = torch.stack(
                [sample[0]["mseWeight"] for sample in multiplied_samples], -1
            ).amax(-1)

            # mask is of shape [1, 256, 256]
            out_data["src_mask"] = torch.stack(
                [sample[0]["src_mask"] for sample in multiplied_samples], -1
            ).amax(-1)

            out_target = multiplied_samples[0][1]
            out_target["Ez"] = torch.stack(
                [sample[1]["Ez"] for sample in multiplied_samples], -1
            ).sum(-1)

            return out_data, out_target
        else:
            data, target = self._load_one_data(device_file, start, batch_strategy=self.batch_strategy)
            sample = {
                "data": data,
                "target": target,
            }
            if self.ToTensorMarker:
                return sample["data"], sample["target"]
            else:
                sample = self.transform(sample)
                return sample["data"], sample["target"]

    def extra_repr(self) -> str:
        return "Split: {}".format("Train" if self.train is True else "Test")


class FDTDDataset:
    def __init__(
        self,
        root: str,
        split: str,
        test_ratio: float,
        train_valid_split_ratio: List[float],
        device_list: List[str],
        processed_dir: str = "processed",
        mixup_cfg=None,
        img_size: int = 256,
        in_frames: int = 8,
        out_frames: int = 100,
        offset_frames: int = 2,
        batch_strategy: str = "resize_and_padding",
    ):
        self.root = root
        self.split = split
        self.test_ratio = test_ratio
        assert 0 < test_ratio < 1, print(
            f"Only support test_ratio from (0, 1), but got {test_ratio}"
        )
        self.train_valid_split_ratio = train_valid_split_ratio
        self.data = None
        self.eps_min = None
        self.eps_max = None
        self.device_list = sorted(device_list)
        self.processed_dir = processed_dir
        self.mixup_cfg = mixup_cfg
        self.in_frames = in_frames
        self.out_frames = out_frames
        self.offset_frames = offset_frames
        self.img_size = img_size

        self.batch_strategy = batch_strategy

        self.load()
        self.n_instance = len(self.data)

    def load(self):
        tran = [
            transforms.ToTensor(),
        ]

        transform = transforms.Compose(tran)

        if self.split == "train" or self.split == "valid":
            train_valid = FDTD(
                self.root,
                train=True if self.split == "train" else False,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                device_list=self.device_list,
                processed_dir=self.processed_dir,
                mixup_cfg=self.mixup_cfg if self.split == "train" else None,
                in_frames=self.in_frames,
                out_frames=self.out_frames,
                offset_frames=self.offset_frames,
                img_size=self.img_size,
                batch_strategy=self.batch_strategy,
            )

            train_len = int(self.train_valid_split_ratio[0] * len(train_valid))
            if (
                self.train_valid_split_ratio[0] + self.train_valid_split_ratio[1]
                > 0.99999
            ):
                valid_len = len(train_valid) - train_len
            else:
                valid_len = int(self.train_valid_split_ratio[1] * len(train_valid))
                train_valid.data = train_valid.data[: train_len + valid_len]

            split = [train_len, valid_len]
            train_subset, valid_subset = torch.utils.data.random_split(
                train_valid, split, generator=torch.Generator().manual_seed(1)
            )

            if self.split == "train":
                self.data = train_subset
            else:
                self.data = valid_subset

        else:
            test = FDTD(
                self.root,
                train=False,
                download=True,
                transform=transform,
                train_ratio=1 - self.test_ratio,
                device_list=self.device_list,
                processed_dir=self.processed_dir,
                mixup_cfg=None,
                in_frames=self.in_frames,
                out_frames=self.out_frames,
                offset_frames=self.offset_frames,
                img_size=self.img_size,
                batch_strategy=self.batch_strategy,
            )

            self.data = test

    def __getitem__(self, index: int) -> Dict[str, Tensor]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)

    def __call__(self, index: int) -> Dict[str, Tensor]:
        return self.__getitem__(index)


def test_FDTD():
    import pdb

    # pdb.set_trace()
    # dataset = FDTD(root="../../data", download=True, processed_dir="mmi_3x3_L_random")
    # print(len(dataset.filenames), dataset.filenames[0])
    # dataset = FDTD(
    #     root="../../data", train=False, download=True, processed_dir="mmi_3x3_L_random"
    # )
    # print(len(dataset.filenames), dataset.filenames[0])
    dataset = FDTDDataset(
        root="../../data",
        split="test",
        test_ratio=0.1,
        train_valid_split_ratio=[0.9, 0.1],
        # device_list=["mmi_3x3_L_random"],
        device_list=[
            "mmi_3x3_L_random",
            "mrr_random",
            "dc_N",
        ],  # 'etchedmmi_3x3_L_random'],
        mixup_cfg=None,
        offset_frames=8,
        in_frames=8,
        out_frames=50,
        processed_dir="processed_small",
    )
    test_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=32,
        shuffle=False,
        pin_memory=False,
        num_workers=8,
    )

    # i = 0
    # list1 = []
    # with DeterministicCtx(42):
    #     for data, target in test_loader:
    #         list1.append((data["Ez"].max(), data["Ez"].min()))
    #         i += 1
    #         if i == 100:
    #             break

    # list2 = []
    # with DeterministicCtx(42):
    #     for data, target in test_loader:
    #         list2.append((data["Ez"].max(), data["Ez"].min()))
    #         i += 1
    #         if i == 100:
    #             break

    # for i in range(len(list1)):
    #     if list1[i][0] != list2[i][0] or list1[i][1] != list2[i][1]:
    #         print("not the same")
    #         quit()
    # print("all the same")
    # quit()

    print(len(dataset))
    idx = 406
    data, target = dataset[idx]
    # data, target = dataset[116]
    # print(data)
    # print(target)
    print(data["eps"].shape)
    print(data["source"].shape)
    print(data["source"].type())
    print(data["Ez"].shape)
    print(target["Ez"].shape)

    import matplotlib.pyplot as plt

    # s, n = 0, 1
    s, n = 2, 4
    imgs = data["Ez"][s : s + n]
    eps = data["eps"]
    epsWeight = data["epsWeight"]
    mseWeight = data["mseWeight"] + eps
    imgs = torch.cat([imgs, mseWeight, epsWeight, eps], dim=0)
    n = imgs.shape[0]
    print_stat(imgs)
    print(imgs.shape)
    fig, axes = plt.subplots(1, n, constrained_layout=True, figsize=(4 * n, 8.1))
    if n == 1:
        axes = [axes]
    for i in range(imgs.shape[0]):
        # axes[i].imshow(imgs[i], cmap="RdBu_r", vmin=-0.002, vmax=0.002, origin="lower")
        axes[i].imshow(imgs[i], cmap="RdBu_r", origin="lower")
    # plt.tight_layout()
    fig.savefig("test_data.png", dpi=150)


if __name__ == "__main__":
    test_FDTD()
