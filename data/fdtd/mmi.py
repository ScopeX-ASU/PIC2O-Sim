import os
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['text.usetex'] = False
from IPython.display import Video
import h5py
import meep as mp
import numpy as np
import torch
from angler import Simulation
from pyutils.general import ensure_dir
from math import sqrt

from device import Device

eps_sio2 = 1.44**2
eps_si = 3.48**2

__all__ = ["MMI_NxM", "mmi_2x2", "mmi_3x3", "mmi_4x4", "mmi_6x6", "mmi_8x8"]


def get_taper(
    width_wg1: float,
    width_wg2: float,
    length_wg1: float,
    length_wg2: float,
    length_taper: float,
    center: Tuple[float, float] = (0, 0),
    medium: mp.Medium = mp.Medium(epsilon=eps_si),
):
    """Generate a taper shape for meep simulation
    https://meep.readthedocs.io/en/latest/Python_Tutorials/Mode_Decomposition/

    Args:
        width_wg1 (float): left waveguide width. unit of um
        width_wg2 (float): right waveguide width. unit of um
        length_wg1 (float): left waveguide length. unit of um
        length_wg2 (float): right waveguide length. unit of um
        length_taper (float): taper length. unit of um
        center (Tuple[float, float], optional): Center coordinate (x, y). Defaults to (0, 0).
        medium (mp.Medium, optional): meep Medium for the taper. Defaults to mp.Medium(epsilon=eps_si).

    Returns:
        _type_: _description_
    """
    left1 = center[0] - length_taper / 2 - length_wg1
    left2 = center[0] - length_taper / 2
    right1 = center[0] + length_taper / 2 + length_wg2
    right2 = center[0] + length_taper / 2
    top1 = center[1] + width_wg1 / 2
    btm1 = center[1] - width_wg1 / 2
    top2 = center[1] + width_wg2 / 2
    btm2 = center[1] - width_wg2 / 2

    ## cannot have duplicate points, otherwise it will impact the centroid calculation.
    taper_vertices = [mp.Vector3(left1, top1)]
    if length_wg1 > 0:
        taper_vertices += [mp.Vector3(left2, top1)]
    taper_vertices += [mp.Vector3(right2, top2)]
    if length_wg2 > 0:
        taper_vertices += [mp.Vector3(right1, top2), mp.Vector3(right1, btm2)]
    taper_vertices += [
        mp.Vector3(right2, btm2),
        mp.Vector3(left2, btm1),
    ]
    if length_wg1 > 0:
        taper_vertices += [
            mp.Vector3(left1, btm1),
        ]

    taper = mp.Prism(
        taper_vertices,
        height=mp.inf,
        material=medium,
    )
    return taper


def apply_regions(reg_list, xs, ys, eps_r_list, eps_bg):
    # feed this function a list of regions and some coordinates and it spits out a permittivity
    if isinstance(eps_r_list, (int, float)):
        eps_r_list = [eps_r_list] * len(reg_list)
    # if it's not a list, make it one
    if not isinstance(reg_list, (list, tuple)):
        reg_list = [reg_list]

    # initialize permittivity
    eps_r = np.zeros(xs.shape) + eps_bg

    # loop through lambdas and apply masks
    for e, reg in zip(eps_r_list, reg_list):
        reg_vec = np.vectorize(reg)
        material_mask = reg_vec(xs, ys)
        eps_r[material_mask] = e

    return eps_r


def gaussian_blurring(x):
    # return x
    x = torch.from_numpy(x).unsqueeze(0).unsqueeze(0).float()
    size = 3
    std = 0.4
    ax = torch.linspace(-(size - 1) / 2.0, (size - 1) / 2.0, size)
    xx, yy = torch.meshgrid(ax, ax, indexing="xy")
    kernel = torch.exp(-0.5 / std**2 * (xx**2 + yy**2))
    kernel = kernel.div(kernel.sum()).unsqueeze(0).unsqueeze(0).float()
    return (
        torch.nn.functional.conv2d(x, kernel, padding=size // 2)
        .squeeze(0)
        .squeeze(0)
        .numpy()
    )


class MMI_NxM(Device):
    def __init__(
        self,
        num_in_ports: int,
        num_out_ports: int,
        box_size: Tuple[float, float],  # box [length, width], um
        wg_width: Tuple[float, float] = (0.4, 0.4),  # in/out wavelength width, um
        port_diff: Tuple[float, float] = (
            4,
            4,
        ),  # distance between in/out waveguides. um
        port_len: float = 10,  # length of in/out waveguide from PML to box. um
        taper_width: float = 0.0,  # taper width near the multi-mode region. um. Default to 0
        taper_len: float = 0.0,  # taper length. um. Default to 0
        eps_r: float = eps_si,  # relative refractive index
        eps_bg: float = eps_sio2,  # background refractive index
    ):
        # remove invalid taper
        if taper_width < 1e-5 or taper_len < 1e-5:
            taper_width = taper_len = 0

        assert (
            max(taper_width, wg_width[0]) * num_in_ports <= box_size[1]
        ), "The input ports cannot fit the multimode region"
        assert (
            max(taper_width, wg_width[1]) * num_out_ports <= box_size[1]
        ), "The output ports cannot fit the multimode region"
        if taper_width > 1e-5:
            assert (
                taper_width >= wg_width[0]
            ), "Taper width cannot be smaller than input waveguide width"
            assert (
                taper_width >= wg_width[1]
            ), "Taper width cannot be smaller than output waveguide width"

        device_cfg = dict(
            num_in_ports=num_in_ports,
            num_out_ports=num_out_ports,
            box_size=box_size,
            wg_width=wg_width,
            port_diff=port_diff,
            port_len=port_len,
            taper_width=taper_width,
            taper_len=taper_len,
            eps_r=eps_r,
            eps_bg=eps_bg,
        )
        super().__init__(**device_cfg)
        self.pad_regions = []

        self.update_device_config("MMI_NxM", device_cfg)

        self.size = [box_size[0] + port_len * 2, box_size[1]]

        # meep definition
        box = mp.Block(
            mp.Vector3(box_size[0], box_size[1], mp.inf),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=eps_r),
        )
        in_ports = [
            get_taper(
                width_wg1=wg_width[0],
                width_wg2=taper_width,
                length_wg1=port_len - taper_len + 2,
                length_wg2=0,
                length_taper=taper_len,
                center=(
                    -box_size[0] / 2 - taper_len / 2,
                    (i - (num_in_ports - 1) / 2) * port_diff[0],
                ),
                medium=mp.Medium(epsilon=eps_r),
            )
            for i in range(num_in_ports)
        ]

        out_ports = [
            get_taper(
                width_wg1=taper_width,
                width_wg2=wg_width[1],
                length_wg1=0,
                length_wg2=port_len - taper_len + 2,
                length_taper=taper_len,
                center=(
                    box_size[0] / 2 + taper_len / 2,
                    (i - (num_out_ports - 1) / 2) * port_diff[0],
                ),
                medium=mp.Medium(epsilon=eps_r),
            )
            for i in range(num_out_ports)
        ]
        self.geometry = [box] + in_ports + out_ports

        # self.design_region = apply_regions(
        #     [box], self.xs, self.ys, eps_r_list=1, eps_bg=0
        # )
        self.pad_regions = None

        self.in_port_centers = [
            (
                -box_size[0] / 2 - 0.98 * port_len,
                (i - (num_in_ports - 1) / 2) * port_diff[0],
            )
            for i in range(num_in_ports)
        ]  # centers

        self.out_port_centers = [
            (
                box_size[0] / 2 + 0.98 * port_len,
                (float(i) - float(num_out_ports - 1) / 2.0) * port_diff[1],
            )
            for i in range(num_out_ports)
        ]  # centers

    def add_source(
        self,
        in_port_idx: int,
        src_type: str = "GaussianSource",
        wl_cen=1.55,
        wl_width: float = 0.1,
        alpha: float = 0.5,
    ):
        fcen = 1 / wl_cen  # pulse center frequency
        ## alpha from 1/3 to 1/2
        fwidth = (
            3 * alpha * (1 / (wl_cen - wl_width / 2) - 1 / (wl_cen + wl_width / 2))
        )  # pulse frequency width
        if src_type == "GaussianSource":
            src_fn = mp.GaussianSource
        else:
            raise NotImplementedError
        src_center = list(self.in_port_centers[in_port_idx]) + [0]
        src_size = (0, 1.5 * self.wg_width[0], 0)
        self.sources.append(
            mp.EigenModeSource(
                src=src_fn(fcen, fwidth=fwidth),
                center=mp.Vector3(*src_center),
                size=src_size,
                eig_match_freq=True,
                eig_parity=mp.ODD_Z + mp.EVEN_Y,
            )
        )

        self.add_source_config(
            dict(
                src_type=src_type,
                in_port_idx=in_port_idx,
                src_center=src_center,
                src_size=src_size,
                eig_match_freq=True,
                eig_parity=mp.ODD_Z + mp.EVEN_Y,
                wl_cen=wl_cen,
                wl_width=wl_width,
                alpha=alpha,
            )
        )

    def create_simulation(
        self,
        resolution: int = 10,  # pixels / um
        border_width: Tuple[float, float] = [1, 1],  # um, [x, y]
        PML: Tuple[int, int] = (2, 2),  # um, [x, y]
        record_interval: float = 0.3,
        store_fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        until: float = None,  # timesteps
        stop_when_decay: bool = False,
    ):
        boundary = [
            mp.PML(PML[0], direction=mp.X),
            mp.PML(PML[1], direction=mp.Y),
        ]

        sx = PML[0] * 2 + self.size[0] + border_width[0] * 2
        sy = PML[1] * 2 + self.size[1] + border_width[1] * 2
        cell_size = (sx, sy, 0)
        self.sim = mp.Simulation(
            resolution=resolution,
            cell_size=mp.Vector3(*cell_size),
            boundary_layers=boundary,
            geometry=self.geometry + self.pad_regions,
            sources=self.sources,
            default_material=mp.Medium(epsilon=self.config.device.cfg.eps_bg),
        )
        self.update_simulation_config(
            dict(
                resolution=resolution,
                border_width=border_width,
                PML=PML,
                cell_size=cell_size,
                record_interval=record_interval,
                store_fields=store_fields,
                until=until,
                stop_when_decay=stop_when_decay,
            )
        )
        return self.sim

    def run_sim(
        self,
        filepath: str = None,
        export_video: bool = False,
    ):
        stop_when_decay = self.config.simulation.stop_when_decay
        output = dict(
            eps=None,
            Ex=[],
            Ey=[],
            Ez=[],
            Hx=[],
            Hy=[],
            Hz=[],
        )
        store_fields = self.config.simulation.store_fields

        def record_fields(sim):
            for field in store_fields:
                if field == "Ex":
                    data = sim.get_efield_x()
                elif field == "Ey":
                    data = sim.get_efield_y()
                elif field == "Ez":
                    data = sim.get_efield_z()
                elif field == "Hx":
                    data = sim.get_hfield_x()
                elif field == "Hy":
                    data = sim.get_hfield_y()
                elif field == "Hz":
                    data = sim.get_hfield_z()
                output[field].append(data)
        at_every = [record_fields]
        if export_video:
            f = plt.figure(dpi=150)
            Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
            at_every.append(Animate)

        if stop_when_decay:
            monitor_cen = list(self.out_port_centers[0]) + [0]

            self.sim.run(
                mp.at_every(self.config.simulation.record_interval, *at_every),
                until_after_sources=mp.stop_when_fields_decayed(
                    50, mp.Ez, monitor_cen, 1e-9
                ),
            )
        else:            
            self.sim.run(
                mp.at_every(self.config.simulation.record_interval, *at_every),
                until=self.config.simulation.until,
            )
        if export_video:
            filename = filepath.rstrip(".h5") + ".mp4"
            Animate.to_mp4(20, filename)
            Video(filename)
        # self.sim.plot2D(fields=mp.Ez)
        PML, res = self.config.simulation.PML, self.config.simulation.resolution
        output["eps"] = self.trim_pml(res, PML,self.sim.get_epsilon().astype(np.float16))
        
        for field, data in output.items():
            if isinstance(data, list) and len(data) > 0:
                output[field] = self.trim_pml(res, PML,np.array(data))
        ensure_dir(os.path.dirname(filepath))
        if filepath is not None:
            hf = h5py.File(filepath, "w")
            hf.create_dataset("eps", data=output["eps"])
            max_vals = (
                np.max(np.abs(output["Ex"])),
                np.max(np.abs(output["Ey"])),
                np.max(np.abs(output["Ez"])),
                np.max(np.abs(output["Hx"])),
                np.max(np.abs(output["Hy"])),
                np.max(np.abs(output["Hz"])),
            )
            # print(max_vals)
            max_val = max(max_vals)
            # print(np.mean(output["Ez"]))
            # print(np.std(output["Ez"]))
            self.config.simulation.update(dict(field_max_val=max_val.item()))
            # hf.create_dataset("Ex", data=(output["Ex"] / max_val).astype(np.float32))
            # hf.create_dataset("Ey", data=(output["Ey"] / max_val).astype(np.float32))
            hf.create_dataset("Ez", data=(output["Ez"] / max_val).astype(np.float16))
            # hf.create_dataset("Hx", data=(output["Hx"] / max_val).astype(np.float16))
            # hf.create_dataset("Hy", data=(output["Hy"] / max_val).astype(np.float16))
            # hf.create_dataset("Hz", data=(output["Hz"] / max_val).astype(np.float32))
            hf.create_dataset("meta", data=str(self.config))

        return output

    def set_pad_region(self, pad_regions, pad_eps):
        # pad_regions = [[xl, xh, yl, yh], [xl, xh, yl, yh], ...] rectanglar pads bounding box
        # (0,0) is the center of the entire region
        # default argument in lambda can avoid lazy evaluation in python!
        self.pad_regions = [
            mp.Block(
            mp.Vector3(xh-xl, yh-yl, mp.inf),
            center=mp.Vector3((xh+xl)/2, (yh+yl)/2),
            material=mp.Medium(epsilon=eps),
        )
        for (xl, xh, yl, yh), eps in zip(pad_regions, pad_eps)
        ]

    def resize(self, x, size, mode="bilinear"):
        if not isinstance(x, torch.Tensor):
            y = torch.from_numpy(x)
        else:
            y = x
        y = y.view(-1, 1, x.shape[-2], x.shape[-1])
        old_grid_step = (self.grid_step, self.grid_step)
        old_size = y.shape[-2:]
        new_grid_step = [
            old_size[0] / size[0] * old_grid_step[0],
            old_size[1] / size[1] * old_grid_step[1],
        ]
        if y.is_complex():
            y = torch.complex(
                torch.nn.functional.interpolate(y.real, size=size, mode=mode),
                torch.nn.functional.interpolate(y.imag, size=size, mode=mode),
            )
        else:
            y = torch.nn.functional.interpolate(y, size=size, mode=mode)
        y = y.view(list(x.shape[:-2]) + list(size))
        if isinstance(x, np.ndarray):
            y = y.numpy()
        return y, new_grid_step

    def extract_transfer_matrix(
        self, eps_map: torch.Tensor, wavelength: float = 1.55, pol: str = "Hz"
    ) -> torch.Tensor:
        # extract the transfer matrix of the N-port MMI from input ports to output ports up to a global phase
        c0 = 299792458
        source_amp = (
            1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
        )
        neff_si = 3.48
        lambda0 = wavelength / 1e6  # free space wavelength (m)
        omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
        transfer_matrix = np.zeros(
            [self.num_out_ports, self.num_in_ports], dtype=np.complex64
        )
        for i in range(self.num_in_ports):
            simulation = Simulation(omega, eps_map, self.grid_step, self.NPML, pol)
            simulation.add_mode(
                neff=neff_si,
                direction_normal="x",
                center=self.in_port_centers_px[i],
                width=int(2 * self.in_port_width_px[i]),
                scale=source_amp,
            )
            simulation.setup_modes()
            # eigenmode analysis
            center = self.in_port_centers_px[i]
            width = int(2 * self.in_port_width_px[i])
            inds_y = [int(center[1] - width / 2), int(center[1] + width / 2)]
            eigen_mode = simulation.src[center[0], inds_y[0] : inds_y[1]].conj()

            simulation.solve_fields()
            if pol == "Hz":
                field = simulation.fields["Hz"]
            else:
                field = simulation.fields["Ez"]
            input_field = field[center[0], inds_y[0] : inds_y[1]]
            input_field_mode = input_field.dot(eigen_mode)
            for j in range(self.num_out_ports):
                out_center = self.out_port_pixel_centers[j]
                out_width = int(2 * self.out_port_width_px[j])
                out_inds_y = [
                    int(out_center[1] - out_width / 2),
                    int(out_center[1] + out_width / 2),
                ]
                output_field = field[out_center[0], out_inds_y[0] : out_inds_y[1]]
                s21 = output_field.dot(eigen_mode) / input_field_mode
                transfer_matrix[j, i] = s21
        return transfer_matrix

    def __repr__(self) -> str:
        str = f"MMI{self.num_in_ports}x{self.num_out_ports}("
        str += f"size = {self.box_size[0]} um x {self.box_size[1]} um)"
        return str


class EtchedMMI_NxM(Device):
    def __init__(
        self,
        num_in_ports: int,
        num_out_ports: int,
        box_size: Tuple[float, float],  # box [length, width], um
        wg_width: Tuple[float, float] = (0.4, 0.4),  # in/out wavelength width, um
        port_diff: Tuple[float, float] = (
            4,
            4,
        ),  # distance between in/out waveguides. um
        slots: Optional[Tuple[int, int]] = [],  # [(center_x, center_y, size_x, size_y)]
        port_len: float = 10,  # length of in/out waveguide from PML to box. um
        taper_width: float = 0.0,  # taper width near the multi-mode region. um. Default to 0
        taper_len: float = 0.0,  # taper length. um. Default to 0
        eps_r: float = eps_si,  # relative refractive index
        eps_bg: float = eps_sio2,  # background refractive index
    ):
        # remove invalid taper
        if taper_width < 1e-5 or taper_len < 1e-5:
            taper_width = taper_len = 0

        assert (
            max(taper_width, wg_width[0]) * num_in_ports <= box_size[1]
        ), "The input ports cannot fit the multimode region"
        assert (
            max(taper_width, wg_width[1]) * num_out_ports <= box_size[1]
        ), "The output ports cannot fit the multimode region"
        if taper_width > 1e-5:
            assert (
                taper_width >= wg_width[0]
            ), "Taper width cannot be smaller than input waveguide width"
            assert (
                taper_width >= wg_width[1]
            ), "Taper width cannot be smaller than output waveguide width"

        device_cfg = dict(
            num_in_ports=num_in_ports,
            num_out_ports=num_out_ports,
            slots=str(slots),
            box_size=box_size,
            wg_width=wg_width,
            port_diff=port_diff,
            port_len=port_len,
            taper_width=taper_width,
            taper_len=taper_len,
            eps_r=eps_r,
            eps_bg=eps_bg,
        )
        super().__init__(**device_cfg)
        self.pad_regions = []

        self.update_device_config("EtchedMMI_NxM", device_cfg)

        self.size = [box_size[0] + port_len * 2, box_size[1]]

        # meep definition
        box = mp.Block(
            mp.Vector3(box_size[0], box_size[1], mp.inf),
            center=mp.Vector3(),
            material=mp.Medium(epsilon=eps_r),
        )
        with open("slots_record.log", "w") as f:
            print("this is the slots: ", slots, file=f)
        etched_boxes = [
            mp.Block(
                mp.Vector3(size_x, size_y, mp.inf),
                center=mp.Vector3(center_x, center_y),
                material=mp.Medium(epsilon=eps_bg),
            )
            for center_x, center_y, size_x, size_y in slots
        ]
        in_ports = [
            get_taper(
                width_wg1=wg_width[0],
                width_wg2=taper_width,
                length_wg1=port_len - taper_len + 2,
                length_wg2=0,
                length_taper=taper_len,
                center=(
                    -box_size[0] / 2 - taper_len / 2,
                    (i - (num_in_ports - 1) / 2) * port_diff[0],
                ),
                medium=mp.Medium(epsilon=eps_r),
            )
            for i in range(num_in_ports)
        ]

        out_ports = [
            get_taper(
                width_wg1=taper_width,
                width_wg2=wg_width[1],
                length_wg1=0,
                length_wg2=port_len - taper_len + 2,
                length_taper=taper_len,
                center=(
                    box_size[0] / 2 + taper_len / 2,
                    (i - (num_out_ports - 1) / 2) * port_diff[0],
                ),
                medium=mp.Medium(epsilon=eps_r),
            )
            for i in range(num_out_ports)
        ]
        self.geometry = [box] + in_ports + out_ports + etched_boxes

        # self.design_region = apply_regions(
        #     [box], self.xs, self.ys, eps_r_list=1, eps_bg=0
        # )
        self.pad_regions = None

        self.in_port_centers = [
            (
                -box_size[0] / 2 - 0.98 * port_len,
                (i - (num_in_ports - 1) / 2) * port_diff[0],
            )
            for i in range(num_in_ports)
        ]  # centers

        self.out_port_centers = [
            (
                box_size[0] / 2 + 0.98 * port_len,
                (float(i) - float(num_out_ports - 1) / 2.0) * port_diff[1],
            )
            for i in range(num_out_ports)
        ]  # centers

    def add_source(
        self,
        in_port_idx: int,
        src_type: str = "GaussianSource",
        wl_cen=1.55,
        wl_width: float = 0.1,
        alpha: float = 0.5,
    ):
        fcen = 1 / wl_cen  # pulse center frequency
        ## alpha from 1/3 to 1/2
        fwidth = (
            3 * alpha * (1 / (wl_cen - wl_width / 2) - 1 / (wl_cen + wl_width / 2))
        )  # pulse frequency width
        if src_type == "GaussianSource":
            src_fn = mp.GaussianSource
        else:
            raise NotImplementedError
        src_center = list(self.in_port_centers[in_port_idx]) + [0]
        src_size = (0, 1.5 * self.wg_width[0], 0)
        self.sources.append(
            mp.EigenModeSource(
                src=src_fn(fcen, fwidth=fwidth),
                center=mp.Vector3(*src_center),
                size=src_size,
                eig_match_freq=True,
                eig_parity=mp.ODD_Z + mp.EVEN_Y,
            )
        )

        self.add_source_config(
            dict(
                src_type=src_type,
                in_port_idx=in_port_idx,
                src_center=src_center,
                src_size=src_size,
                eig_match_freq=True,
                eig_parity=mp.ODD_Z + mp.EVEN_Y,
                wl_cen=wl_cen,
                wl_width=wl_width,
                alpha=alpha,
            )
        )

    def create_simulation(
        self,
        resolution: int = 10,  # pixels / um
        border_width: Tuple[float, float] = [1, 1],  # um, [x, y]
        PML: Tuple[int, int] = (2, 2),  # um, [x, y]
        record_interval: float = 0.3,
        store_fields=["Ex", "Ey", "Ez", "Hx", "Hy", "Hz"],
        until: float = None,  # timesteps
        stop_when_decay: bool = False,
    ):
        boundary = [
            mp.PML(PML[0], direction=mp.X),
            mp.PML(PML[1], direction=mp.Y),
        ]

        sx = PML[0] * 2 + self.size[0] + border_width[0] * 2
        sy = PML[1] * 2 + self.size[1] + border_width[1] * 2
        cell_size = (sx, sy, 0)
        self.sim = mp.Simulation(
            resolution=resolution,
            cell_size=mp.Vector3(*cell_size),
            boundary_layers=boundary,
            geometry=self.geometry + self.pad_regions,
            sources=self.sources,
            default_material=mp.Medium(epsilon=self.config.device.cfg.eps_bg),
        )
        self.update_simulation_config(
            dict(
                resolution=resolution,
                border_width=border_width,
                PML=PML,
                cell_size=cell_size,
                record_interval=record_interval,
                store_fields=store_fields,
                until=until,
                stop_when_decay=stop_when_decay,
            )
        )
        return self.sim

    def run_sim(
        self,
        filepath: str = None,
        export_video: bool = False,
    ):
        stop_when_decay = self.config.simulation.stop_when_decay
        output = dict(
            eps=None,
            Ex=[],
            Ey=[],
            Ez=[],
            Hx=[],
            Hy=[],
            Hz=[],
        )
        store_fields = self.config.simulation.store_fields

        def record_fields(sim):
            for field in store_fields:
                if field == "Ex":
                    data = sim.get_efield_x()
                elif field == "Ey":
                    data = sim.get_efield_y()
                elif field == "Ez":
                    data = sim.get_efield_z()
                elif field == "Hx":
                    data = sim.get_hfield_x()
                elif field == "Hy":
                    data = sim.get_hfield_y()
                elif field == "Hz":
                    data = sim.get_hfield_z()
                output[field].append(data)
        at_every = [record_fields]
        if export_video:
            f = plt.figure(dpi=150)
            Animate = mp.Animate2D(fields=mp.Ez, f=f, realtime=False, normalize=True)
            at_every.append(Animate)

        if stop_when_decay:
            monitor_cen = list(self.out_port_centers[0]) + [0]

            self.sim.run(
                mp.at_every(self.config.simulation.record_interval, *at_every),
                until_after_sources=mp.stop_when_fields_decayed(
                    50, mp.Ez, monitor_cen, 1e-9
                ),
            )
        else:            
            self.sim.run(
                mp.at_every(self.config.simulation.record_interval, *at_every),
                until=self.config.simulation.until,
            )
        if export_video:
            filename = filepath[:-3] + ".mp4"
            Animate.to_mp4(20, filename)
            Video(filename)
        # self.sim.plot2D(fields=mp.Ez)
        PML, res = self.config.simulation.PML, self.config.simulation.resolution
        output["eps"] = self.trim_pml(res, PML,self.sim.get_epsilon().astype(np.float16))
        
        for field, data in output.items():
            if isinstance(data, list) and len(data) > 0:
                output[field] = self.trim_pml(res, PML,np.array(data))
        ensure_dir(os.path.dirname(filepath))
        if filepath is not None:
            hf = h5py.File(filepath, "w")
            hf.create_dataset("eps", data=output["eps"])
            max_vals = (
                np.max(np.abs(output["Ex"])),
                np.max(np.abs(output["Ey"])),
                np.max(np.abs(output["Ez"])),
                np.max(np.abs(output["Hx"])),
                np.max(np.abs(output["Hy"])),
                np.max(np.abs(output["Hz"])),
            )
            # print(max_vals)
            max_val = max(max_vals)
            # print(np.mean(output["Ez"]))
            # print(np.std(output["Ez"]))
            self.config.simulation.update(dict(field_max_val=max_val.item()))
            # hf.create_dataset("Ex", data=(output["Ex"] / max_val).astype(np.float32))
            # hf.create_dataset("Ey", data=(output["Ey"] / max_val).astype(np.float32))
            hf.create_dataset("Ez", data=(output["Ez"] / max_val).astype(np.float16))
            # hf.create_dataset("Hx", data=(output["Hx"] / max_val).astype(np.float16))
            # hf.create_dataset("Hy", data=(output["Hy"] / max_val).astype(np.float16))
            # hf.create_dataset("Hz", data=(output["Hz"] / max_val).astype(np.float32))
            hf.create_dataset("meta", data=str(self.config))

        return output

    def set_pad_region(self, pad_regions, pad_eps):
        # pad_regions = [[xl, xh, yl, yh], [xl, xh, yl, yh], ...] rectanglar pads bounding box
        # (0,0) is the center of the entire region
        # default argument in lambda can avoid lazy evaluation in python!
        self.pad_regions = [
            mp.Block(
            mp.Vector3(xh-xl, yh-yl, mp.inf),
            center=mp.Vector3((xh+xl)/2, (yh+yl)/2),
            material=mp.Medium(epsilon=eps),
        )
        for (xl, xh, yl, yh), eps in zip(pad_regions, pad_eps)
        ]

    def resize(self, x, size, mode="bilinear"):
        if not isinstance(x, torch.Tensor):
            y = torch.from_numpy(x)
        else:
            y = x
        y = y.view(-1, 1, x.shape[-2], x.shape[-1])
        old_grid_step = (self.grid_step, self.grid_step)
        old_size = y.shape[-2:]
        new_grid_step = [
            old_size[0] / size[0] * old_grid_step[0],
            old_size[1] / size[1] * old_grid_step[1],
        ]
        if y.is_complex():
            y = torch.complex(
                torch.nn.functional.interpolate(y.real, size=size, mode=mode),
                torch.nn.functional.interpolate(y.imag, size=size, mode=mode),
            )
        else:
            y = torch.nn.functional.interpolate(y, size=size, mode=mode)
        y = y.view(list(x.shape[:-2]) + list(size))
        if isinstance(x, np.ndarray):
            y = y.numpy()
        return y, new_grid_step

    def extract_transfer_matrix(
        self, eps_map: torch.Tensor, wavelength: float = 1.55, pol: str = "Hz"
    ) -> torch.Tensor:
        # extract the transfer matrix of the N-port MMI from input ports to output ports up to a global phase
        c0 = 299792458
        source_amp = (
            1e-9  # amplitude of modal source (make around 1 for nonlinear effects)
        )
        neff_si = 3.48
        lambda0 = wavelength / 1e6  # free space wavelength (m)
        omega = 2 * np.pi * c0 / lambda0  # angular frequency (2pi/s)
        transfer_matrix = np.zeros(
            [self.num_out_ports, self.num_in_ports], dtype=np.complex64
        )
        for i in range(self.num_in_ports):
            simulation = Simulation(omega, eps_map, self.grid_step, self.NPML, pol)
            simulation.add_mode(
                neff=neff_si,
                direction_normal="x",
                center=self.in_port_centers_px[i],
                width=int(2 * self.in_port_width_px[i]),
                scale=source_amp,
            )
            simulation.setup_modes()
            # eigenmode analysis
            center = self.in_port_centers_px[i]
            width = int(2 * self.in_port_width_px[i])
            inds_y = [int(center[1] - width / 2), int(center[1] + width / 2)]
            eigen_mode = simulation.src[center[0], inds_y[0] : inds_y[1]].conj()

            simulation.solve_fields()
            if pol == "Hz":
                field = simulation.fields["Hz"]
            else:
                field = simulation.fields["Ez"]
            input_field = field[center[0], inds_y[0] : inds_y[1]]
            input_field_mode = input_field.dot(eigen_mode)
            for j in range(self.num_out_ports):
                out_center = self.out_port_pixel_centers[j]
                out_width = int(2 * self.out_port_width_px[j])
                out_inds_y = [
                    int(out_center[1] - out_width / 2),
                    int(out_center[1] + out_width / 2),
                ]
                output_field = field[out_center[0], out_inds_y[0] : out_inds_y[1]]
                s21 = output_field.dot(eigen_mode) / input_field_mode
                transfer_matrix[j, i] = s21
        return transfer_matrix

    def __repr__(self) -> str:
        str = f"EtchedMMI{self.num_in_ports}x{self.num_out_ports}("
        str += f"size = {self.box_size[0]} um x {self.box_size[1]} um)"
        return str
    

def mmi_2x2():
    N = 2
    wl = 1.55
    index_si = 3.48
    size = (12, 3)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (10, 0.8)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_2x2_L_random():
    N = 2
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(15, 20), np.random.uniform(3, 5))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_3x3():
    N = 3
    wl = 1.55
    size = (13.5, 3.5)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(0.8, 0.8),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        taper_width=1.1,
        taper_len=2,
        eps_r=eps_si,
        eps_bg=eps_sio2,
    )
    w, h = (10, 1)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    pad_eps = [11.8, 11.5, 12.2]
    mmi.set_pad_region(pad_regions, pad_eps)


    mmi.add_source(0)
    mmi.create_simulation(
        20, stop_when_decay=False, until=200, border_width=[0, 1], PML=[2, 2]
    )
    mmi.run_sim(filepath="./raw/mmi_3x3.h5", export_video=True)
    mmi.dump_config("./raw/mmi_3x3.yml")
    return mmi


def mmi_3x3_L():
    N = 3
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    size = (25.9, 6.1)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        # wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        wg_width=(1.1, 1.1),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (20, 1.2)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_3x3_L_random(random_seed=0, port_idx=0):
    np.random.seed(random_seed)
    N = 3
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.0)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        taper_width=wg_width+0.3,
        taper_len=2,
        eps_r=eps_si,
        eps_bg=eps_sio2,
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    pad_eps = [np.random.uniform(eps_si-0.4, eps_si+0.4) for _ in range(N)]
    mmi.set_pad_region(pad_regions, pad_eps)

    mmi.add_source(port_idx, wl_width=0.1)
    mmi.create_simulation(
        resolution=20, 
        stop_when_decay=False, 
        until=250, 
        border_width=[0, 1], 
        PML=[2, 2],
    )
    

    return mmi

def mmi_3x3_L_determined(id=0, port_idx=1):
    min_w = 18
    min_h = 4.5
    w_step = (18*sqrt(20)-18)/10
    h_step = (4.5*sqrt(20)-4.5)/10
    size = (min_w + id * w_step, min_h + id * h_step)
    N = 3
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.0)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        taper_width=wg_width+0.3,
        taper_len=2,
        eps_r=eps_si,
        eps_bg=eps_sio2,
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    pad_eps = [np.random.uniform(eps_si-0.4, eps_si+0.4) for _ in range(N)]
    mmi.set_pad_region(pad_regions, pad_eps)

    mmi.add_source(port_idx, wl_width=0.1)
    mmi.create_simulation(
        resolution=15, 
        stop_when_decay=False, 
        until=250, 
        border_width=[0, 1], 
        PML=[2, 2],
    )

    return mmi

def mmi_3x3_L_swp_res(res=0, port_idx=1):
    np.random.seed(42)
    width = 25
    height = 6
    size = (width, height)
    N = 3
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.0)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        taper_width=wg_width+0.3,
        taper_len=2,
        eps_r=eps_si,
        eps_bg=eps_sio2,
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    pad_eps = [np.random.uniform(eps_si-0.4, eps_si+0.4) for _ in range(N)]
    mmi.set_pad_region(pad_regions, pad_eps)

    mmi.add_source(port_idx, wl_width=0.1)
    mmi.create_simulation(
        resolution=res, 
        stop_when_decay=False, 
        until=250, 
        border_width=[0, 1], 
        PML=[2, 2],
    )
    return mmi

def etched_mmi_3x3_L_random(random_seed=0, port_idx=0):
    np.random.seed(random_seed)
    N = 3
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.0)
    
    n_slots = (30, 7)
    total_slots = n_slots[0] * n_slots[1]
    n_sampled_slots = int(np.random.uniform(0.1, 0.2) * total_slots)
    w, h = (
        size[0] / n_slots[0] * 0.8,
        size[1] / n_slots[1] * 0.8,
    )  # do not remove materials on the boundary
    slot_centers_x = np.linspace(
        -(n_slots[0] / 2 - 0.5) * w, (n_slots[0] / 2 - 0.5) * w, n_slots[0]
    )
    slot_centers_y = np.linspace(
        -(n_slots[1] / 2 - 0.5) * h, (n_slots[1] / 2 - 0.5) * h, n_slots[1]
    )

    centers_x = np.random.choice(slot_centers_x, size=n_sampled_slots, replace=True)
    centers_y = slot_centers_y[
        (
            np.round(
                np.random.choice(
                    len(slot_centers_y), size=n_sampled_slots, replace=True
                )
            )
        ).astype(np.int32)
    ]  # a trick to generate slots along the prop direction
    slots = [(x, y, w, h) for x, y in zip(centers_x, centers_y)]

    mmi = EtchedMMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        slots=slots,
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        taper_width=wg_width+0.3,
        taper_len=2,
        eps_r=eps_si,
        eps_bg=eps_sio2,
    )
    # w, h = (
    #     size[0] * np.random.uniform(0.7, 0.9),
    #     size[1] / N * np.random.uniform(0.4, 0.65),
    # )
    # pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    # pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    # pad_eps = [np.random.uniform(eps_si-0.4, eps_si+0.4) for _ in range(N)]
    # mmi.set_pad_region(pad_regions, pad_eps)

    mmi.add_source(port_idx)
    mmi.create_simulation(
        resolution=20, 
        stop_when_decay=False, 
        until=300, 
        border_width=[0, 1], 
        PML=[2, 2],
    )
    return mmi


def mmi_3x3_L_random_slots():
    ## random rectangular SiO2 slots
    N = 3
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    n_slots = (30, 7)
    total_slots = n_slots[0] * n_slots[1]
    n_sampled_slots = int(np.random.uniform(0.05, 0.1) * total_slots)
    w, h = (
        size[0] / n_slots[0] * 0.8,
        size[1] / n_slots[1] * 0.8,
    )  # do not remove materials on the boundary
    slot_centers_x = np.linspace(
        -(n_slots[0] / 2 - 0.5) * w, (n_slots[0] / 2 - 0.5) * w, n_slots[0]
    )
    slot_centers_y = np.linspace(
        -(n_slots[1] / 2 - 0.5) * h, (n_slots[1] / 2 - 0.5) * h, n_slots[1]
    )

    centers_x = np.random.choice(slot_centers_x, size=n_sampled_slots, replace=True)
    centers_y = slot_centers_y[
        (
            np.round(
                np.random.choice(
                    len(slot_centers_y), size=n_sampled_slots, replace=True
                )
                / 2
            )
            * 2
        ).astype(np.int32)
    ]  # a trick to generate slots along the prop direction
    pad_centers = np.stack([centers_x, centers_y], axis=-1)
    # pad_centers = np.array(list(product(slot_centers_x, slot_centers_y)))[np.random.choice(total_slots, size=n_sampled_slots, replace=False)]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_4x4():
    N = 4
    wl = 1.55
    index_si = 3.48
    size = (16, 4)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        # wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        wg_width=(0.8, 0.8),
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (11, 0.7)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_4x4_L():
    N = 4
    wl = 1.55
    index_si = 3.48
    size = (31.5, 6.1)
    port_len = 3
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        # wg_width=(wl / index_si / 2, wl / index_si / 2),  # in/out wavelength width, um
        wg_width=(1.0, 1.0),
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (20, 1)
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_4x4_L_random():
    N = 4
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_4x4_L_random_3pads():
    N = 4
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(20, 30), np.random.uniform(5.5, 7))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    mmi3 = MMI_NxM(
        3,
        3,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi3.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_5x5_L_random():
    N = 5
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(25, 30), np.random.uniform(7, 9))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_6x6_L_random():
    N = 6
    wl = 1.55
    index_si = 3.48
    # size = (13.5, 3.5)
    # size = (25.9, 6.1)
    size = (np.random.uniform(40, 50), np.random.uniform(10, 14))
    port_len = 3
    wg_width = np.random.uniform(0.8, 1.1)
    mmi = MMI_NxM(
        N,
        N,
        box_size=size,  # box [length, width], um
        wg_width=(wg_width, wg_width),  # in/out wavelength width, um
        port_diff=(size[1] / N, size[1] / N),  # distance between in/out waveguides. um
        port_len=port_len,  # length of in/out waveguide from PML to box. um
        border_width=0.25,  # space between box and PML. um
        grid_step=0.05,  # isotropic grid step um
        NPML=(30, 30),  # PML pixel width. pixel
    )
    w, h = (
        size[0] * np.random.uniform(0.7, 0.9),
        size[1] / N * np.random.uniform(0.4, 0.65),
    )
    pad_centers = [(0, y) for _, y in mmi.in_port_centers]
    pad_regions = [(x - w / 2, x + w / 2, y - h / 2, y + h / 2) for x, y in pad_centers]
    mmi.set_pad_region(pad_regions)
    return mmi


def mmi_6x6(
    *args,
    **kwargs,
):
    return MMI_NxM(6, 6, *args, **kwargs)


def mmi_8x8(
    *args,
    **kwargs,
):
    return MMI_NxM(8, 8, *args, **kwargs)

if __name__ == "__main__":
    device = mmi_3x3_L_random(random_seed=0, port_idx=1)
    device.run_sim(filepath="test_mmi3x3_2.h5", export_video=True)
    device.dump_config("test_mmi3x3_2.yml")
