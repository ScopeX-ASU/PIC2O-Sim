import os
from itertools import product
from typing import Optional, Tuple
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams["text.usetex"] = False
from IPython.display import Video
import h5py
import meep as mp
import numpy as np
import torch
from angler import Optimization, Simulation
from angler.structures import get_grid
from pyutils.compute import gen_gaussian_filter2d_cpu
from pyutils.general import ensure_dir

from device import Device

eps_sio2 = 1.44**2
eps_si = 3.48**2
eps_si3n4 = 2.45**2

__all__ = ["MMI_NxM", "mmi_2x2", "mmi_3x3", "mmi_4x4", "mmi_6x6", "mmi_8x8"]


class MRR(Device):
    def __init__(
        self,
        bus_wg_widths: Tuple[float, float] = [
            0.8,
            0.8,
        ],  # through bus width and drop bus width, um
        ring_wg_width: float = 0.8,  # in/out wavelength width, um
        coupling_gaps: Tuple[float, float] = (
            0.1,
            0.1,
        ),  # through to ring and drop to ring. um
        radius: float = 5,  # radius of the MRR, average between inner and outer circle. um
        bus_lens: Tuple[float, float] = [30, 30],  # length of through and drop bus. um
        eps_r: float = eps_si,  # relative refractive index
        eps_bg: float = eps_sio2,  # background refractive index
    ):

        device_cfg = dict(
            bus_wg_widths=bus_wg_widths,
            ring_wg_width=ring_wg_width,
            coupling_gaps=coupling_gaps,
            radius=radius,
            bus_lens=bus_lens,
            eps_r=eps_r,
            eps_bg=eps_bg,
        )
        super().__init__(**device_cfg)
        self.pad_regions = []

        self.update_device_config("MRR", device_cfg)

        self.size = [
            max(bus_lens),
            2 * radius
            + ring_wg_width
            + bus_wg_widths[0]
            + bus_wg_widths[1]
            + coupling_gaps[0]
            + coupling_gaps[1],
        ]

        # meep definition
        ring_outer = mp.Cylinder(
            radius=radius + 1 / 2 * ring_wg_width, material=mp.Medium(epsilon=eps_r)
        )
        ring_inner = mp.Cylinder(
            radius=radius - 1 / 2 * ring_wg_width, material=mp.Medium(epsilon=eps_bg)
        )
        wg_ycenter_1 = -(
            radius + 1 / 2 * ring_wg_width + coupling_gaps[0] + bus_wg_widths[0] / 2
        )
        through_bus = mp.Block(
            mp.Vector3(mp.inf, bus_wg_widths[0], mp.inf),
            center=mp.Vector3(y=wg_ycenter_1),
            material=mp.Medium(epsilon=eps_r),
        )
        wg_ycenter_2 = (
            radius + 1 / 2 * ring_wg_width + coupling_gaps[1] + bus_wg_widths[1] / 2
        )
        drop_bus = mp.Block(
            mp.Vector3(mp.inf, bus_wg_widths[1], mp.inf),
            center=mp.Vector3(y=wg_ycenter_2),
            material=mp.Medium(epsilon=eps_r),
        )
        self.geometry = [ring_outer, ring_inner, through_bus, drop_bus]

        # self.design_region = apply_regions(
        #     [box], self.xs, self.ys, eps_r_list=1, eps_bg=0
        # )
        self.pad_regions = None

        self.in_port_centers = [
            [-bus_lens[0] / 2 * 0.98, wg_ycenter_1],
            [-bus_lens[1] / 2 * 0.98, wg_ycenter_2],
        ]  # centers

        self.out_port_centers = [
            [bus_lens[0] / 2 * 0.98, wg_ycenter_1],
            [bus_lens[1] / 2 * 0.98, wg_ycenter_2],
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
        src_size = (0, 1.5 * self.bus_wg_widths[in_port_idx], 0)
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
        record_interval: float = 0.3,  # timeunits
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
            geometry=self.geometry,
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
                output[field] = self.trim_pml(res, PML, np.array(data))
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
                mp.Vector3(xh - xl, yh - yl, mp.inf),
                center=mp.Vector3((xh + xl) / 2, (yh + yl) / 2),
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
        str = f"Add-drop MRR("
        str += f"radius = {self.radius} um, ring_wg_width={self.ring_wg_width} um, wg_width(through, drop) = {self.bus_wg_widths} um, wg_gap(through, drop) = {self.coupling_gaps} um, bus_lens(through, drop) = {self.bus_lens} um"
        return str


def addrop_mrr_random(random_seed=0, port_idx=0):
    np.random.seed(random_seed)
    num_rounds = 2
    radius = np.random.uniform(5, 15)
    bus_wg_widths = (np.random.uniform(0.5, 0.8), np.random.uniform(0.5, 0.8))
    coupling_gaps = (np.random.uniform(0.1, 0.15), np.random.uniform(0.1, 0.15))
    bus_lens = [radius * 2 + 2, radius * 2 + 2]
    ring_wg_width = np.random.uniform(0.5, 0.8)

    ## 200 timeunit can propagate one round for 5um MRR
    ## scale up timeunit with radius and number of round trips
    mrr = MRR(
        bus_wg_widths=bus_wg_widths,
        ring_wg_width=ring_wg_width,
        coupling_gaps=coupling_gaps,
        radius=radius,
        bus_lens=bus_lens,
        eps_r=eps_si3n4,
        eps_bg=1,
    )

    mrr.add_source(port_idx, wl_width = 0.1)
    mrr.create_simulation(
        # resolution=15,
        resolution=20,
        stop_when_decay=False,
        until=num_rounds * int(200*radius/5),
        border_width=[0, 1],
        PML=[2, 2],
    )

    return mrr


if __name__ == "__main__":
    mrr = addrop_mrr_random()
    print(mrr)
    mrr.run_sim(filepath="test_mrr" + ".h5", export_video=False)
