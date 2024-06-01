'''
Date: 2023-12-09 01:50:50
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-25 02:43:25
FilePath: /NeurOLight_Local/data/fdtd/simulation_mmi.py
'''
import os
import meep as mp
import matplotlib.pyplot as plt
import numpy as np
from mmi import mmi_3x3_L_random, mmi_3x3_L_determined, mmi_3x3_L_swp_res
from multiprocessing import Pool
from pyutils.general import TimerCtx, print_stat, TorchTracemalloc

def generate_one_random_mmi(args):
    idx, N, total_samples = args
    dirname = "./raw"
    file_prefix = f"mmi_{N}x{N}_L_random"
    
    for port in range(N):
        print("Generating MMI_3x3_L_random (port={}): {}/{}".format(port, idx, total_samples))
        mmi = mmi_3x3_L_random(random_seed=idx, port_idx=port)
        file_name = os.path.join(dirname, file_prefix + f"-{idx:04d}-p{port}")
        mmi.run_sim(filepath=file_name + ".h5", export_video=True)
        # used to test the time for meep simulation
        # with TimerCtx() as t:
        #     mmi.run_sim(filepath=file_name + ".h5", export_video=False)
        # with open(file_name + ".txt", "w") as f:
        #     f.write(f"Time: {t.interval}")
        mmi.dump_config(filepath=file_name + ".yml")
    
        
if __name__ == "__main__":
    total_samples = 1
    N = 3
    tasks = [
        [idx, N, total_samples]
        for idx in range(total_samples)
    ]
    print(f"pid: {os.getpid()} ppid: {os.getppid()}")
    with Pool(8) as pool:
        pool.map(generate_one_random_mmi, tasks)
