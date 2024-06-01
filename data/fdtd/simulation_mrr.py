'''
Date: 2023-12-09 01:50:50
LastEditors: Jiaqi Gu && jiaqigu@asu.edu
LastEditTime: 2023-12-25 02:40:33
FilePath: /NeurOLight_Local/data/fdtd/simulation_mrr.py
'''
import os
from mrr import addrop_mrr_random
from multiprocessing import Pool


def generate_one_random_mrr(args):
    idx, total_samples = args
    dirname = "./raw"
    file_prefix = f"mrr_random"
    
    for port in range(1): # only port 0 is enough
        print("Generating MRR_random (port={}): {}/{}".format(port, idx, total_samples))
        mrr = addrop_mrr_random(random_seed=idx, port_idx=port)
        file_name = os.path.join(dirname, file_prefix + f"-{idx:04d}-p{port}")
        mrr.run_sim(filepath=file_name + ".h5", export_video=True)
        mrr.dump_config(filepath=file_name + ".yml")
    
        
if __name__ == "__main__":
    total_samples = 1
    tasks = [
        [idx, total_samples]
        for idx in range(total_samples)
    ]
    print(f"pid: {os.getpid()} ppid: {os.getppid()}")
    with Pool(8) as pool:
        pool.map(generate_one_random_mrr, tasks)
