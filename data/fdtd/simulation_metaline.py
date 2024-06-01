import os
from metaline import metaline_3x3_small_random
from multiprocessing import Pool


def generate_one_random_metaline(args):
    idx, N, total_samples = args
    dirname = "./raw"
    file_prefix = f"metaline_{N}x{N}"
    
    for port in range(N):
        print("Generating Metaline_3x3_small_random (port={}): {}/{}".format(port, idx, total_samples))
        metaline = metaline_3x3_small_random(random_seed=idx, port_idx=port, n_layers=2)
        file_name = os.path.join(dirname, file_prefix + f"-{idx:04d}-p{port}")
        metaline.run_sim(filepath=file_name + ".h5", export_video=True)
        metaline.dump_config(filepath=file_name + ".yml")
    
        
if __name__ == "__main__":
    total_samples = 1
    N = 3
    tasks = [
        [idx, N, total_samples]
        for idx in range(total_samples)
    ]
    print(f"pid: {os.getpid()} ppid: {os.getppid()}")
    with Pool(8) as pool:
        pool.map(generate_one_random_metaline, tasks)
