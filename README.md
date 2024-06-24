# PIC2O-Sim

**PIC2O** is a physics-inspired CNN-based neural operator that serves as a surrogate for the FDTD simulation in solving Maxwell's equations.

## Prerequisites

To run the code, please install the open-source FDTD simulator [MEEP](https://github.com/NanoComp/meep) and [pyutils](https://github.com/JeremieMelo/pyutility).

## Data Generation

To run the code, first, you need to generate data for three different types of photonic devices:

1. Change directory to the data folder for FDTD simulations:
    ```sh
    cd ./data/fdtd
    ```

2. Run the following scripts to generate data for each device:
    ```sh
    python simulation_mmi.py
    python simulation_mrr.py
    python simulation_metaline.py
    ```

## Training

After generating the dataset, the training could be launched using the following command:
```sh
python ./scripts/fdtd/cnn/train_PICCO.py
