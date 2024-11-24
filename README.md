# SpikeRL

**SpikeRL: A Highly Scalable Framework for Spiking Reinforcement Learning**

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Installation](#installation)
- [Requirements](#requirements)
<!-- - [Usage](#usage)
- [Examples](#examples)
- [Contributing](#contributing)
- [License](#license) -->
- [Contact](#contact)

## Introduction

SpikeRL is a highly scalable framework designed for implementing and experimenting with spiking reinforcement learning algorithms. Leveraging the power of spiking neural networks, SpikeRL aims to bridge the gap between traditional reinforcement learning and biologically plausible neural computations, offering researchers and developers a robust platform for advanced AI experiments.

## Features

- **Scalability:** Efficiently handles large-scale spiking neural networks.
- **Efficiency:** Utilizes mixed-precision techniques for optimization and energy efficiency.
- **Extensibility:** Easily extendable to accommodate new models and environments.

## Installation

Follow the steps below to set up SpikeRL on your local machine.

### Prerequisites

- Python 3.7 or higher
- pip package manager

### Steps

1. **Clone the Repository**

   ```bash
   git clone https://github.com/tokey-tahmid/spike-rl.git
   cd spike-rl
   ```

2. **Create a Virtual Environment (Optional but Recommended)**

   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**

   For mpi support with spikeRL, compile PyTorch from source (instructions at https://github.com/pytorch/pytorch) with either `openmpi` or `mpich` installation in your system. Ensure you have the required version of `numpy` and `mujoco` as specified in requirements.txt.

   ```bash
   pip install -r requirements.txt
   ```

## Requirements

The project dependencies are listed in the `requirements.txt` file:

```
wheel
setuptools
pybind11
ipython
notebook
pyglet
six
pillow
imageio
scipy
scikit-learn
pandas
matplotlib
gymnasium
networkx
torch
gym
mpi4py
mujoco==2.3.3
numpy==1.24.0
```

## Usage

```bash
# Using NCCL backend
torchrun --nproc_per_node=8 main.py --env Ant-v4 --hidden_sizes 100 100 --batch_size 100 --epochs 1 --backend nccl

# Using MPI backend
mpiexec -n 8 python main.py --env Ant-v4 --hidden_sizes 100 100 --batch_size 100 --epochs 1 --backend mpi

```

<!-- ## Examples

Detailed examples and tutorials can be found in the [examples](examples/) directory. These include:

- **Basic Training:** Step-by-step guide to training a spiking agent in a simple environment.
- **Advanced Architectures:** Implementing custom spiking neural network architectures.
- **Visualization:** Tools and scripts for visualizing network activity and learning metrics. -->

<!-- ## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. **Fork the Repository**

2. **Create a New Branch**

   ```bash
   git checkout -b feature/YourFeature
   ```

3. **Commit Your Changes**

   ```bash
   git commit -m "Add your feature"
   ```

4. **Push to the Branch**

   ```bash
   git push origin feature/YourFeature
   ```

5. **Open a Pull Request**

Please ensure your code follows the project's coding standards and includes appropriate tests.

## License

This project is licensed under the [MIT License](LICENSE). -->

## Contact

For any questions or support, please contact [ttahmid@vols.utk.edu](mailto:ttahmid@vols.utk.edu).

---

**Acknowledgements**

This work was supported by the U.S. Department of Energy, Office of Science,
ASCR under Award Number DE-SC0021419.
This material is based upon work supported by the Assistant Secretary of Defense
for Research and Engineering under Air Force Contract No. FA8702-15-D-0001.
Any opinions, findings, conclusions or recommendations expressed in this material
are those of the author(s) and do not necessarily reflect the views of the Assistant
Secretary of Defense for Research and Engineering.
