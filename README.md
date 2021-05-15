[![TorchFI Logo](https://github.com/bfgoldstein/tiny_torchfi/blob/master/docs/img/torchfi-logo.png)](https://github.com/bfgoldstein/tiny_torchfi)

--------------------------------------------------------------------------------
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://github.com/bfgoldstein/tiny_torchfi/blob/master/LICENSE)

Tiny version of [TorchFI](https://github.com/bfgoldstein/tiny_torchfi/).

- [Installation](#installation)
  - [Clone Project](#clone-project)
  - [Install Dependencies](#install-dependencies)
- [Citation](#citation)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Clone Project

  ```bash
    git clone git@github.com:bfgoldstein/tiny_torchfi.git
  ```

### Install Dependencies

We highly recommend installing an [Anaconda](https://www.continuum.io/downloads) environment.

  ```bash
  conda env create -f torchfi.yml
  conda activate torchfi
  cd ${PROJECT_PATH}
  export PYTHON_PATH=$PYTHON_PATH:${PROJECT_PATH}
  ```

## Citation

Please cite [Goldstein'20](https://doi.org/10.1109/LASCAS45839.2020.9069026) and [Goldstein'21](https://doi.org/10.1109/ISQED51717.2021.9424287) in your publications if it helps your research:

```
@INPROCEEDINGS{goldstein20,
  Author = {Goldstein, Brunno F. and Srinivasan, Sudarshan and Das, Dipankar and Banerjee, Kunal and Santiago, Leandro and Ferreira, Victor C. and Nery, Alexandre S. and Kundu, Sandip and França, Felipe M. G.},
  Booktitle={2020 IEEE 11th Latin American Symposium on Circuits   Systems (LASCAS)},
  Title = {Reliability Evaluation of Compressed Deep Learning Models}
  Year = {2020},
  Keywords={resilience, soft error, transient fault, neural network, deep learning},
  pages={1-5},
  doi = {10.1109/LASCAS45839.2020.9069026},
  url = {https://doi.org/10.1109/LASCAS45839.2020.9069026}
}

@INPROCEEDINGS{goldstein21,
  author={Goldstein, Brunno F. and Ferreira, Victor C. and Srinivasan, Sudarshan and Das, Dipankar and Nery, Alexandre S. and Kundu, Sandip and França, Felipe M. G.},
  booktitle={2021 22nd International Symposium on Quality Electronic Design (ISQED)}, 
  title={A Lightweight Error-Resiliency Mechanism for Deep Neural Networks}, 
  year={2021},
  volume={},
  number={},
  pages={311-316},
  doi={10.1109/ISQED51717.2021.9424287}
}

```

## License

Tiny TorchFI code is released under the [Apache license 2.0](https://github.com/bfgoldstein/tiny_torchfi/blob/master/LICENSE).

## Acknowledgments

- [PyTorch](https://github.com/pytorch/pytorch) - Python package for fast tensors computation and DNNs execution
