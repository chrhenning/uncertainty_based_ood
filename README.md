# Uncertainty-based OOD detection

Exploring the link between uncertainty estimates obtained by "exact" Bayesian inference and out-of-distribution (OOD) detection. [Here](https://slideslive.at/38962915/are-bayesian-neural-networks-intrinsically-good-at-outofdistribution-detection), you can find a talk motivating the project.

## Prerequisits

The code in this repository requires the installation of the [hypnettorch](https://github.com/chrhenning/hypnettorch) package.

## Experiments and example usage

The subfolder [notebooks](notebooks) jupyter notebook to reproduce experiments from our papers, but they also show how to use the code in this repo. Further usage examples can be found in the subfolder [tutorials](tutorials).

## Neural network Gaussian process

The folder [nngp](nngp) contains utilities to work with [NNGP kernels](https://arxiv.org/abs/1711.00165).

## Documentation

Documentation can be found in folder [docs](docs). Using [sphinx](https://www.sphinx-doc.org/en/master/usage/quickstart.html), the documentation can be compiled within this folder by executing ``make html``. The compiled documentation can be opened via the file [index.html](docs/html/index.html).

## Citation

When using this package in your research project, please consider citing one of our papers for which this package has been developed.

```
@misc{dangelo:henning:2022:uncertainty:based:ood,
      title={On out-of-distribution detection with Bayesian neural networks}, 
      author={Francesco D'Angelo and Christian Henning},
      year={2021},
      eprint={2110.06020},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```

```
@inproceedings{henning:dangelo:2021:bayesian:ood,
title={Are Bayesian neural networks intrinsically good at out-of-distribution detection?},
author={Christian Henning and Francesco D'Angelo and Benjamin F. Grewe},
booktitle={ICML Workshop on Uncertainty and Robustness in Deep Learning},
year={2021},
url={https://arxiv.org/abs/2107.12248}
}
```
