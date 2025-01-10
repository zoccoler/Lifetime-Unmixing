# Lifetime Unmixing 

This repository hosts code for distinguishing 2 dyes with overlapping spectra via lifetime.
The workflows presented here mainly rely on the [napari-flim-phasor-plotter](https://zenodo.org/records/13319544) plugin, along with some post-processing/analysis steps done using [pyclesperanto-prototype](https://zenodo.org/records/10432619), [scikit-image](https://scikit-image.org/) and [scikit-learn](https://scikit-learn.org/stable/index.html), plus file saving using [tifffile](https://pypi.org/project/tifffile/).
Images of worm embryos are 6D (3D FLIM timelapse multichannel). Images of adult worms are 5D (3D FLIM multichannel).

## Installation

Install [Miniforge](https://github.com/conda-forge/miniforge?tab=readme-ov-file#miniforge) in your computer (which contains [`mamba`](https://mamba.readthedocs.io/en/latest/installation/mamba-installation.html#)).

Clone this repository locally (for example using [Github Desktop](https://desktop.github.com/)).

From a command line prompt, navigate to where you cloned this repository locally (for example by typing `cd Lifetime-Unmixing`) and create a specific conda environment by typing the following line:

```bash
mamba env create -f env.yml
```

## Usage

From a terminal, activate the conda environment with:

```bash
mamba activate lifetime-env
```

Then open [jupyter lab](https://jupyter.org/) or your prefered IDE (for example VSCode, Pycharm, etc) and run the code of interest from the "code" folder. Remember to replace the data path with your local path to the images.

We recommend running this code on a powerful workstation, preferably with > 48GB RAM and a GPU.

## Acknowledgements

We thank the Bio-Image Analysis Technology Development team of Physics of Life at TU Dresden (BiA-PoL) for the provision of infrastructure, code development and fruitful discussions.

