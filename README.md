# Lifetime Unmixing 

This repository hosts code for distinguishing 2 dyes with overlapping spectra via lifetime.
Images are 6D (3D FLIM timelapse multichannel).

## Installation

Install Miniconda or Mambaforge in your computer.

Clone this repository locally (we recommend using [Github Desktop](https://desktop.github.com/)).

From a command line prompt, navigate to where you cloned this repository locally (for example by typing `cd Lifetime-Unmixing`) and create this specific conda environment by typing the following line:

```bash
conda env create -f env.yml
```

## Usage

From a terminal, activate the conda environment with:

```bash
conda activate lifetime-env
```

Then open jupyter lab or your prefered IDE (for example VSCode) and run the code of interest from the "code" folder. Remember to replace the data path with your local path to the images.