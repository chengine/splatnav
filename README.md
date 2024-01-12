# GEMSPLAT: Guassian Splatting with Language-Conditioned Semantic Overlay
## This repo implements Gaussian Splatting with knowledge distillation from the vision-language model CLIP to extract semantic information of the scene.

## Installation Instructions

## The installation instructions assumes you have installed [Nerfstudio](https://docs.nerf.studio/quickstart/installation.html) from source.

### 1. Clone this repo.
`git clone git@github.com:shorinwa/gemsplat.git`

### 2. Install `gemsplat` as a python package.
`python -m pip install -e .`

## 3. Register `gemsplat` with Nerfstudio.
`ns-install -cli`

### Now, you can run `gemsplat` like other models in Nerfstudio using the `ns-train gemsplat` command.