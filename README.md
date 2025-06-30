# U2-Net Background Removal

## Overview

A simplified version of [U2-Net](https://github.com/xuebinqin/U-2-Net) for background removal, removing unnecessary components while maintaining core functionality.

## Installation

```bash
conda create -n remove-bg python=3.8
conda activate remove-bg
pip install -r setting/requirements_minimal.txt
```

## How to Run

```bash
python run.py
```

### Input/Output

- **Input**: Images in `test_data/test_images/` folder
- **Output**: Results saved as `[filename]_whitebg.png` in `test_data/u2net_results/` folder

## Modifications

- PyTorch warning message resolved (`weights_only=False`)
- Deprecated function replaced (`F.upsample` → `F.interpolate`)
- matplotlib dependency added
- Background removal and conversion to white background

## Usage

1. Place test images in `test_data/test_images/` folder
2. Run `python run.py`
3. Check results in `test_data/u2net_results/` folder
