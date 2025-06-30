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

1. Download u2net.pth from [here](https://drive.google.com/file/d/1ao1ovG1Qtx4b7EoskHXmi2E9rp5CHLcZ/view) (from [original U-2-Net repo](https://github.com/xuebinqin/U-2-Net?tab=readme-ov-file)'s "Usage for salient object detection" section) and place it in `saved_models/` folder
2. Place test images in `test_data/test_images/` folder
3. Run `python run.py`
4. Check results in `test_data/u2net_results/` folder
