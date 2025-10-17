# wheel_legged_amp

AMP implementation with minimal changes on legged_gym and rsl_rl

**Code reference: [AMP_for_hardware](https://github.com/Alescontrela/AMP_for_hardware)**

## Setup

Clone the code

```bash
https://github.com/XinLang2019/wheel_legged_amp.git
cd wheel_legged_amp
```

Download isaacgym
1. Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended):
    - `conda create -n amp-wl python=3.8`
    - `conda activate amp-wl`
2. Install pytorch:
    - `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117`
3. Install Isaac Gym
    - Download and install Isaac Gym Preview 3 (Preview 2 will not work!) from https://developer.nvidia.com/isaac-gym
    - `cd isaacgym/python && pip install -e .`
4. Install rsl_rl
    - `cd rsl_rl && pip install -e .`

5. Install this repo
    - `pip install -e .`
## Usage

Train and play

1. train:
```python legged_gym/scripts/train.py --task=a1_amp --headless``
    - To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible).
    - To run headless (no rendering) add `--headless`.
    - **Important**: To improve performance, once the training starts press `v` to stop the rendering. You can then enable it later to check the progress.

2. play:
 Play a trained policy:  
```python legged_gym/scripts/play.py --task=a1_amp```
    - By default the loaded policy is the last model of the last run of the experiment folder.
    - Other runs/model iteration can be selected by setting `load_run` and `checkpoint` in the train config.


