<div align="center">
<h1 align="center">Matrix-Game 3.0</h1>
<h3 align="center">Real-Time and Streaming Interactive World Model with Long-Horizon Memory</h3>
</div>

<font size=7><div align='center' >  [[🤗 HuggingFace](https://huggingface.co/Skywork/Matrix-Game-3.0)] [[📖 Technical Report](assets/pdf/report.pdf)] [[🚀 Project Website](https://matrix-game-v3.github.io/)] </div></font>


https://github.com/user-attachments/assets/5b95bb21-bc77-4bb5-bc2b-7b12de2d3f21

## 📝 Overview
**Matrix-Game-3.0** is an open-sourced, memory-augmented interactive world model designed for 720p real-time long-form video generation.
- **Upgraded Data Engine**: Combines Unreal Engine-based synthetic data, large-scale automated AAA game data, and real-world video augmentation to generate high-quality Video–Pose–Action–Prompt data. 
- **Long-horizon Memory & Consistency**: Uses prediction residuals and frame re-injection for self-correction, while camera-aware memory ensures long-term spatiotemporal consistency. 
- **Real-Time Interactivity & Open Access**: It employs a multi-segment autoregressive distillation strategy based on Distribution Matching Distillation (DMD), combined with model quantization and VAE decoder distillation to support [40fps] real-time generation at 720p resolution with a 5B model, while maintaining stable memory consistency over minute-long sequence.
- **Scale Up 28B-MoE Model**: Scaling up to a 2×14B model further improves generation quality, dynamics, and generalization. 

## 🤗 Matrix-Game-3.0 Model
We provide two pretrained 5B model weights, including the base model and the distilled model, for first-person generation in unreal scenes. These resources are available on our HuggingFace page. 

In addition, the model trained on a combination of unreal and real-world data, as well as the 28B large model, will be released soon! 🚀🚀

## Requirements
It supports one gpu or multi-gpu inference. We tested this repo on the following setup:
* A/H series GPUs are tested.
* Linux operating system.
* 64 GB RAM.

## ⚙️ Quick Start
### Installation
Create a conda environment and install dependencies:
```
conda create -n matrix-game-3.0 python=3.12 -y
conda activate matrix-game-3.0
# install FlashAttention
# Our project also depends on [FlashAttention](https://github.com/Dao-AILab/flash-attention)
git clone https://github.com/SkyworkAI/Matrix-Game-3.0.git
cd Matrix-Game-3.0
pip install -r requirements.txt
```

### Model Download
```
pip install "huggingface_hub[cli]"
huggingface-cli download Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```
If you are downloading the model on a server, you can also use:
```bash
hf download Skywork/Matrix-Game-3.0 --local-dir Matrix-Game-3.0
```
### Inference
Before running inference, you need to prepare:
- Input image
- Text prompt

After downloading pretrained models, you can use the following command to generate an interactive video with random actions:
``` sh
torchrun --nproc_per_node=$NUM_GPUS generate.py --size 704*1280 --dit_fsdp --t5_fsdp --ckpt_dir Matrix-Game-3.0 --fa_version 3 --use_int8 --num_iterations 12 --num_inference_steps 3 --image demo_images/001/image.png --prompt "A colorful, animated cityscape with a gas station and various buildings." --save_name test --seed 42 --compile_vae --lightvae_pruning_rate 0.5 --vae_type mg_lightvae --output_dir ./output
# "num_iterations" refers to the number of iterations you want to generate. The total number of frames generated is given by:57 + (num_iterations - 1) * 40 
```
Tips: 
If you want to use the base model, you can use `--use_base_model --num_inference_steps 50`. To run with your own input actions, use `--interactive`.
For LightVAE, use `--vae_type mg_lightvae` with `--lightvae_pruning_rate 0.5`, or `--vae_type mg_lightvae_v2` with `--lightvae_pruning_rate 0.75`. `mg_lightvae_v2` is faster than `mg_lightvae` while keeping quality close to the latter.
With multiple GPUs, you can pass `--use_async_vae --async_vae_warmup_iters 1` to speed up inference (see [`test.sh`](test.sh)).

### Replaying actions from a file (`--actions_file`)

Instead of random actions (non-interactive) or typing actions by hand (`--interactive`),
you can replay a fixed action sequence from a JSON file:

```sh
torchrun --nproc_per_node=$NUM_GPUS generate.py --size 704*1280 --ckpt_dir Matrix-Game-3.0 \
  --num_iterations 3 --num_inference_steps 3 --image demo_images/001/image.png \
  --prompt "A colorful, animated cityscape." --save_name replay --output_dir ./output \
  --actions_file my_actions.json
```

- When `--actions_file` is set, random action generation and manual `input()` are both
  disabled and the file's actions are used everywhere (overlay, async VAE, final video).
- It works with **and** without `--interactive`; in interactive mode it never waits for input.
- All ranks read the same file deterministically, so multi-GPU runs use identical actions.
- The total frame count must be `57 + (num_iterations - 1) * 40`.

Two JSON schemas are supported.

**Schema A — Tensor format** (frame-level, validated and consumed directly):

```json
{
  "keyboard_condition": [[0,0,1,0,0,0], [0,0,1,0,0,0]],
  "mouse_condition":    [[0.0,-0.1],    [0.0,-0.1]]
}
```

`keyboard_condition` must be `[T, 6]`, `mouse_condition` must be `[T, 2]`, with the same
`T`, and `T == 57 + (num_iterations - 1) * 40`.

**Schema B — Clip format** (token-level, expanded to frames on load):

The simplest form has one clip entry per diffusion iteration (default granularity):

```json
{
  "clips": [
    {"mouse": "u", "keyboard": "w"},
    {"mouse": "j", "keyboard": "a"},
    {"mouse": "u", "keyboard": "d"}
  ]
}
```

With no `frames` field and no `unit_frames`, there must be exactly `num_iterations` clips:
clip 0 expands to 57 frames and each later clip to 40 frames (≈ 1–1.4 s each at 40 fps).

**Finer-grained control with `unit_frames`**

For 0.5 s granularity (20 frames at 40 fps), add a top-level `"unit_frames": 20` field.
Each entry then covers 20 frames; the last entry absorbs the remainder.
Number of clips required = `ceil(total_frames / unit_frames)`.

```json
{
  "unit_frames": 20,
  "clips": [
    {"mouse": "u", "keyboard": "w"},
    {"mouse": "u", "keyboard": "w"},
    {"mouse": "j", "keyboard": "a"},
    {"mouse": "j", "keyboard": "a"},
    {"mouse": "u", "keyboard": "d"}
  ]
}
```

For `num_iterations=2` (total = 97 frames), `unit_frames=20` requires
`ceil(97/20) = 5` clips (four × 20 frames + one × 17 frames).
You can also pass the granularity on the command line instead of embedding it in the file:

```sh
... --actions_file my_actions.json --control_unit_frames 20
```

Common values at 40 fps: `20` → 0.5 s, `10` → 0.25 s, `40` → 1 s.
A `unit_frames` key inside the file takes precedence over `--control_unit_frames`.

**Explicit per-clip frames** (full control):

```json
{
  "clips": [
    {"mouse": "u", "keyboard": "w", "frames": 57},
    {"mouse": "j", "keyboard": "a", "frames": 40},
    {"mouse": "u", "keyboard": "d", "frames": 40}
  ]
}
```

Frame counts must sum exactly to `57 + (num_iterations - 1) * 40`.
Cannot be combined with a top-level `unit_frames` field.

**Supported tokens** (identical to interactive mode):

- Mouse — `i` (up), `k` (down), `j` (left), `l` (right), `u` (no move).
- Keyboard — `w` (forward), `s` (back), `a` (left), `d` (right), `q` (no movement).

A clip that omits `mouse`/`keyboard` defaults to the no-op token (`u`/`q`). Invalid files
(missing file, bad JSON, wrong schema, wrong shapes, frame/clip-count mismatch, unknown
tokens) raise a clear error before the model is loaded.

## ⭐ Acknowledgements
- [Diffusers](https://github.com/huggingface/diffusers) for their excellent diffusion model framework
- [Self-Forcing](https://github.com/guandeh17/Self-Forcing) for their excellent work
- [GameFactory](https://github.com/KwaiVGI/GameFactory) for their idea of action control module
- [LightX2V](https://github.com/ModelTC/lightx2v) for their excellent quantization framework
- [Wan2.2](https://github.com/Wan-Video/Wan2.2) for their strong base model
- [lingbot-world](https://github.com/Robbyant/lingbot-world) for their context parallel framework 
## 📜 License
This project is licensed under the Apache License, Version 2.0 — see [LICENSE.txt](LICENSE.txt).

## 📖 Citation
If you find this work useful for your research, please kindly cite our paper:

```
  @misc{2026matrix,
    title={Matrix-Game 3.0: Real-Time and Streaming Interactive World Model with Long-Horizon Memory},
    author={{Skywork AI Matrix-Game Team}},
    year={2026},
    howpublished={Technical report},
    url={https://github.com/SkyworkAI/Matrix-Game/blob/main/Matrix-Game-3/assets/pdf/report.pdf}
  }
```
