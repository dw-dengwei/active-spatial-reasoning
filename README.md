<div align="center">

# Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models

[![arXiv](https://img.shields.io/badge/arXiv-2606.02459-b31b1b.svg)](https://arxiv.org/abs/2606.02459)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.12.11-3776AB.svg)](https://www.python.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.4-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6-EE4C2C.svg)](https://pytorch.org/)
[![Venue](https://img.shields.io/badge/ICML-2026-68499b.svg)](https://icml.cc/virtual/2026/poster/61450)

[**Wei Deng**](https://dw-dengwe.cn)<sup>1,2</sup>, **Xianlin Zhang**<sup>2,3</sup>, [**Mengshi Qi**](https://jueduilingdu.github.io/)<sup>\* 1,2</sup>

<sup>1</sup> State Key Laboratory of Networking and Switching Technology<br>
<sup>2</sup> Beijing University of Posts and Telecommunications, China<br>
<sup>3</sup> School of Digital Media & Design Arts<br>

</div>

---

## News

- **[2026.06]** We release the RL training and inference code.
- **[2026.04]** Our paper is accepted to **ICML 2026**.

---

## Overview

We propose an **agentic pipeline for spatial reasoning** in Vision-Language Models (VLMs), Our approach introduces three key contributions:
1. We propose a novel agentic spatial reasoning pipeline
within VLMs that maintains a dynamic cognitive map, a
updatable memory parameterizing the spatial layout.
2. We introduce the new Spatial Assertion Code (SAC)
that collaborates with the dynamic cognitive map to verify
the correctness of intermediate spatial reasoning, providing
dense reward signals for reinforcement learning.
3. Our model achieves state-of-the-art performance on the
MindCube dataset, surpassing the best
existing work by a relative improvement of 7.0%, and by
29.5 accuracy score (a relative improvement of 53.2%) on
the challenging ROTATION subset.

---

## Environment Setup

### Requirements

- **CUDA 12.4**
- **uv**

### Install `uv`

We use `uv` for dependency management. If you don't have it installed:

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

For other installation methods (Windows, package managers, etc.), see the [official documentation](https://docs.astral.sh/uv/getting-started/installation/).

### Install Dependencies

```bash
git clone https://github.com/dw-dengwei/active-spatial-reasoning.git
cd active-spatial-reasoning

# Install core dependencies + PyTorch 2.6 (CUDA 12.4)
uv sync --extra build

# Additionally install flash-attention 2.7.4 (CUDA 12 + PyTorch 2.6)
uv sync --extra build --extra compile
```

The `--extra build` flag installs PyTorch 2.6 from the `cu124` index. The `--extra compile` flag installs a pre-built Flash Attention 2.7.4 wheel for CUDA 12 + PyTorch 2.6.

---

## TODO List

- [x] Release the inference code
- [x] Release the RL training code
- [ ] Release the SFT code
- [ ] Release model checkpoints on Huggingface
- [ ] Release the training and inference data on Huggingface

---

## Training

To train the model with GRPO on 8 GPUs:

```bash
bash train.sh
```

Logging is handled by [SwanLab](https://swanlab.cn/). Set `trainer.logger` to also include `tensorboard` or `wandb` if desired.

---

## Inference & Evaluation

To run inference on the MindCube tinybench with a trained checkpoint, first edit `CHECKPOINT_PATH` in `inference.sh`, then run:

```bash
bash inference.sh
```
---

## Acknowledgments

This project builds upon the following open-source works:

- [verl](https://github.com/verl-project/verl) — RL framework for LLMs
- [MindCube](https://github.com/mll-lab-nu/MindCube) — Spatial reasoning benchmark
- [verl-agent](https://github.com/langfengQ/verl-agent) — Multi-turn agent training framework
- [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) — Efficient SFT framework for LLMs

---

## Citation

If you find this work useful, please cite:

```bibtex
@inproceedings{deng2026active,
  title     = {Active Exploring like a Pigeon: Reinforcing Spatial Reasoning via Agentic Vision-Language Models},
  author    = {Deng, Wei and Zhang, Xianlin and Qi, Mengshi},
  booktitle = {International Conference on Machine Learning (ICML)},
  year      = {2026},
}
```
---

## License

This project is licensed under the [Apache License 2.0](LICENSE).
