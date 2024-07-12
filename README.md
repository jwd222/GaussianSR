# GaussianSR: High Fidelity 2D Gaussian Splatting for Arbitrary-Scale Image Super-Resolution
[![](https://img.shields.io/badge/Dataset-🔰DIV2K-blue.svg)](https://data.vision.ee.ethz.ch/cvl/DIV2K/) [![](https://img.shields.io/badge/Dataset-🔰Set5-blue.svg)](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) [![](https://img.shields.io/badge/Dataset-🔰BSD100-blue.svg)](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) [![](https://img.shields.io/badge/Dataset-🔰Urban100-blue.svg)](https://cv.snu.ac.kr/research/EDSR/benchmark.tar) [![](https://img.shields.io/badge/Dataset-🔰General100-blue.svg)](https://drive.google.com/drive/folders/1satiNXA73tksZjormVquIlsdbhMVdq2M?usp=drive_link) [![](https://img.shields.io/badge/Dataset-🔰Manga109-blue.svg)](https://drive.google.com/drive/folders/1k2GriT5t3wh9T4Hi4J1OELL83dsM4Z_o?usp=drive_link)
***
>**Abstract**: _Implicit neural representations (INRs) have revolutionized arbitrary-scale super-resolution (ASSR) by modeling images as continuous functions. However, existing INR-based ASSR methods discretely store latent codes, neglecting the continuous nature of image intensity variations and lacking interpretability of latent representations. This paper proposes a novel and elegant ASSR paradigm called GaussianSR that addresses these limitations through 2D Gaussian Splatting (2DGS). Instead of treating pixels as discrete points, GaussianSR models each pixel as a continuous Gaussian field. A classifier is trained to assign learnable Gaussian kernels with adaptive variances and opacities to each pixel, accommodating diverse input characteristics. By applying 2DGS to the encoder features, they are reorganized into a continuous field capturing inherent intensity variations. GaussianSR achieves flexible, adaptive receptive fields through mutually stacked Gaussian kernels, effectively capturing multi-scale features and long-range dependencies. Extensive experiments demonstrate that GaussianSR yields superior performance with reduced parameters, underscoring the great capability of our pipeline._
>
![image](https://github.com/tljxyys/GaussianSR/blob/main/fig/Figure_2.png)
***
## 1. Environment
- Python 3
- Pytorch 1.6.0
- TensorboardX
- yaml, numpy, tqdm, imageio, einops

## 2. Running the code

**0. Preliminaries**

- For `train.py` or `test.py`, use `--gpu [GPU]` to specify the GPUs (e.g. `--gpu 0` or `--gpu 0,1`).

- For `train.py`, by default, the save folder is at `save/_[CONFIG_NAME]`. We can use `--name` to specify a name if needed.

- For dataset args in configs, `cache: in_memory` denotes pre-loading into memory (may require large memory, e.g. ~40GB for DIV2K), `cache: bin` denotes creating binary files (in a sibling folder) for the first time, `cache: none` denotes direct loading. We can modify it according to the hardware resources before running the training scripts.

**1. DIV2K experiments**

**Train**: `python train_gaussian.py --config configs/train/train_edsr-baseline-gaussian-batch16.yaml`. We use 4 V100 GPUs for training EDSR-baseline-LIIF.

**Test**: `bash scripts/test-div2k.sh [MODEL_PATH] [GPU]` for div2k validation set, `bash scripts/test-benchmark.sh [MODEL_PATH] [GPU]` for benchmark datasets. `[MODEL_PATH]` is the path to a `.pth` file, we use `epoch-last.pth` in corresponding save folder.
