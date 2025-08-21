# MOPE: Mixture of Optimal Pruned Experts

算法验证参考模型：FLAME-MoE-115M-459M

## FLAME-MoE :fire:​: A Transparent End-to-End Research Platform for Mixture-of-Experts Language Models

**FLAME-MoE** is a transparent, end-to-end research platform for Mixture-of-Experts (MoE) language models. It is designed to facilitate scalable training, evaluation, and experimentation with MoE architectures. [arXiv](https://www.arxiv.org/abs/2505.20225)

### 🔗 Model Checkpoints

Explore our publicly released checkpoints on Hugging Face:

* [FLAME-MoE-1.7B-10.3B](https://huggingface.co/CMU-FLAME/FLAME-MoE-1.7B-10.3B)
* [FLAME-MoE-721M-3.8B](https://huggingface.co/CMU-FLAME/FLAME-MoE-721M-3.8B)
* [FLAME-MoE-419M-2.2B](https://huggingface.co/CMU-FLAME/FLAME-MoE-419M-2.2B)
* [FLAME-MoE-290M-1.3B](https://huggingface.co/CMU-FLAME/FLAME-MoE-290M-1.3B)
* [FLAME-MoE-115M-459M](https://huggingface.co/CMU-FLAME/FLAME-MoE-115M-459M)
* [FLAME-MoE-98M-349M](https://huggingface.co/CMU-FLAME/FLAME-MoE-98M-349M)
* [FLAME-MoE-38M-100M](https://huggingface.co/CMU-FLAME/FLAME-MoE-38M-100M)

---

### 🚀 Getting Started

#### 1. Clone the Repository

Ensure you clone the repository **recursively** to include all submodules:

```bash
git clone --recursive https://github.com/cmu-flame/MoE-Research
cd MoE-Research
```
同时下载 clone apex, Megatron-LM, TransformerEngine 到对应文件夹。

#### 2. Set Up the Environment

Set up the Conda environment using the provided script:

安装好cuda12.4 https://developer.nvidia.com/cuda-12-4-1-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local
官网安装cudnn https://developer.nvidia.com/cudnn-downloads?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=deb_network

```bash
sudo apt-get install build-essential.
# scripts/miscellaneous/install.sh里的安装分开一步步执行就好
# sbatch scripts/miscellaneous/install.sh
```


> **Note:** This assumes you're using a SLURM-managed cluster. Adapt accordingly if running locally.

---

### 📚 Data Preparation

#### 3. Download and Tokenize the Dataset

从此下载数据：https://huggingface.co/datasets/mlfoundations/dclm-baseline-1.0/tree/main/global-shard_03_of_10/local-shard_1_of_10
切词使用
```bash
bash local_tokenize_pythia.sh
```

---

### 🧠 Training

#### 4. Train FLAME-MoE Models

下载weight
```python
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="CMU-FLAME/FLAME-MoE-38M-100M",
    repo_type="model",
    local_dir="./weight_initial",  # 本地保存路径
    local_dir_use_symlinks=False  # 避免软链接，确保是完整文件
)
```

```bash
bash scripts/release/mymoe-38m-froze.sh
```

---

### 📈 Evaluation

#### 5. Evaluate the Model

To evaluate a trained model, set the appropriate job ID and iteration number before submitting the evaluation script:

```bash
#export JOBID=...    # Replace with your training job ID
#export ITER=...     # Replace with the iteration to evaluate (e.g., 11029)
sbatch scripts/myeval.sh
```
