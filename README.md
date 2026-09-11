# FSP-Net

面向**跨域小样本高光谱图像分类**的 PyTorch 研究代码。项目以 Chikusei 为源域，提供 Indian Pines、Pavia University 和 Houston 三个目标域的训练与评估入口，将空谱特征、类别文本语义和源域原型先验结合用于分类。

本仓库保留模型、训练、配置及必要的数据处理代码。数据集、BERT 权重、训练日志、缓存和生成的分类图需在本地准备或生成，不随源码发布。以下说明对应当前代码，不提供未经验证的精度或论文复现承诺。

## 方法概览

1. **域内波段映射**：源域与目标域分别经过 `1 × 1` 卷积和 BatchNorm，将不同数量的光谱波段映射到共同维度，默认 100 维。
2. **空谱双分支编码**：空间分支和光谱分支使用三维卷积与残差结构提取特征，按 `0.5 / 0.5` 融合，默认输出 128 维嵌入。
3. **类别语义对齐**：冻结的 `bert-base-uncased` 为英文类别名称生成 768 维 `[CLS]` 特征，再经可训练投影映射到视觉嵌入空间，使用跨模态对比损失进行对齐。
4. **跨域原型融合**：根据目标域与源域原型的余弦相似度，经过可学习温度的 Softmax 计算源域先验；先验投影后，由门控网络与目标域原型融合。代码中将该部分标记为 `DG-CPF`。
5. **距离分类**：查询样本与类别原型的负平方欧氏距离作为分类分数。源域使用原始原型，目标域使用融合后的原型。

当前训练目标为：

```text
L = L_CE_source + L_CE_target + 2 × (L_align_source + L_align_target)
```

`utils/loss_function.py` 中仍保留 `SupConLoss`，但当前三个训练入口均未调用它。目标域训练数据仍使用辐射噪声增强。

## 目录结构

```text
FSP-Net/
├── model/
│   ├── encoder.py              # 空谱编码器、语义投影、原型投影和融合门控
│   └── mapping.py              # 源域/目标域波段映射
├── config/
│   ├── Indian_pines.py
│   ├── PaviaU.py
│   └── houston.py
├── utils/
│   ├── dataloader.py           # 小样本任务采样及目标域划分
│   ├── data_augment.py         # 辐射噪声增强
│   ├── loss_function.py        # 距离函数及对比损失
│   ├── chikusei_imdb_128.py    # Chikusei 原始数据预处理
│   └── utils.py                # 数据读取、随机种子、日志及分类图
├── train.py                   # Indian Pines
├── train-up.py                # Pavia University
├── train-ht.py                # Houston
├── requirements.txt
└── README.md
```

## 环境准备

建议使用独立的 Python 3.10 环境和 NVIDIA GPU。训练入口使用 CUDA 设备索引，当前没有 CPU 训练适配。

```bash
git clone https://github.com/AIYAU/FSP-Net.git
cd FSP-Net
python -m venv .venv
```

激活环境：

```bash
# Linux / WSL
source .venv/bin/activate
```

```powershell
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
```

先按 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/)安装与显卡驱动及 CUDA 运行环境匹配的 `torch` 和 `torchvision`，然后安装其余依赖：

```bash
python -m pip install -r requirements.txt
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

主要依赖包括 PyTorch、TorchVision、Transformers、NumPy、SciPy、scikit-learn、Matplotlib、TensorBoard 和 hdf5storage。`hdf5storage` 用于 Chikusei 原始 MAT 文件预处理。依赖清单不代表已完成所有版本组合的训练验证。

## 数据准备

下载并按照各数据集的使用条款准备数据。原项目 README 提供了[数据下载地址](https://pan.baidu.com/s/1erafsAzPYCdomqBziJpy-w?pwd=ddf1)（提取码 `ddf1`）；此链接的可用性及其中是否包含 PaviaU 未经本仓库验证。

仅需准备公共源域数据和计划运行的目标域数据：

```text
datasets/
├── Chikusei_imdb_128_7_7.pickle
├── Chikusei_raw_mat/                         # 仅自行预处理时需要
│   ├── HyperspecVNIR_Chikusei_20140729.mat
│   └── HyperspecVNIR_Chikusei_20140729_Ground_Truth.mat
├── IP/
│   ├── indian_pines_corrected.mat
│   └── indian_pines_gt.mat
├── PaviaU/
│   ├── PaviaU.mat
│   └── PaviaU_gt.mat
└── Houston/
    ├── data.mat
    ├── mask_train.mat
    └── mask_test.mat
```

目标域 MAT 文件中的数据变量名需与文件名（不含 `.mat`）一致，例如 `PaviaU.mat` 中的变量应为 `PaviaU`。图像形状为 `[height, width, bands]`，标签为二维数组；`0` 为背景，地物类别按 `1 ... C` 编号。

| 目标域 | 训练入口 | 配置 | 波段数 | 类别数 | 训练/测试划分 |
| --- | --- | --- | ---: | ---: | --- |
| Indian Pines | `train.py` | `config/Indian_pines.py` | 200 | 16 | 每类随机抽取少量标注样本，余下标注像元用于测试 |
| Pavia University | `train-up.py` | `config/PaviaU.py` | 103 | 9 | 每类随机抽取少量标注样本，余下标注像元用于测试 |
| Houston | `train-ht.py` | `config/houston.py` | 144 | 15 | 从训练掩码采样，使用测试掩码中的标注像元评估 |

Chikusei 的预处理 pickle 应包含 `data` 和 `Labels`：`data` 形状为 `[N, 7, 7, 128]`，`Labels` 为长度 `N` 的类别标签。训练时按类别分组，并通过 `sanity_check` 保留样本数不少于 200 的类别，每类取 200 个样本。

若使用原始 Chikusei MAT 文件，将其放到上述目录后，在仓库根目录执行：

```bash
python utils/chikusei_imdb_128.py
```

脚本默认生成 `datasets/Chikusei_imdb_128_7_7.pickle`，对应配置中的 `source_data`。预处理会将大量图像块载入内存，请根据可用内存安排运行。

**类别语义对应关系需要核对**：训练脚本中的 `labels_src` 当前包含 18 个名称。更换或自行预处理源域数据时，需确保筛选后的类别索引与这些名称一一对应。目标域的 `labels_tar` 也在各训练脚本中定义，切换数据集不能只替换配置路径。

## 准备 BERT 权重

训练脚本从本地 `pretrain-model/bert-base-uncased/` 读取模型和 tokenizer。可从 [Hugging Face 的 bert-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) 下载并保存：

```bash
python -c "from transformers import BertModel, BertTokenizer; p='pretrain-model/bert-base-uncased'; BertTokenizer.from_pretrained('google-bert/bert-base-uncased').save_pretrained(p); BertModel.from_pretrained('google-bert/bert-base-uncased').save_pretrained(p)"
```

首次下载需要网络；离线运行时，将完整模型目录复制到对应位置。目录至少应包含模型配置、受当前 Transformers 版本支持的权重文件和 tokenizer 文件，例如：

```text
pretrain-model/bert-base-uncased/
├── config.json
├── pytorch_model.bin            # 或受当前版本支持的 model.safetensors
├── tokenizer_config.json
└── vocab.txt
```

## 训练与评估

在仓库根目录执行，按目标数据集选择对应命令：

```bash
# Indian Pines
python train.py --config config/Indian_pines.py

# Pavia University
python train-up.py --config config/PaviaU.py

# Houston
python train-ht.py --config config/houston.py
```

主要参数在对应的 `config/*.py` 中修改：

| 参数 | 默认值 | 含义 |
| --- | ---: | --- |
| `gpu` | `0` | CUDA 设备索引 |
| `patch_size` | `7` | 空间图像块边长，需与源数据预处理一致 |
| `episode` | `5000` | 每次实验的训练 episode 数 |
| `lr` / `weight_decay` | `1e-3` / `1e-4` | SGD 学习率与权重衰减 |
| `n_dim` / `d_emb` | `100` / `128` | 波段映射维度与嵌入维度 |
| `shot_num_per_class` | `1` | 每个训练 episode 的每类支持样本数 |
| `query_num_per_class` | `19` | 每个训练 episode 的每类查询样本数 |
| `tar_lsample_num_per_class` | `5` | 每类实际使用的目标域标注样本数 |

`shot_num_per_class=1` 与 `tar_lsample_num_per_class=5` 含义不同：每类先抽取 5 个真实标注样本并生成增强样本，再在训练 episode 中采样 1 个支持样本及 19 个查询样本。当前语义张量与对比损失按每类 1 个支持样本实现，请保持 `shot_num_per_class=1`；直接增大该值可能造成索引或维度错误。

训练任务的类别数实际使用 `tar_class_num`。配置中的 `num_ways`、`num_shots` 未用于决定当前训练循环的 way/shot，修改它们不会改变上述采样。

三个入口默认各运行 10 个随机种子，每个种子训练 5000 个 episode；在第 1 个 episode 及之后每 500 个 episode 评估一次。如需缩短试运行，可减少配置中的 `episode`，并在训练脚本中调整 `nDataSet`（不超过 `seeds` 长度）。

## 输出与当前局限

- `logs/<实验设置>/`：文本日志，包括 OA、AA、Kappa、各类别精度及随机种子统计。
- `runs/`：TensorBoard 事件，可用 `tensorboard --logdir runs` 查看训练曲线。
- `classificationMap/`：自动创建目录并生成 `IP_5shot.png`、`UP_5shot.png` 或 `houston_5shot.png` 等分类图，文件名随标注样本数变化。

当前脚本在多次评估中按**测试集最高精度**选择每个种子的结果，没有单独验证集；正式对比实验需明确这一选择方式。推理还使用当前 episode 的源域原型，尚未提供独立推理入口、模型权重保存或断点续训。分类图来自训练脚本最后保留的预测结果，不代表所有随机种子结果的集成。

## 代码来源说明

项目原 README 将代码标记为 `CDFS-CASCL-2024`，并关联题为 “Cross-Domain Few-shot Hyperspectral Image Classification with Cross-Modal Alignment and Supervised Contrastive Learning” 的工作。当前版本在此基础上包含跨域原型融合等改动；上述名称用于保留原说明中的来源信息，不表示当前实现与原论文完全一致。
