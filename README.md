# FSP-Net 消融实验

当前压缩包默认配置为 **DG-CPF + FADM**。三个数据集入口统一读取 `config/ablation.py`，运行时日志会记录 DG-CPF 和 FADM 的启用状态。

| 数据集 | 在解压后的 `FSP-Net` 目录执行 |
|---|---|
| Indian Pines (IP) | `python train.py` |
| University of Pavia (UP) | `python train-up.py` |
| Houston 2013 (HT) | `python train-ht.py` |

运行前请配置支持 CUDA 的 PyTorch 和匹配的 torchvision，以及 NumPy、SciPy、scikit-learn、Matplotlib、Transformers、TensorBoard。数据和 BERT 权重沿用原项目路径。源域使用 `datasets/Chikusei_imdb_128_7_7.pickle`，目标域文件名见 `config/` 下对应的数据集配置，BERT 使用 `pretrain-model/bert-base-uncased/`。压缩包包含源码和配置，数据和权重需另行准备。

- DG-CPF 开启时，使用跨域原型融合进行目标域训练和推理。关闭时，目标域直接使用支持集的类别均值原型。四种配置均使用相同的距离度量和分类损失。
- FADM 开启时，源域和目标域各自使用一套参数独立的 FADM。基础分支沿用 `1×1 Conv → BN`。频域分支在光谱维做 rFFT，保留前 `min(FADM_NUM_MODES, C_in // 2 + 1)` 个频率分量（含直流分量），用 irFFT 恢复输入波段数，再执行 `1×1 Conv → BN → GELU → 1×1 Conv → BN`，与基础分支相加。两个卷积均输出 `n_dim` 通道，原配置为 100。
- 频域分支末端卷积在通用权重初始化后再次置零，使初始输出等于基础分支。图中未给出 `num_modes` 的具体值，本实现默认用 16。如需与已有实验对照，请核对这一参数。
- 四种配置沿用上传源码中的数据划分、随机种子、训练轮数和文本对齐损失。建议将四个压缩包分别解压到不同的工作目录，使日志与分类图各自保留。
