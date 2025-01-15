## 项目简介

本项目是一个基于联邦学习的深度学习模型训练框架，主要包含以下几种卷积神经网络（CNN）模型的实现和训练：

- **SimpleCNN**：三层卷积神经网络模型
- **ComplexCNN**：五层卷积神经网络模型
- **ComplexMore**：十层卷积神经网络模型
- **VGGnet**：经典VGG网络模型

此外，项目还支持模型剪枝和模型聚合等功能，以优化模型性能，提升训练效率。

### 剪枝与聚合

项目支持在子用户训练时进行剪枝，并在剪枝后进行模型聚合。您可以通过修改 `client_pruned_ratio` 变量来控制每个子用户的剪枝比例，从而达到定制化的剪枝效果。

- **Channel Pruning（通道剪枝）**：基于通道的重要性对卷积神经网络进行剪枝。
- **Layer Pruning（卷积层剪枝）**：根据卷积层的影响力和重要性，进行层级剪枝。

这些剪枝方法旨在减少冗余参数，提高模型的推理效率，并在保证准确度的前提下减少计算资源的消耗。

## 目录结构

```
.
├── ComplexCNN.py            # 复杂卷积神经网络模型定义
├── GetDataSet.py            # 数据集获取和预处理
├── main.py                  # 旧程序主入口
├── model_merge.py           # 模型合并相关代码
├── model.py                 # 模型定义和训练相关代码
├── output_*.txt             # 输出日志文件
├── picture_test.py          # 图片测试相关代码
├── Pruning.py               # 模型剪枝相关代码
├── Readme.md                # 项目说明文件
├── SimpleCNN.py             # 简单卷积神经网络模型定义
├── test.py                  # 测试代码
├── train_iid/               # 独立同分布训练相关文件
│   ├── Flavg.py             # 不剪枝
│   ├── Flavg_channel_pruning.py # 按照通道剪枝
│   ├── Flavg_pruning.py     # 按照层剪枝
│   ├── FedPer.py
│   ├── FedPer_channel_pruning.py
│   ├── FedPer_pruning.py
│   ├── FedPer_Tweak.py
│   ├── FedPer_Tweak_pruning.py
│   ├── initial_FL.py
│   ├── pFedPara.py
│   ├── temp_train.py
├── train_noiid/             # 非独立同分布训练相关文件
│   ├── Flavg_channel_pruning.py
│   ├── FedPer_channel_pruning.py
├── train.sh                 # 训练脚本
├── UE_train.py              # 用户设备训练代码
├── UE_train_copy.py         # 用户设备训练副本
└── .gitignore               # Git忽略文件
```

## 安装依赖

请确保已安装以下依赖：

- Python 3.8
- PyTorch
- torchvision
- tqdm
- Pillow
- matplotlib


## 使用方法

### 数据集准备

1. 将数据集放置在指定目录下。
2. 在 `GetDataSet.py` 文件中配置数据集路径。请修改以下几个变量：
    - `save_path`：保存模型的路径。
    - `dataset_path`：数据集所在的路径。
    - `top_model_name`：修改保存文件夹的名称。

### 训练模型

1. 进入相应的训练目录（`train_iid` 或 `train_noiid`）。
2. 运行以下命令开始训练模型：

```bash
cd train_iid    # 如果使用独立同分布训练
# 或
cd train_noiid  # 如果使用非独立同分布训练

python Flavg.py   # 运行具体训练脚本，例如 Flavg, FedPer, FedPer_pruning 等
```

### 训练脚本

您可以使用 `train.sh` 脚本进行批量训练，脚本中已经配置了多种训练模式，您只需修改相应的模型和训练参数即可。

```bash
bash train.sh   # 运行批量训练脚本
```

## 注意事项

- 请确保数据集路径和保存路径已正确配置。
- 根据需求选择适当的训练模式和模型。
- 训练时可根据不同场景使用不同的训练脚本，譬如 `Flavg`, `FedPer`, 或使用模型剪枝相关的训练脚本。
