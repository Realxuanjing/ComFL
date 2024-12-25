import copy
import torch
import torch.nn as nn

def layer_pruning(model, prune_ratio):
    """按层剪枝"""
    pruned_model = copy.deepcopy(model)
    
    # 获取所有卷积层
    conv_layers = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 计算层的重要性（使用参数的L1范数）
            importance = torch.sum(torch.abs(module.weight.data))
            conv_layers.append((name, module, importance))
    
    # 按重要性排序
    conv_layers.sort(key=lambda x: x[2])
    
    # 确定要剪枝的层数
    num_layers = len(conv_layers)
    num_pruned = int(num_layers * prune_ratio)
    
    # 保存剪枝的层名称
    pruned_layers = []
    
    # 剪枝最不重要的层（将其替换为身份映射）
    for i in range(num_pruned):
        name, module, _ = conv_layers[i]
        # 将卷积层替换为1x1卷积，模拟身份映射
        identity_conv = nn.Conv2d(
            module.in_channels,
            module.out_channels,
            kernel_size=1,
            stride=module.stride,
            padding=0,
            bias=False
        )
        # 初始化为近似单位矩阵
        identity_conv.weight.data.zero_()
        for c in range(min(module.in_channels, module.out_channels)):
            # 遍历输入通道和输出通道
            identity_conv.weight.data[c, c, 0, 0] = 1.0
            # 这相当于创建了一个单位矩阵的效果，使得该卷积层在前向传播时不会改变输入特征图。
        
        # 将剪枝的层名称添加到列表
        pruned_layers.append(name)
        
        # 替换原始模块
        setattr(pruned_model, name, identity_conv)

    # 返回剪枝后的模型和剪掉的层名称
    return pruned_model, pruned_layers

