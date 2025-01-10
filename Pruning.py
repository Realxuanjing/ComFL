import copy
import torch
import torch.nn as nn
import math
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

def channel_pruning(model, prune_ratio=0.1):
    pruned_model = copy.deepcopy(model)
    conv_layers = []
    bn_layers = []
    for name, module in pruned_model.named_modules():
        if isinstance(module, nn.Conv2d):
            conv_layers.append((name, module))
    for name,module in pruned_model.named_modules():
        if isinstance(module, nn.BatchNorm2d):
            bn_layers.append((name,module))
    temp_feature = 0
    # for name, module in conv_layers:
    end_indices = []
    for i, (name, module) in enumerate(conv_layers):
        # 计算每个卷积层的通道重要性（L1范数或者其他指标）
        abs_weights = torch.abs(module.weight.data)
        channel_importance = abs_weights.sum(dim=(1, 2, 3))  # 对每个通道计算L1范数
        _, indices = torch.sort(channel_importance, descending=True)  # 按照重要性降序排序
        
        
        num_channels_to_prune = int(prune_ratio * module.out_channels)
        prune_indices = indices[-num_channels_to_prune:]  # 选取最不重要的通道
        
        new_weight = module.weight.data.clone()
        new_weight = new_weight[indices]  
        module.weight.data = new_weight
        if module.bias is not None:
            new_bias = module.bias.data.clone()
            new_bias = new_bias[indices] 
            module.bias.data = new_bias
        
        bn_layer = bn_layers[i]
        bn_layer[1].weight.data = bn_layer[1].weight.data[indices]  
        if bn_layer[1].bias is not None:
            bn_layer[1].bias.data = bn_layer[1].bias.data[indices]
        bn_layer[1].running_mean = bn_layer[1].running_mean[indices]
        bn_layer[1].running_var = bn_layer[1].running_var[indices]

        new_conv_layer = replace_layer(module,name,math.ceil(module.in_channels*(1-prune_ratio)),module.out_channels-num_channels_to_prune)
        new_bn_layer = replace_layer(bn_layer[1],bn_layer[0],math.ceil(module.in_channels*(1-prune_ratio)),module.out_channels-num_channels_to_prune)
        
        setattr(pruned_model, name, new_conv_layer)
        setattr(pruned_model, bn_layer[0], new_bn_layer)
        end_indices = indices
    new_in_features = math.ceil(pruned_model.classifier[0].in_features*(1-prune_ratio) )
    # print(new_in_features)
    # new_linear = nn.Linear(new_in_features, 1024)  
    # setattr(pruned_model.classifier, '0', new_linear)  
    # print(pruned_model.classifier[0].weight.data.shape,len(end_indices))
    # pruned_model.classifier[0].weight.data = pruned_model.classifier[0].weight.data[:,end_indices]
    pruned_model.classifier[0].weight.data = pruned_model.classifier[0].weight.data[:, end_indices]
    # if pruned_model.classifier[0].bias is not None:
    #     pruned_model.classifier[0].bias.data = pruned_model.classifier[0].bias.data[end_indices]
    # if pruned_model.classifier[0].bias is not None:
    #     pruned_model.classifier[0].bias.data = pruned_model.classifier[0].bias.data[end_indices]
    # print(pruned_model.classifier[0].weight.data.shape)
    new_classifier_in_layer = replace_layer(pruned_model.classifier[0],'Linear',new_in_features,1024)
    
    # new_classifier = nn.Sequential(
    #     new_classifier_in_layer,
    #     nn.ReLU(),
    #     nn.Linear(1024, 512),
    #     nn.ReLU(),
    #     nn.Linear(512, 10)
    # )
    # setattr(new_classifier,pruned_model.classifier[0],new_classifier_in_layer)
    setattr(pruned_model.classifier, '0', new_classifier_in_layer)
    # print(pruned_model.classifier[0].weight.data.shape)

    return pruned_model

def replace_layer(layer,layer_name,in_num_features ,out_num_features):
    # layer_list = list(model.named_modules())
    # print('layer_name',layer_name)
    if isinstance(layer, nn.Conv2d):
        if layer_name == 'Conv1':
            # print("conv1")
            with torch.no_grad():
                new_layer = nn.Conv2d(layer.in_channels, out_num_features, layer.kernel_size, layer.stride, layer.padding)
                new_layer.weight.data = layer.weight.data[:out_num_features, :, :, :]
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data[:out_num_features]
        else:
            with torch.no_grad():
                new_layer = nn.Conv2d(in_num_features, out_num_features, layer.kernel_size, layer.stride, layer.padding)
                new_layer.weight.data = layer.weight.data[:out_num_features, :in_num_features, :, :]
                if layer.bias is not None:
                    new_layer.bias.data = layer.bias.data[:out_num_features]
    elif isinstance(layer, nn.BatchNorm2d):
        with torch.no_grad():
            new_layer = nn.BatchNorm2d(out_num_features)
            new_layer.weight.data = layer.weight.data[:out_num_features]
            new_layer.bias.data = layer.bias.data[:out_num_features]
            new_layer.running_mean = layer.running_mean[:out_num_features]
            new_layer.running_var = layer.running_var[:out_num_features]
    elif isinstance(layer, nn.Linear):
        with torch.no_grad():
            new_layer = nn.Linear(in_num_features, layer.out_features)
            # print("Linear",new_layer.weight.shape)
            new_layer.weight.data = layer.weight.data[:, :in_num_features]
            new_layer.bias.data = layer.bias.data
    else:
        print("Unsupported layer type: ", type(layer))
    return new_layer

def prune_by_gradient(model, pruning_fraction=0.2):
    """
    根据梯度的重要性剪枝，保留梯度较大的参数
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    for module, weight in parameters_to_prune:
        weight_tensor = getattr(module, weight)
        weight_grad = weight_tensor.grad  # 获取权重的梯度
        importance = torch.abs(weight_grad)  # 使用梯度的绝对值作为重要性度量
        threshold = torch.quantile(importance, pruning_fraction)  # 设置阈值
        mask = importance > threshold  # 保留梯度较大的部分
        weight_tensor.data = weight_tensor.data * mask

import torch.nn.utils.prune as prune

def prune_by_gradient(model, pruning_fraction=0.2):
    """
    根据梯度的重要性剪枝，保留梯度较大的参数
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    for module, weight in parameters_to_prune:
        weight_tensor = getattr(module, weight)
        weight_grad = weight_tensor.grad  # 获取权重的梯度
        importance = torch.abs(weight_grad)  # 使用梯度的绝对值作为重要性度量
        threshold = torch.quantile(importance, pruning_fraction)  # 设置阈值
        mask = importance > threshold  # 保留梯度较大的部分
        
        # 使用结构化剪枝：将重要性较低的通道剪掉（以通道为单位的剪枝）
        if isinstance(module, torch.nn.Conv2d):
            # 通道剪枝：剪掉重要性较小的卷积核
            num_channels = weight_tensor.shape[0]
            pruning_indices = torch.argsort(importance.view(num_channels))[:int(pruning_fraction * num_channels)]
            mask[pruning_indices] = 0
        
        # 更新权重：剪去不重要的部分
        weight_tensor.data = weight_tensor.data * mask

    return model

def prune_by_activation(model, pruning_fraction=0.2):
    """
    根据激活值的重要性剪枝，保留激活值较大的神经元
    """
    parameters_to_prune = []
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d) or isinstance(module, torch.nn.Linear):
            parameters_to_prune.append((module, 'weight'))

    for module, weight in parameters_to_prune:
        # 计算激活值
        # activations = module(input_tensor)  # 使用模型的输入tensor计算激活
        # importance = activations.abs().mean(dim=(0, 2, 3))  # 对卷积激活进行处理，计算每个通道的平均激活
        # threshold = torch.quantile(importance, pruning_fraction)
        # mask = importance > threshold  # 剪去激活较小的部分
        # weight_tensor = getattr(module, 'weight')
        # weight_tensor.data = weight_tensor.data * mask.unsqueeze(0).unsqueeze(2).unsqueeze(3)
        pass
