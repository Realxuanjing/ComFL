# import torch
# import torch.nn as nn
# import torch.optim as optim

# # 假设这是全局模型 (gNB_model)
# class GlobalModel(nn.Module):
#     def __init__(self):
#         super(GlobalModel, self).__init__()
#         self.layer1 = nn.Linear(10, 20)
#         self.layer2 = nn.Linear(20, 30)
#         self.layer3 = nn.Linear(30, 40)  # 假设layer3被剪枝
#         self.layer4 = nn.Linear(40, 50)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer3(x)
#         x = self.layer4(x)
#         return x

# # 假设这是剪枝后的模型 (sum_module)
# class PrunedModel(nn.Module):
#     def __init__(self):
#         super(PrunedModel, self).__init__()
#         self.layer1 = nn.Linear(10, 20)
#         self.layer2 = nn.Linear(20, 30)
#         # layer3 被剪枝了，不再包含
#         self.layer4 = nn.Linear(30, 50)

#     def forward(self, x):
#         x = self.layer1(x)
#         x = self.layer2(x)
#         x = self.layer4(x)
#         return x

# # 初始化两个模型
# gNB_model = GlobalModel()
# sum_module = PrunedModel()

# # 打印两个模型的参数
# # print("gNB_model:")
# # print(gNB_model.state_dict())
# # print("\nsum_module:")
# # print(sum_module.state_dict())

# # 合并两个模型的参数：加权平均已存在的层，保留 gNB_model 的原始参数
# gNB_model_state_dict = gNB_model.state_dict()
# sum_module_state_dict = sum_module.state_dict()
# for name, param_gNB in gNB_model_state_dict.items():
#     print('name', name)
# for name, param_sum in sum_module_state_dict.items():
#     print('name2', name)
# # 遍历gNB_model中的每个参数
# for name, param_gNB in gNB_model_state_dict.items():
#     # print('name', name)
#     if name in sum_module_state_dict:
#         param_sum = sum_module_state_dict[name]
        
#         # 检查两者的形状是否一致，确保可以进行平均
#         if param_gNB.shape == param_sum.shape:
#             # 对共有的层进行平均
#             print(f"Updating layer {name}: average of {param_gNB.shape}")
#             param_gNB.data = (param_gNB.data + param_sum.data) / 2
#         else:
#             print(f"Layer {name} has mismatched shapes, keeping gNB_model's parameter.")
#     else:
#         print(f"Layer {name} is not in sum_module, keeping gNB_model's parameter.")

# # 更新 gNB_model 的参数
# gNB_model.load_state_dict(gNB_model_state_dict)

# # 打印合并后的全局模型参数
# # print("\nUpdated gNB_model:")
# print(gNB_model.state_dict())
# import random
# clients_num = 10
# print(range(0,clients_num))
# print(min(3, clients_num))
# print(random.sample(range(0,clients_num),min(3, clients_num)))

# # 安装Flower库: pip install flwr
# import flwr as fl
# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.utils.data import DataLoader, random_split, Dataset
# from torchvision import datasets, transforms
# device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

# class SimpleNet(nn.Module):
#     def __init__(self):
#         super(SimpleNet, self).__init__()
#         self.layer = nn.Sequential(
#             nn.Flatten(),
#             nn.Linear(28*28, 128),
#             nn.ReLU(),
#             nn.Linear(128, 10)
#         )

#     def forward(self, x):
#         return self.layer(x)

# class Client:
#     def __init__(self, train_data, test_data):
#         self.train_data = train_data
#         self.test_data = test_data
#         self.model = SimpleNet()

#     def train(self, epochs=1):
#         self.model.train()
#         criterion = nn.CrossEntropyLoss()
#         optimizer = optim.SGD(self.model.parameters(), lr=0.01)
#         train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)

#         for epoch in range(epochs):
#             for data, target in train_loader:
#                 data, target = data.to(device), target.to(device)
#                 optimizer.zero_grad()
#                 output = self.model(data)
#                 loss = criterion(output, target)
#                 loss.backward()
#                 optimizer.step()

#     def evaluate(self):
#         self.model.eval()
#         correct = 0
#         test_loader = DataLoader(self.test_data, batch_size=32)
#         with torch.no_grad():
#             for data, target in test_loader:
#                 data, target = data.to(device), target.to(device)
#                 output = self.model(data)
#                 pred = output.argmax(dim=1, keepdim=True)
#                 correct += pred.eq(target.view_as(pred)).sum().item()

#         accuracy = correct / len(self.test_data)
#         return accuracy

# def create_noniid_data(trainset, num_clients):
#     data_per_client = len(trainset) // num_clients
#     clients_data = []
#     idxs = list(range(len(trainset)))
#     for i in range(num_clients):
#         data_idxs = idxs[i * data_per_client:(i + 1) * data_per_client]
#         clients_data.append(torch.utils.data.Subset(trainset, data_idxs))
#     return clients_data

# transform = transforms.Compose([transforms.ToTensor()])
# trainset = datasets.CIFAR10('/home/data1/xxx/dataset/COMFL/datasets/CRF_10', train=True, download=True, transform=transform)
# testset = datasets.CIFAR10('/home/data1/xxx/dataset/COMFL/datasets/CRF_10', train=False, download=True, transform=transform)

# clients_data = create_noniid_data(trainset, num_clients=3)

# clients = [Client(clients_data[i], testset) for i in range(3)]

# def client_fn(cid: str):
#     return clients[int(cid)]

# strategy = fl.server.strategy.FedAvg()

# fl.server.start_server(config={"num_rounds": 5}, strategy=strategy, client_manager=fl.server.SimpleClientManager())

# for i in range(3):
#     fl.client.start_numpy_client(client_fn=str(i))

# print("Done!")


import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import torch.optim as optim
import torch.nn.functional as F

# 定义一个复杂的CNN模型
class ComplexCNN(nn.Module):
    def __init__(self, num_classes):
        super(ComplexCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)  
        self.conv5 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)  
        
        self.bn1 = nn.BatchNorm2d(64)
        self.bn2 = nn.BatchNorm2d(128)
        self.bn3 = nn.BatchNorm2d(256)
        self.bn4 = nn.BatchNorm2d(512)
        self.bn5 = nn.BatchNorm2d(512)
        
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        
        self.dropout = nn.Dropout(p=0.2)  
        
        self.classifier = nn.Sequential(
            nn.Linear(512 * 1 * 1 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  
        
        x = x.view(-1, 512 * 1 * 1)  # 扁平化
        
        x = self.classifier(x)  # 全连接层
        return x

import torch
import torch.nn.utils.prune as prune

# 结构化剪枝示例：剪掉conv1的某些通道
# def prune_conv_layer(module, pruning_fraction):
#     # 计算每个通道的L1范数（即每个通道的绝对权重总和）
#     weight = module.weight.data.abs().sum(dim=(1, 2, 3))  # 计算每个通道的L1范数
#     num_channels = weight.size(0)
#     threshold = torch.quantile(weight, pruning_fraction)  # 计算阈值
    
#     # 找到低于阈值的通道
#     mask = weight > threshold
#     keep_channels = mask.nonzero().squeeze()  # 保留的通道索引
    
#     # 截取新的权重矩阵，仅保留重要通道
#     new_weight = module.weight.data[keep_channels, :, :, :]
#     module.out_channels = len(keep_channels)
#     module.weight.data = new_weight
    
#     return module


def prune_conv_layer(module, pruning_fraction):
    """
    对卷积层进行结构化剪枝，按L1范数剪掉最不重要的通道
    :param module: 卷积层（nn.Conv2d）
    :param pruning_fraction: 剪枝比例，表示保留的通道比例
    :return: 剪枝后的卷积层
    """
    # 计算每个通道的L1范数（即每个通道的绝对权重总和）
    weight = module.weight.data.abs().sum(dim=(1, 2, 3))  # 计算每个通道的L1范数
    num_channels = weight.size(0)
    
    # 计算阈值，按L1范数剪掉最不重要的通道
    threshold = torch.quantile(weight, pruning_fraction)  # 计算阈值
    
    # 找到低于阈值的通道，返回一个布尔值mask
    mask = weight > threshold
    keep_channels = mask.nonzero().squeeze()  # 保留的通道索引
    
    # 截取新的权重矩阵，仅保留重要通道
    new_weight = module.weight.data[keep_channels, :, :, :]
    
    # 更新模块的输出通道数
    module.out_channels = len(keep_channels)
    
    # 更新卷积层的权重
    module.weight.data = new_weight
    
    # 如果卷积层有偏置，更新偏置
    if module.bias is not None:
        new_bias = module.bias.data[keep_channels]
        module.bias.data = new_bias
    
    return module

def print_param_count(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f'Total parameters: {total_params}')
def print_layer_info(model):
    print(f"{'Layer Name':<30} {'Parameters Count'}")
    print('-' * 50)
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            params_count = sum(p.numel() for p in module.parameters())
            print(f"{name:<30} {params_count}")    
# 对每一层进行剪枝
def prune_model_structure(model, pruning_fraction=0.2):
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            module = prune_conv_layer(module, pruning_fraction)
    return model
def print_shapes_hook(module, input, output):
    # 打印层的名称，输入的尺寸和输出的尺寸
    print(f"Layer: {module.__class__.__name__}")
    print(f"Input shape: {input[0].shape}")
    print(f"Output shape: {output.shape}")
    print('-' * 50)
# 使用结构化剪枝
model = ComplexCNN(num_classes=10)
print("\nBefore pruning:")
print_param_count(model)
print_layer_info(model)
for name, layer in model.named_modules():
    # 只注册卷积层和线性层的钩子
    if isinstance(layer, (nn.Conv2d, nn.Linear)):
        layer.register_forward_hook(print_shapes_hook)
dummy_input = torch.randn(48, 3, 32, 32)  # 假设输入是一个 224x224 的 RGB 图像
output = model(dummy_input)


# model = prune_model_structure(model, pruning_fraction=0.2)
# # 进行结构化剪枝
# # prune.random_structured(model.conv1, name="weight", amount=0.2, dim=0)  # 例如剪枝conv1的20%

# # 检查剪枝后的层和偏置
# for name, module in model.named_modules():
#     if isinstance(module, nn.Conv2d):
#         print(f"{name} - Weight shape: {module.weight.shape}, Bias shape: {module.bias.shape}")

#         # 如果剪枝后输出通道数改变，调整偏置
#         new_out_channels = module.weight.shape[0]  # 新的输出通道数
#         module.bias = nn.Parameter(torch.zeros(new_out_channels))  # 调整偏置
# # 打印剪枝后的模型参数数量
# print("\nAfter pruning (structured pruning):")
# print_param_count(model)
# print_layer_info(model)
# # 为每一层注册钩子函数


# # 创建一个假输入，并进行前向传播，查看每一层的输入输出维度
# # dummy_input = torch.randn(48, 3, 32, 32)  # 假设输入是一个 224x224 的 RGB 图像
# output = model(dummy_input)