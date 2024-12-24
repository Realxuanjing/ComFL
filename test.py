import torch
import torch.nn as nn
import torch.optim as optim

# 假设这是全局模型 (gNB_model)
class GlobalModel(nn.Module):
    def __init__(self):
        super(GlobalModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        self.layer3 = nn.Linear(30, 40)  # 假设layer3被剪枝
        self.layer4 = nn.Linear(40, 50)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        return x

# 假设这是剪枝后的模型 (sum_module)
class PrunedModel(nn.Module):
    def __init__(self):
        super(PrunedModel, self).__init__()
        self.layer1 = nn.Linear(10, 20)
        self.layer2 = nn.Linear(20, 30)
        # layer3 被剪枝了，不再包含
        self.layer4 = nn.Linear(30, 50)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer4(x)
        return x

# 初始化两个模型
gNB_model = GlobalModel()
sum_module = PrunedModel()

# 打印两个模型的参数
# print("gNB_model:")
# print(gNB_model.state_dict())
# print("\nsum_module:")
# print(sum_module.state_dict())

# 合并两个模型的参数：加权平均已存在的层，保留 gNB_model 的原始参数
gNB_model_state_dict = gNB_model.state_dict()
sum_module_state_dict = sum_module.state_dict()
for name, param_gNB in gNB_model_state_dict.items():
    print('name', name)
for name, param_sum in sum_module_state_dict.items():
    print('name2', name)
# 遍历gNB_model中的每个参数
for name, param_gNB in gNB_model_state_dict.items():
    # print('name', name)
    if name in sum_module_state_dict:
        param_sum = sum_module_state_dict[name]
        
        # 检查两者的形状是否一致，确保可以进行平均
        if param_gNB.shape == param_sum.shape:
            # 对共有的层进行平均
            print(f"Updating layer {name}: average of {param_gNB.shape}")
            param_gNB.data = (param_gNB.data + param_sum.data) / 2
        else:
            print(f"Layer {name} has mismatched shapes, keeping gNB_model's parameter.")
    else:
        print(f"Layer {name} is not in sum_module, keeping gNB_model's parameter.")

# 更新 gNB_model 的参数
gNB_model.load_state_dict(gNB_model_state_dict)

# 打印合并后的全局模型参数
# print("\nUpdated gNB_model:")
print(gNB_model.state_dict())
