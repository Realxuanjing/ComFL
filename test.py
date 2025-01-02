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

# 安装Flower库: pip install flwr
import flwr as fl
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from torchvision import datasets, transforms
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

class SimpleNet(nn.Module):
    def __init__(self):
        super(SimpleNet, self).__init__()
        self.layer = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28*28, 128),
            nn.ReLU(),
            nn.Linear(128, 10)
        )

    def forward(self, x):
        return self.layer(x)

class Client:
    def __init__(self, train_data, test_data):
        self.train_data = train_data
        self.test_data = test_data
        self.model = SimpleNet()

    def train(self, epochs=1):
        self.model.train()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(self.model.parameters(), lr=0.01)
        train_loader = DataLoader(self.train_data, batch_size=32, shuffle=True)

        for epoch in range(epochs):
            for data, target in train_loader:
                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = self.model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

    def evaluate(self):
        self.model.eval()
        correct = 0
        test_loader = DataLoader(self.test_data, batch_size=32)
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = self.model(data)
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        accuracy = correct / len(self.test_data)
        return accuracy

def create_noniid_data(trainset, num_clients):
    data_per_client = len(trainset) // num_clients
    clients_data = []
    idxs = list(range(len(trainset)))
    for i in range(num_clients):
        data_idxs = idxs[i * data_per_client:(i + 1) * data_per_client]
        clients_data.append(torch.utils.data.Subset(trainset, data_idxs))
    return clients_data

transform = transforms.Compose([transforms.ToTensor()])
trainset = datasets.CIFAR10('/home/data1/xxx/dataset/COMFL/datasets/CRF_10', train=True, download=True, transform=transform)
testset = datasets.CIFAR10('/home/data1/xxx/dataset/COMFL/datasets/CRF_10', train=False, download=True, transform=transform)

clients_data = create_noniid_data(trainset, num_clients=3)

clients = [Client(clients_data[i], testset) for i in range(3)]

def client_fn(cid: str):
    return clients[int(cid)]

strategy = fl.server.strategy.FedAvg()

fl.server.start_server(config={"num_rounds": 5}, strategy=strategy, client_manager=fl.server.SimpleClientManager())

for i in range(3):
    fl.client.start_numpy_client(client_fn=str(i))

print("Done!")
