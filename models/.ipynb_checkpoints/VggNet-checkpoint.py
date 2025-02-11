import torch
import torch.nn as nn
from torchvision import models 
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from torchvision.transforms import transforms
import os
import glob
from PIL import Image 
import csv
import random
import numpy as np
from torch.utils.data import random_split
import copy

class GetData(Dataset):
    def __init__(self, root, resize, mode):
        super(GetData, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {'cat': 0, 'dog': 1}                   # "类别名称": 编号,对自己的类别进行定义
        for name in sorted(os.listdir(os.path.join(root))):
            # 判断是否为一个目录
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = self.name2label.get(name)           # 将类别名称转换为对应编号



        # image, label 划分
        self.images, self.labels = self.load_csv('cat_dogs.csv')          # csv文件存在 直接读取
        if mode == 'train':                                             # 对csv中的数据集80%划分为训练集                   
            self.images = self.images[:int(0.8 * len(self.images))]
            self.labels = self.labels[:int(0.8 * len(self.labels))]
        
        else:                                                           # 剩余20%划分为测试集
            self.images = self.images[int(0.8 * len(self.images)):]
            self.labels = self.labels[int(0.8 * len(self.labels)):]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img, label = self.images[idx], self.labels[idx]
        # 这里首先做一个数据预处理，因为VGG16是要求输入224*224*3的
        tf = transforms.Compose([                                               # 常用的数据变换器
					            lambda x:Image.open(x).convert('RGB'),          # string path= > image data 
					                                                            # 这里开始读取了数据的内容了
					            transforms.Resize(                              # 数据预处理部分
					                (int(self.resize * 1.25), int(self.resize * 1.25))), 
					            transforms.RandomRotation(15), 
					            transforms.CenterCrop(self.resize),             # 防止旋转后边界出现黑框部分
					            transforms.ToTensor(),
					            transforms.Normalize(mean=[0.485, 0.456, 0.406],
					                                 std=[0.229, 0.224, 0.225])
       							 ])
        img = tf(img)
        label = torch.tensor(label)                                 # 转化tensor
        return img, label                                           # 返回当前的数据内容和标签
    
    def load_csv(self, filename):
    	# 这个函数主要将data中不同class的图片读入csv文件中并打上对应的label，就是在做数据集处理
        # 没有csv文件的话，新建一个csv文件
        if not os.path.exists(os.path.join(self.root, filename)): 
            images = []
            for name in self.name2label.keys():   					# 将文件夹内所有形式的图片读入images列表
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            random.shuffle(images)									# 随机打乱

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:  # 新建csv文件，进行数据写入
                writer = csv.writer(f)
                for img in images:                                              # './data/class1/spot429.jpg'
                    name = img.split(os.sep)[-2]                                # 截取出class名称
                    label = self.name2label[name]                               # 根据种类写入标签
                    writer.writerow([img, label])                               # 保存csv文件
        	
        
        # 如果有csv文件的话直接读取
        images, labels = [], []
        with open(os.path.join(self.root, filename)) as f:
            reader = csv.reader(f)
            for row in reader:
                img, label = row
                label = int(label)
                images.append(img)
                labels.append(label)
        assert len(images) == len(labels)
        return images, labels

    
def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.required_grad = False
            
def self_iid(dataset, num_users):
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    dict_users, all_idxs = {}, [i for i in range(dataset.__len__())]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

class DatasetSplit(Dataset):
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = list(idxs)

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return image, label

class LocalUpdate(object):
    def __init__(self, dataset=None, idxs=None, local_bs=16):
        self.dataset = dataset
        self.idxs=idxs
        self.local_bs=local_bs
        
    def return_data(self):                
        return DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=self.local_bs, shuffle=True)
    
    
def layer_pruning(model, prune_ratio):
    """按层剪枝"""
    # 获取所有卷积层
    conv_layers = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            # 计算层的重要性（使用参数的L1范数）
            importance = torch.sum(torch.abs(module.weight.data))
            conv_layers.append((name, module, importance))
    #这里是获得所有的卷积层的参数
    # 按重要性排序
    conv_layers.sort(key=lambda x: x[2])
    #按照importance进行排序
    
    # 确定要剪枝的层数
    num_layers = len(conv_layers)
    num_pruned = int(num_layers * prune_ratio)
    
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
        #将新创建的1*1卷积层的权重全部初始化为0
        for c in range(min(module.in_channels, module.out_channels)):
        #遍历输入通道和输出通道
            identity_conv.weight.data[c, c, 0, 0] = 1.0
            #这相当于创建了一个单位矩阵的效果，使得该卷积层在前向传播时不会改变输入特征图。
        # 替换原始模块
        setattr(model, name, identity_conv)
        #python中的内置函数，用于设置对象的属性，将旧参数的值替换成为新参数

    
class VGGnet(nn.Module):
    def __init__(self,feature_extract=True,num_classes=3):
        super(VGGnet, self).__init__()
        model = models.vgg16(pretrained=True)
        self.features = model.features
        set_parameter_requires_grad(self.features, feature_extract)#固定特征提取层参数
        #自适应输出宽高尺寸为7×7
        self.avgpool=model.avgpool
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 1024),
            nn.ReLU(),
            nn.Linear(1024, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), 512*7*7)
        out=self.classifier(x)
        return out

def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model=VGGnet().to(device)

    #设置参数 
    learning_rate=0.01 #设置学习率
    num_epochs=5   #本地训练次数
    train_batch_size=16
    test_batch_size=16
    fl_epochs=5 #联邦学习次数
    clients_num=5
    
    #设置优化器，使用CrossEntropyLoss函数
    loss_fn=nn.CrossEntropyLoss()
    optimizer=torch.optim.Adam(model.classifier.parameters(),lr=learning_rate)

    #加载数据集和Dataloader
    train_data=GetData('../dataset/cat_dog_train',224,'train')
    test_data=GetData('../dataset/cat_dog_train',224,'test')
    client_train_dataloader={}
    client_test_dataloader={}
    user_train_idx={}
    user_test_idx={}
    user_train_idx=self_iid(train_data,clients_num)
    user_test_idx=self_iid(test_data,clients_num)
    for i in range(clients_num):
        client_train_dataloader[i]=LocalUpdate(train_data,user_train_idx[i],train_batch_size).return_data()
        client_test_dataloader[i]=LocalUpdate(test_data,user_test_idx[i],test_batch_size).return_data()
        
    #选择不同的随机用户进行训练
    #client_each_epoch=random.sample(1,clients_num/10)
    gNB_model=model
    #进行模型聚合
    for fl in range(fl_epochs):
        #设置联邦学习次数：
        selected_clients=random.sample(range(0,clients_num),min(5, clients_num))
        for client in selected_clients: #每轮设置随机5个用户进行训练
            #每次最多设置10个用户进行训练
            correct = 0
            total = 0
            for epoch in range(num_epochs):
                total_step=len(client_train_dataloader[client])
                loss_sum=0
                gNB_model.train()
                for i,(img,label) in enumerate(client_train_dataloader[client]):
                    #梯度归零，准备开始训练
                    optimizer.zero_grad()

                    #加标签与图片
                    img=img.to(device)
                    label=label.to(device)

                    #前向计算
                    output=model(img)
                    loss=loss_fn(output,label) #前向计算损失函数，但是需要传入output和label两个参数

                    #进行优化
                    loss.backward()
                    optimizer.step()

                    loss_sum+=loss.item()

                    #print("running Epoch:{}, client_idx:{}, client_round:{},loss is {}".format(epoch,client,i,loss.item()))
                    #print("running Epoch:{}, round:{},avg_loss is {}".format(epoch,i,loss_sum/total_step))


                gNB_model.eval()
                with torch.no_grad():
                    for images, labels in client_test_dataloader[client]:
                        images = images.to(device)
                        labels = labels.to(device)
                        output = model(images)
                        _, predicted = torch.max(output.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum().item()
            
            print('fl{}Client{} Test Accuracy  {} %'.format(fl,client,100*(correct/total)))
            model_name="fl{}client{}.pth".format(fl,client)
            torch.save(obj = model.state_dict(), f='model/{}'.format(model_name))
        
        client_models=[]
        for client in selected_clients:
            model_path=f'model/fl{fl}client{client}.pth'
            client_models.append(torch.load(model_path))
        print("get all clients models ready")

        sum_module=FedAvg(client_models)
        gNB_model.load_state_dict(sum_module)
        torch.save(gNB_model.state_dict(),f'model/fl{fl}gNB.pth')
        print("FegAvg is done")
        print("Fl Module prunning")
        layer_pruning(gNB_model,0.3)
        torch.save(gNB_model.state_dict(),f'model/fl{fl}gNB_Prun.pth')
        print("model prunning done")


    
    
if __name__=='__main__':
    main()