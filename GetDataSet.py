from collections import defaultdict
from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms
from PIL import Image, ImageFilter
from utils import LowQualityTransform, HighQualityTransform
import glob
import torch
import random
import csv
import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, random_split, Dataset, TensorDataset, Subset
import numpy as np
class GetData(Dataset):
    def __init__(self, root, resize = None, mode = 'train'):
        super(GetData, self).__init__()
        self.root = root
        self.resize = resize
        self.name2label = {'Cat': 0, 'Dog': 1}                   # "类别名称": 编号,对自己的类别进行定义
        print('root',self.root)
        for name in sorted(os.listdir(os.path.join(root))):
            # 判断是否为一个目录
            if not os.path.isdir(os.path.join(root, name)):
                continue
            self.name2label[name] = self.name2label.get(name)           # 将类别名称转换为对应编号
            print('self.name2label',self.name2label)



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
            # print('self.name2label.keys()',self.name2label.keys())
            for name in self.name2label.keys():   					# 将文件夹内所有形式的图片读入images列表
                images += glob.glob(os.path.join(self.root, name, '*.png'))
                images += glob.glob(os.path.join(self.root, name, '*.jpg'))
                images += glob.glob(os.path.join(self.root, name, '*.jpeg'))
            random.shuffle(images)									# 随机打乱

            with open(os.path.join(self.root, filename), mode='w', newline='') as f:  # 新建csv文件，进行数据写入
                writer = csv.writer(f)
                for img in images:                                              # './data/class1/spot429.jpg'
                    name = img.split(os.sep)[-2]                                # 截取出class名称
                    label = self.name2label[name]
                    if label:                              # 根据种类写入标签
                        writer.writerow([img, label]) 


                                                      
        	
        
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
        
    def return_data(self,shuffle=True):                
        return DataLoader(DatasetSplit(self.dataset, self.idxs), batch_size=self.local_bs, shuffle=shuffle,pin_memory= True,num_workers=24)
   



class CRF_10():
    def __init__(self, batch_size ,logger = None,data_root='/home/data1/xxx/dataset/COMFL/datasets/CRF_10'):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        self.batch_size = batch_size
        self.logger = logger
        self.trainset = torchvision.datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
        self.testset = torchvision.datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
        
        self.trainloader = DataLoader(self.trainset, batch_size=self.batch_size, shuffle=True, num_workers=24)
        self.testloader = DataLoader(self.testset, batch_size=self.batch_size, shuffle=False, num_workers=24)


        self.classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    

    def getdata(self):
        return self.trainloader, self.testloader, self.classes

    def split_data(self, n_users, validation_split=0):
        """
        将训练集和测试集划分为 n 个 IID 子数据集，并为每个子数据集创建一个 DataLoader。
        
        参数：
        - n_users: 用户数，即划分为多少个子数据集
        - validation_split: 验证集的比例，默认为 0
        
        返回：
        - user_trainloaders: 一个包含 n 个训练集子数据集 DataLoader 的列表，每个用户对应一个训练集
        - user_testloaders: 一个包含 n 个测试集子数据集 DataLoader 的列表，每个用户对应一个测试集
        """
        trainset = self.trainset
        testset = self.testset
        
        total_train_size = len(trainset)
        total_test_size = len(testset)
        print('total_train_size,total_test_size',total_train_size,total_test_size)
        validation_size = int(total_train_size * validation_split)
        train_size = total_train_size - validation_size

        train_subset, val_subset = random_split(trainset, [train_size, validation_size])

        user_trainloaders = []
        user_testloaders = []
        assert train_size % n_users == 0, f"Error: train_size ({train_size}) is not divisible by n_users ({n_users})."
        assert total_test_size % n_users == 0, f"Error: total_test_size ({total_test_size}) is not divisible by n_users ({n_users})."
        user_train_subset = torch.utils.data.random_split(train_subset, [train_size // n_users] * n_users)
        user_val_subset = torch.utils.data.random_split(val_subset, [validation_size // n_users] * n_users)
        print('every user size',train_size // n_users,validation_size // n_users)
        user_test_subset = torch.utils.data.random_split(testset, [total_test_size // n_users] * n_users)

        for i in range(n_users):
            user_trainloaders.append(DataLoader(user_train_subset[i], batch_size=self.batch_size, shuffle=True, num_workers=24,pin_memory=True))
            user_testloaders.append(DataLoader(user_test_subset[i], batch_size=self.batch_size, shuffle=False, num_workers=24,pin_memory=True))
        
        return user_trainloaders, user_testloaders

    def split_quality_data(self, n_users, percent_low_quality=0.5):
        m_per_user = n_users * percent_low_quality
        user_trainloaders, user_testloaders = self.split_data(n_users)
        # 前 m_per_user 个loader为低质量数据，后面的为高质量数据
        # 使用之前定义的 split_data 函数获取训练和测试的 DataLoader
        user_trainloaders, user_testloaders = self.split_data(n_users)

        # 计算每个用户低质量数据的数量
        m_per_user = int(n_users * percent_low_quality)
        print(f"Number of users with low-quality data: {m_per_user}")
        # 创建低质量数据转换（假设已有 LowQualityTransform 定义）
        low_quality_transform = LowQualityTransform(noise_factor=25, blur_radius=2)
        high_quality_transform = HighQualityTransform()  # 高质量数据转换（原样）

        # 处理每个用户的训练数据
        # 处理每个用户的训练数据
        for i in range(n_users):
            # 获取每个用户的训练集
            train_loader = user_trainloaders[i]
            processed_train_images = []
            processed_train_labels = []

            # 遍历每个batch的数据
            for batch_images, batch_labels in train_loader:
                # 对batch中的每个图像逐个应用转换
                for img, label in zip(batch_images, batch_labels):
                    # 如果是低质量数据，应用低质量转换
                    if i < m_per_user:
                        img = low_quality_transform(img)
                    else:
                        img = high_quality_transform(img)
                    
                    processed_train_images.append(img)
                    processed_train_labels.append(label)

            # 创建新的 DataLoader
            processed_train_data = list(zip(processed_train_images, processed_train_labels))
            user_trainloaders[i] = DataLoader(processed_train_data, batch_size=self.batch_size, shuffle=True, num_workers=24)

        # 处理每个用户的测试数据
        for i in range(n_users):
            # 获取每个用户的测试集
            test_loader = user_testloaders[i]
            processed_test_images = []
            processed_test_labels = []

            # 遍历每个batch的数据
            for batch_images, batch_labels in test_loader:
                # 对batch中的每个图像逐个应用转换
                for img, label in zip(batch_images, batch_labels):
                    # 如果是低质量数据，应用低质量转换
                    if i < m_per_user:
                        img = low_quality_transform(img)
                    else:
                        img = high_quality_transform(img)
                    
                    processed_test_images.append(img)
                    processed_test_labels.append(label)

            # 创建新的 DataLoader
            processed_test_data = list(zip(processed_test_images, processed_test_labels))
            user_testloaders[i] = DataLoader(processed_test_data, batch_size=self.batch_size, shuffle=False, num_workers=24)

        return user_trainloaders, user_testloaders


    def non_iid_partition(self, dataset, n_users, n_classes=10):
        """
        将数据集划分为 n_users 个 non-IID 子集。训练集和测试集都会被划分为 non-IID 数据。
        
        参数:
        - dataset: 需要划分的数据集，应该是一个包含训练集和测试集的数据集。
        - n_users: 子集的数量，即用户数量。
        - n_classes: 数据集中的类别数，默认为 10。
        
        返回:
        - user_trainloaders: 每个用户的训练集 DataLoader 列表。
        - user_testloaders: 每个用户的测试集 DataLoader 列表。
        """
        # 假设 dataset 中包含 trainset 和 testset
        trainset, testset = dataset['train'], dataset['test']

        # 对训练集进行类别划分
        train_class_dict = defaultdict(list)
        for idx, (_, label) in enumerate(trainset):
            train_class_dict[label].append(idx)

        # 对测试集进行类别划分
        test_class_dict = defaultdict(list)
        for idx, (_, label) in enumerate(testset):
            test_class_dict[label].append(idx)
        
        # 为每个用户分配数据
        user_train_data = [[] for _ in range(n_users)]
        user_test_data = [[] for _ in range(n_users)]

        # 为每个类别分配训练集和测试集给用户
        user_idx = 0
        for label in range(n_classes):
            # 将每个类别的训练数据分配给用户
            for i in range(len(train_class_dict[label])):
                user_train_data[user_idx].append(train_class_dict[label][i])
                user_idx = (user_idx + 1) % n_users  # 循环分配给不同的用户
            
            # 将每个类别的测试数据分配给用户
            for i in range(len(test_class_dict[label])):
                user_test_data[user_idx].append(test_class_dict[label][i])
                user_idx = (user_idx + 1) % n_users  # 循环分配给不同的用户
        
        for i in range(n_users):
            user_train_labels = set([trainset[idx][1] for idx in user_train_data[i]])
            user_test_labels = set([testset[idx][1] for idx in user_test_data[i]])
            print(f"User {i + 1}: Categories in Train: {sorted(user_train_labels)}, Categories in Test: {sorted(user_test_labels)}")
            self.logger.info(f"User {i + 1}: Categories in Train: {sorted(user_train_labels)}, Categories in Test: {sorted(user_test_labels)}")
        # 创建用户的训练集和测试集 DataLoader
        user_trainloaders = [DataLoader(Subset(trainset, user_train_data[i]), batch_size=32, shuffle=True) for i in range(n_users)]
        user_testloaders = [DataLoader(Subset(testset, user_test_data[i]), batch_size=32, shuffle=False) for i in range(n_users)]
        
        return user_trainloaders, user_testloaders
    

    def img_to_tensor(self, img):
        # 如果 img 已经是 Tensor，则直接返回
        if isinstance(img, torch.Tensor):
            return img
        # 如果 img 是 PIL 图像或 ndarray，则转换为 Tensor
        elif isinstance(img, (np.ndarray, Image.Image)):
            return transforms.ToTensor()(img)
        else:
            raise TypeError(f"Unsupported image type: {type(img)}")

    
