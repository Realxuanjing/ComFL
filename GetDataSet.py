from torch.utils.data import Dataset, DataLoader
import os
from torchvision.transforms import transforms
from PIL import Image
import glob
import torch
import random
import csv
class GetData(Dataset):
    def __init__(self, root, resize, mode):
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
   