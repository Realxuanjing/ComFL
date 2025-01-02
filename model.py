import torch
import torch.nn as nn
from utils import set_parameter_requires_grad
from torchvision import models 
import torch.nn.functional as F

class VGGnet(nn.Module):
    def __init__(self,feature_extract=True,num_classes=3):
        super(VGGnet, self).__init__()
        model = models.vgg16(weights=None)
        self.features = model.features
        set_parameter_requires_grad(self.features, feature_extract)#固定特征提取层参数
        #自适应输出宽高尺寸为7×7
        self.avgpool=nn.AdaptiveAvgPool2d((2, 2))
        self.classifier = nn.Sequential(
            nn.Linear(512*2*2 , 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        out=self.classifier(x)
        return out
    
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super(SimpleCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        # self.fc1 = nn.Linear(256 * 28 * 28, 1024) 
        self.fc1 = nn.Linear(256 * 4 * 4, 1024) 
        self.fc2 = nn.Linear(1024, num_classes)     
        
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))  
        x = self.pool(F.relu(self.conv2(x)))  
        x = self.pool(F.relu(self.conv3(x)))  
        
        # x = x.view(-1, 256 * 28 * 28)
        x = x.view(-1, 256 * 4 * 4)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)  
        
        return x




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
        
        
        # self.fc1 = nn.Linear(512 * 7 * 7, 4096)  
        # self.fc2 = nn.Linear(4096, 1024)         
        # self.fc3 = nn.Linear(1024, num_classes)  

        self.fc1 = nn.Linear(512 * 1 * 1, 1024)
        self.fc2 = nn.Linear(1024, 512)
        self.fc3 = nn.Linear(512, num_classes)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))  
        x = self.pool(F.relu(self.bn2(self.conv2(x))))  
        x = self.pool(F.relu(self.bn3(self.conv3(x))))  
        x = self.pool(F.relu(self.bn4(self.conv4(x))))  
        x = self.pool(F.relu(self.bn5(self.conv5(x))))  
        
        # x = x.view(-1, 512 * 7 * 7)  
        x = x.view(-1, 512 * 1 * 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)  
        x = F.relu(self.fc2(x))
        x = self.fc3(x)      
        
        return x
