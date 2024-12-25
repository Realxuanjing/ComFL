import torch
import torch.nn as nn
from utils import set_parameter_requires_grad
from torchvision import models 

class VGGnet(nn.Module):
    def __init__(self,feature_extract=True,num_classes=3):
        super(VGGnet, self).__init__()
        model = models.vgg16(pretrained=False)
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