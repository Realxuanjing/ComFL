import random
import os
from matplotlib import transforms
import numpy as np
import torch
from torchvision.transforms import transforms
from PIL import Image, ImageFilter
def seed_everything(seed=73):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True

def set_parameter_requires_grad(model, feature_extracting):
    if feature_extracting:
        for param in model.parameters():
            param.requires_grad = False

def self_iid(dataset, num_users):# 存在问题，后期应该按照标签随机分
    """
    Sample I.I.D. client data from MNIST dataset
    :param dataset:
    :param num_users:
    :return: dict of image index
    """
    num_items = int(len(dataset)/num_users)
    print("数据集德长度是：{}".format(len(dataset)))
    dict_users, all_idxs = {}, [i for i in range(dataset.__len__())]
    for i in range(num_users):
        dict_users[i] = set(np.random.choice(all_idxs, num_items, replace=False))
        all_idxs = list(set(all_idxs) - dict_users[i])
    return dict_users

class LowQualityTransform:
    def __init__(self, noise_prob=0.2, blur_prob=0.2):
        self.noise_prob = noise_prob
        self.blur_prob = blur_prob

    def __call__(self, img):
        if random.random() < self.noise_prob:
            img = self.add_noise(img)
        
        if random.random() < self.blur_prob:
            img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        return img

    def add_noise(self, img):
        np_img = np.array(img)
        noise = np.random.normal(0, 25, np_img.shape)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)
        return Image.fromarray(noisy_img)

class HighQualityTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(20),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        return self.transform(img)