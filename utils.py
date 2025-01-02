import random
import os
from matplotlib import transforms
import numpy as np
import torch
import copy
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
    def __init__(self, noise_factor=25, blur_radius=2):
        self.noise_factor = noise_factor
        self.blur_radius = blur_radius  # 控制模糊的程度

    def add_noise(self, img):
        # 将图像转为 numpy 数组
        np_img = np.array(img)

        # 如果图像是 (3, H, W) 格式（C, H, W），转为 (H, W, C) 格式
        if np_img.ndim == 3 and np_img.shape[0] == 3:
            np_img = np.transpose(np_img, (1, 2, 0))  # 转换成 (height, width, channels)

        # 添加高斯噪声
        noise = np.random.normal(0, self.noise_factor, np_img.shape)
        noisy_img = np_img + noise
        noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

        noisy_img_pil = Image.fromarray(noisy_img)

        # 添加模糊效果
        blurred_img = noisy_img_pil.filter(ImageFilter.GaussianBlur(radius=self.blur_radius))

        return blurred_img

    def __call__(self, img):
        # Ensure img is a PIL Image before applying filters
        if isinstance(img, torch.Tensor):
            # Convert from Tensor to PIL Image
            img = transforms.ToPILImage()(img)

        # Apply noise to the image
        img = self.add_noise(img)
        
        # Apply Gaussian blur
        img = img.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Convert back to Tensor for further processing
        img = transforms.ToTensor()(img)
        return img

    # def add_noise(self, img):
    #     # 将输入的 PIL 图像转换为 numpy 数组
    #     np_img = np.array(img)

    #     # 打印图像的维度信息
    #     print(f"Image shape: {np_img.shape}, Image ndim: {np_img.ndim}")

    #     # 如果图像的维度是 (3, 32, 32)，我们需要将其转置为 (32, 32, 3)
    #     if np_img.ndim == 3 and np_img.shape[0] == 3:
    #         np_img = np.transpose(np_img, (1, 2, 0))  # 转换成 (height, width, channels)

    #     # 检查图像的维度，确保它是灰度、RGB 或 RGBA 图像
    #     if np_img.ndim == 2:
    #         # 对灰度图像添加噪声
    #         noise = np.random.normal(0, 25, np_img.shape)
    #         noisy_img = np_img + noise
    #     elif np_img.ndim == 3:
    #         if np_img.shape[2] == 3:
    #             # 对 RGB 图像添加噪声
    #             noise = np.random.normal(0, 25, np_img.shape)
    #             noisy_img = np_img + noise
    #         elif np_img.shape[2] == 4:
    #             # 对 RGBA 图像添加噪声（支持透明通道）
    #             noise = np.random.normal(0, 25, np_img.shape)
    #             noisy_img = np_img + noise
    #         else:
    #             raise ValueError("Unsupported number of channels in image")
    #     else:
    #         raise ValueError("Unsupported image format")

    #     # 确保像素值在 [0, 255] 范围内，并转换为 uint8 类型
    #     noisy_img = np.clip(noisy_img, 0, 255).astype(np.uint8)

    #     # 将噪声图像从 numpy 数组转换回 PIL 图像
    #     return Image.fromarray(noisy_img)

class HighQualityTransform:
    def __init__(self):
        self.transform = transforms.Compose([
            # transforms.RandomHorizontalFlip(),
            # transforms.RandomRotation(20),
            # transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
            transforms.ToTensor(),
        ])

    def __call__(self, img):
        # 如果 img 是一个 Tensor，先转换成 PIL.Image
        if isinstance(img, torch.Tensor):
            img = transforms.ToPILImage()(img)  # 转换为 PIL Image
        
        # 然后应用所有的变换
        return self.transform(img)
    


