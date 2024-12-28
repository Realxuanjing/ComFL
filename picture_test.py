from PIL import Image
from pathlib import Path

save_path = Path('/home/data1/xxx/dataset/COMFL')
dataset_path = Path(save_path) / 'datasets' / 'PetImages' / 'Dog'

# 存储损坏图片的列表
corrupted_images = []

# 遍历路径下的所有图片文件
for image_path in dataset_path.glob('*'):
    try:
        # 尝试打开图片
        with Image.open(image_path) as img:
            img.verify()  # 验证图片是否损坏
    except (IOError, SyntaxError) as e:
        # 如果图片损坏，加入到损坏图片列表中
        corrupted_images.append(image_path)
        print(f"Corrupted image: {image_path}")

# 输出损坏的图片列表
print("Corrupted images:", corrupted_images)
