import os
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms
import torch
import numpy as np

class WaterDataset(Dataset):
    def __init__(self, images_dir, masks_dir, scale=(360, 480), mode='train'):
        """
        images_dir: 图片文件夹路径 (e.g., data/water/train/images)
        masks_dir:  mask文件夹路径 (e.g., data/water/train/masks)
        scale: (height, width) 输入模型的尺寸，默认360x480 (可根据显存调整)
        """
        self.images_dir = images_dir
        self.masks_dir = masks_dir
        self.scale = scale
        
        # 获取所有图片文件名
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        self.image_filenames = [
            f for f in os.listdir(images_dir) 
            if f.lower().endswith(valid_exts)
        ]
        
        if len(self.image_filenames) == 0:
            raise ValueError(f"在 {images_dir} 中未找到任何图片文件！")

        # 定义变换
        # 图片需要归一化到 [0,1] 并转为 Tensor
        self.img_transform = transforms.Compose([
            transforms.Resize((scale[0], scale[1])),
            transforms.ToTensor(),
            # 如果需要标准归一化 (ImageNet均值方差)，可以取消下面注释
            # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Mask 只需要 Resize 和 转为 Tensor，不需要归一化，且需保持像素值整数
        self.mask_transform = transforms.Compose([
            transforms.Resize((scale[0], scale[1]), interpolation=Image.NEAREST),
            transforms.ToTensor() # 这会变成 [0, 1] 之间的 float
        ])

    def __len__(self):
        return len(self.image_filenames)

    def __getitem__(self, idx):
        img_name = self.image_filenames[idx]
        img_path = os.path.join(self.images_dir, img_name)
        
        # 1. 加载图片
        image = Image.open(img_path).convert('RGB')
        
        # 2. 加载 Mask
        # 尝试匹配同名文件，支持不同后缀
        base_name = os.path.splitext(img_name)[0]
        mask_path = None
        valid_mask_exts = ('.png', '.jpg', '.jpeg', '.bmp')
        
        for ext in valid_mask_exts:
            potential_path = os.path.join(self.masks_dir, base_name + ext)
            if os.path.exists(potential_path):
                mask_path = potential_path
                break
        
        if mask_path is None:
            # 如果找不到完全同名的，尝试遍历 masks 目录找第一个存在的（容错）
            # 但最好保证文件名一致
            raise FileNotFoundError(f"未找到 {img_name} 对应的 Mask 文件。请检查 masks 文件夹。")
            
        mask = Image.open(mask_path)
        
        # 3. 处理 Mask 格式 (关键步骤)
        # 无论原图是 RGB 还是 单通道(L)，我们都将其转换为 0 和 1 的单通道索引图
        if mask.mode != 'L':
            mask = mask.convert('L') # 转为灰度
            
        mask_np = np.array(mask)
        
        # 二值化处理：假设非水是0，水是>0的任何值 (如1或255)
        # 如果你的数据里水是255，这里会自动变成1.0 (因为后面ToTensor会除以255)
        # 为了保险，我们手动确保只有0和1
        binary_mask = (mask_np > 0).astype(np.uint8) * 255
        mask = Image.fromarray(binary_mask)
        
        # 4. 应用变换
        image = self.img_transform(image)
        mask = self.mask_transform(mask)
        
        # 5. 格式化输出
        # BiSeNet 需要 Label 是 LongTensor 类型，且形状为 [H, W] (没有通道维)
        # ToTensor 后 shape 是 [1, H, W]，值为 0.0 或 1.0
        mask = mask.squeeze(0) # 去掉通道维 -> [H, W]
        mask = mask.long()     # 转为 Long (0, 1) -> 用于 CrossEntropyLoss
        
        return image, mask