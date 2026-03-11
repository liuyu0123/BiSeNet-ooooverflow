import os
import shutil
import random
import argparse
from pathlib import Path

def split_dataset(root_dir, train_ratio=0.8, val_ratio=0.1, seed=42):
    root_path = Path(root_dir)
    img_dir = root_path / 'images'
    mask_dir = root_path / 'masks'
    
    if not img_dir.exists() or not mask_dir.exists():
        print(f"错误：找不到 {img_dir} 或 {mask_dir}")
        return

    # 获取所有图片文件 (支持 jpg, png, jpeg)
    valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    all_files = [f.name for f in img_dir.iterdir() if f.suffix.lower() in valid_extensions]
    
    if len(all_files) == 0:
        print("错误：images 文件夹中没有找到图片文件。")
        return

    # 随机打乱
    random.seed(seed)
    random.shuffle(all_files)

    n_total = len(all_files)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    # 剩下的给 test
    
    splits = {
        'train': all_files[:n_train],
        'val': all_files[n_train:n_train+n_val],
        'test': all_files[n_train+n_val:]
    }

    print(f"总文件数: {n_total}")
    print(f"训练集: {len(splits['train'])}")
    print(f"验证集: {len(splits['val'])}")
    print(f"测试集: {len(splits['test'])}")

    # 创建目标目录结构
    # 结构: root_dir/split_type/images 和 root_dir/split_type/masks
    for split_name in splits.keys():
        target_img_dir = root_path / split_name / 'images'
        target_mask_dir = root_path / split_name / 'masks'
        
        target_img_dir.mkdir(parents=True, exist_ok=True)
        target_mask_dir.mkdir(parents=True, exist_ok=True)

        print(f"正在处理 {split_name} 集...")
        for filename in splits[split_name]:
            src_img = img_dir / filename
            # 寻找对应的 mask (尝试不同后缀)
            base_name = filename.rsplit('.', 1)[0]
            src_mask = None
            
            # 在 mask 目录中查找匹配的文件
            for ext in valid_extensions:
                potential_mask = mask_dir / f"{base_name}{ext}"
                if potential_mask.exists():
                    src_mask = potential_mask
                    break
            
            if src_mask is None:
                print(f"警告：未找到 {filename} 对应的 mask，跳过。")
                continue

            # 复制文件
            shutil.copy(src_img, target_img_dir / filename)
            # 保持 mask 文件名与图片一致（如果后缀不同，这里统一改为.png 或者保持原样，建议保持原样或统一为png）
            # 为了保险，我们将 mask 重命名为与图片完全一致的名字（除了后缀可能不同，但代码逻辑通常只认前缀）
            # 这里我们直接复制，文件名保持一致
            shutil.copy(src_mask, target_mask_dir / src_mask.name)
            
            # 如果原图片和原 mask 后缀不一致导致文件名对不上，这里做一个重命名操作以匹配图片名
            # 假设我们希望 mask 的文件名（不含后缀）与图片完全一致
            mask_dest_name = base_name + src_mask.suffix 
            # 如果上面复制的名字不匹配图片的前缀逻辑，可以在这里强制重命名
            # 但通常只要前缀一致即可。为了最稳妥，我们把 mask 改名为和图片一样的名字（保留mask原后缀）
            final_mask_name = base_name + src_mask.suffix
            if (target_mask_dir / src_mask.name).name != final_mask_name:
                 os.rename(target_mask_dir / src_mask.name, target_mask_dir / final_mask_name)

    print("数据集分割完成！")
    print(f"新结构位于: {root_path}")
    print("目录结构示例:")
    print(f"  {root_path}/train/images/")
    print(f"  {root_path}/train/masks/")
    print(f"  {root_path}/val/images/")
    print(f"  {root_path}/val/masks/")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Split dataset into train/val/test")
    parser.add_argument('--root_dir', type=str, required=True, help='根目录路径 (包含 images 和 masks 文件夹)')
    parser.add_argument('--train_ratio', type=float, default=0.8, help='训练集比例')
    parser.add_argument('--val_ratio', type=float, default=0.1, help='验证集比例')
    args = parser.parse_args()
    
    split_dataset(args.root_dir, args.train_ratio, args.val_ratio)