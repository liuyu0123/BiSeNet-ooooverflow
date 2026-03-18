import argparse
import os
import tqdm
import torch
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

# 引入自定义数据集
from dataset.WaterDataset import WaterDataset
from model.build_BiSeNet import BiSeNet
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu

def evaluate(args):
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    use_gpu = torch.cuda.is_available() and args.use_gpu
    print(f"Using GPU: {args.cuda}" if use_gpu else "Using CPU")

    # 1. 准备数据路径（支持新旧两种方式）
    # 检查是否使用了新方式（直接指定路径）
    use_direct_paths = args.test_images is not None or args.test_masks is not None
    
    if use_direct_paths:
        # 新方式：直接使用指定的测试集路径
        if args.test_images is None or args.test_masks is None:
            raise ValueError("--test-images 和 --test-masks 必须同时指定")
        
        img_dir = args.test_images
        mask_dir = args.test_masks
        split_name = "test"  # 用于保存预测结果的子目录名
        
        print("使用直接指定的测试集路径模式")
        print(f"测试图像: {img_dir}")
        print(f"测试掩码: {mask_dir}")
    else:
        # 旧方式：通过 data_root + split 构建路径
        img_dir = os.path.join(args.data_root, args.split, 'images')
        mask_dir = os.path.join(args.data_root, args.split, 'masks')
        split_name = args.split
        
        print(f"使用 data_root 模式: {args.data_root} (split: {args.split})")

    # 检查路径是否存在
    if not os.path.exists(img_dir):
        print(f"Error: Image path does not exist: {img_dir}")
        return
    if not os.path.exists(mask_dir):
        print(f"Error: Mask path does not exist: {mask_dir}")
        return

    dataset = WaterDataset(
        images_dir=img_dir,
        masks_dir=mask_dir,
        scale=(args.crop_height, args.crop_width),
        mode='val'
    )
    
    dataloader = DataLoader(
        dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=args.num_workers
    )

    # 2. 加载模型
    model = BiSeNet(args.num_classes, args.context_path)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    
    # 加载权重
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found at {args.model_path}")
        return
        
    print(f"Loading model from {args.model_path}...")
    if use_gpu:
        model.module.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model.eval()

    # 3. 创建保存预测结果的目录
    save_pred_dir = os.path.join(args.save_pred_path, f'{split_name}_predictions')
    os.makedirs(save_pred_dir, exist_ok=True)
    print(f"Predictions will be saved to: {save_pred_dir}")

    # 4. 开始评估
    hist = np.zeros((args.num_classes, args.num_classes))
    precision_record = []
    
    print(f"Start evaluating on {len(dataset)} images...")
    
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm.tqdm(dataloader)):
            if use_gpu:
                data = data.cuda()
                label = label.cuda()
            
            # 前向传播
            outputs = model(data)
            predict = outputs[0] # 取主输出
            
            # 后处理：Logits -> Class Index
            predict = reverse_one_hot(predict)
            predict_np = predict.squeeze().cpu().numpy()
            label_np = label.squeeze().cpu().numpy()
            
            # 计算指标
            precision = compute_global_accuracy(predict_np, label_np)
            precision_record.append(precision)
            hist += fast_hist(label_np.flatten(), predict_np.flatten(), args.num_classes)
            
            # 保存预测图 (可选：将 0/1 矩阵转为可视化的黑白图)
            # 0->黑 (非水), 1->白 (水)
            pred_img = (predict_np * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_img)
            
            # 获取原文件名
            img_name = dataset.image_filenames[i]
            pred_pil.save(os.path.join(save_pred_dir, img_name))

    # 5. 输出最终结果
    mean_precision = np.mean(precision_record)
    miou_list = per_class_iu(hist)
    miou = np.mean(miou_list)
    
    print("-" * 40)
    if use_direct_paths:
        print(f"Test Images: {img_dir}")
        print(f"Test Masks:  {mask_dir}")
    else:
        print(f"Dataset: {args.data_root} ({args.split})")
    print(f"Total Images: {len(dataset)}")
    print(f"Global Pixel Accuracy: {mean_precision:.4f}")
    print(f"Class IoUs: {miou_list}")
    print(f"Mean IoU (mIoU): {miou:.4f}")
    print("-" * 40)
    print("Evaluation Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate BiSeNet for Water Segmentation")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth)')
    
    # 新方式：直接指定测试集路径
    parser.add_argument('--test-images', type=str, default=None, dest='test_images', 
                        help='测试图像目录路径（直接指定方式）')
    parser.add_argument('--test-masks', type=str, default=None, dest='test_masks', 
                        help='测试掩码目录路径（直接指定方式）')
    
    # 旧方式：通过 data_root + split 构建路径
    parser.add_argument('--data_root', type=str, default='./my_water_data', 
                        help='Dataset root (仅当未直接指定 test-images 时使用)')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], 
                        help='Split to evaluate (仅当使用 data_root 模式时使用)')
    
    parser.add_argument('--save_pred_path', type=str, default='./eval_results', help='Dir to save prediction images')
    
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--context_path', type=str, default='resnet18')
    parser.add_argument('--crop_height', type=int, default=360)
    parser.add_argument('--crop_width', type=int, default=480)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--cuda', type=str, default='0')
    
    args = parser.parse_args()
    evaluate(args)