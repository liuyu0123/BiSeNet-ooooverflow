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

    # 1. 准备数据
    # 注意：这里通常评估 test 集，如果没有 test 集，可以用 val 集
    img_dir = os.path.join(args.data_root, args.split, 'images')
    mask_dir = os.path.join(args.data_root, args.split, 'masks')
    
    if not os.path.exists(img_dir):
        print(f"Error: Path {img_dir} does not exist.")
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
    save_pred_dir = os.path.join(args.save_pred_path, f'{args.split}_predictions')
    os.makedirs(save_pred_dir, exist_ok=True)
    print(f"Predictions will be saved to: {save_pred_dir}")

    # 4. 开始评估
    hist = np.zeros((args.num_classes, args.num_classes))
    precision_record = []
    
    print(f"Start evaluating on {args.split} set...")
    
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
    
    print("-" * 30)
    print(f"Dataset: {args.data_root} ({args.split})")
    print(f"Total Images: {len(dataset)}")
    print(f"Global Pixel Accuracy: {mean_precision:.4f}")
    print(f"Class IoUs: {miou_list}")
    print(f"Mean IoU (mIoU): {miou:.4f}")
    print("-" * 30)
    print("Evaluation Done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate BiSeNet for Water Segmentation")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to the trained model (.pth)')
    parser.add_argument('--data_root', type=str, default='./my_water_data', help='Dataset root')
    parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Split to evaluate')
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