import argparse
import os
import csv
import time
from pathlib import Path
from collections import defaultdict

import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import cv2

from dataset.WaterDataset import WaterDataset
from model.build_BiSeNet import BiSeNet
from utils import reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu


def get_model_info(model):
    """获取模型静态信息"""
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / (1024 * 1024),
    }


def compute_metrics_from_hist(hist, num_classes):
    """从混淆矩阵计算各项指标"""
    precision_per_class = []
    recall_per_class = []
    f1_per_class = []
    iou_per_class = []

    for i in range(num_classes):
        tp = hist[i, i]
        fp = hist[:, i].sum() - tp
        fn = hist[i, :].sum() - tp

        precision = tp / (tp + fp + 1e-10)
        recall = tp / (tp + fn + 1e-10)
        f1 = 2 * precision * recall / (precision + recall + 1e-10)
        iou = tp / (tp + fp + fn + 1e-10)

        precision_per_class.append(precision)
        recall_per_class.append(recall)
        f1_per_class.append(f1)
        iou_per_class.append(iou)

    return {
        'precision': np.mean(precision_per_class),
        'recall': np.mean(recall_per_class),
        'f1': np.mean(f1_per_class),
        'miou': np.mean(iou_per_class),
    }


def compute_metrics_per_image(pred_mask, gt_mask, num_classes):
    """计算单张图片的指标"""
    hist = fast_hist(gt_mask.flatten(), pred_mask.flatten(), num_classes)
    metrics = compute_metrics_from_hist(hist, num_classes)
    
    # 计算像素准确率
    acc = compute_global_accuracy(pred_mask, gt_mask)
    metrics['acc'] = acc
    
    return metrics, hist


def overlay_mask_on_image(image, mask, alpha=0.5, color=(255, 0, 0)):
    """
    将mask以透明蒙版形式叠加到原图
    image: PIL Image or numpy array (RGB)
    mask: numpy array (H, W), 0或1，1表示水体（前景）
    alpha: 透明度
    color: 蒙版颜色 (B, G, R) 或 (R, G, B)，这里使用红色 (255, 0, 0)
    返回: PIL Image
    """
    # 转换为numpy数组
    if isinstance(image, Image.Image):
        img_array = np.array(image)
    else:
        img_array = image.copy()
    
    # 确保mask是二值的
    mask_binary = (mask > 0).astype(np.uint8)
    
    # 创建红色蒙版图层
    overlay = img_array.copy()
    
    # 如果图像是灰度图，转为RGB
    if len(img_array.shape) == 2:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_GRAY2RGB)
        overlay = img_array.copy()
    
    # 在mask为1的区域应用红色
    overlay[mask_binary == 1] = color
    
    # 混合原图和蒙版
    result = cv2.addWeighted(img_array, 1-alpha, overlay, alpha, 0)
    
    return Image.fromarray(result)


def save_results_to_csv(per_image_results, overall_metrics, model_info, args, save_path):
    """保存测试结果到 CSV（包含逐图结果和整体平均）"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 写入CSV
    with open(save_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # 写入表头
        writer.writerow(['filename', 'precision', 'recall', 'f1', 'miou', 'accuracy', 'inference_time_ms', 'fps'])

        # 写入每幅图的结果
        for result in per_image_results:
            writer.writerow([
                result['filename'],
                f"{result['precision']:.6f}",
                f"{result['recall']:.6f}",
                f"{result['f1']:.6f}",
                f"{result['miou']:.6f}",
                f"{result['acc']:.6f}",
                f"{result['inference_time_ms']:.4f}",
                f"{result['fps']:.2f}"
            ])

        # 空行分隔
        writer.writerow([])

        # 写入整体平均指标
        writer.writerow(['OVERALL_AVERAGE',
                        f"{overall_metrics['precision']:.6f}",
                        f"{overall_metrics['recall']:.6f}",
                        f"{overall_metrics['f1']:.6f}",
                        f"{overall_metrics['miou']:.6f}",
                        f"{overall_metrics['acc']:.6f}",
                        f"{overall_metrics['inference_time_ms']:.4f}",
                        f"{overall_metrics['fps']:.2f}"])
        
        # 写入模型信息
        writer.writerow([])
        writer.writerow(['Model Info'])
        writer.writerow(['model_path', args.model_path])
        writer.writerow(['model_type', 'BiSeNet'])
        writer.writerow(['backbone', args.context_path])
        writer.writerow(['total_params', model_info['total_params']])
        writer.writerow(['model_size_mb', f"{model_info['model_size_mb']:.2f}"])
        writer.writerow(['num_classes', args.num_classes])
        writer.writerow(['total_images', overall_metrics['total_images']])
        writer.writerow(['test_time_total_s', f"{overall_metrics['test_time_total_s']:.2f}"])

    print(f"Results saved to {save_path}")


def load_image(image_path, scale=(360, 480)):
    """加载单张图片并进行预处理"""
    image = Image.open(image_path).convert('RGB')
    original_size = image.size  # (W, H)
    
    # 调整尺寸
    image = image.resize((scale[1], scale[0]), Image.BILINEAR)
    
    # 转换为tensor并归一化
    import torchvision.transforms as transforms
    transform = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]) #与训练时完全一致！不使用Normalize
    ])
    
    tensor = transform(image).unsqueeze(0)  # 添加batch维度
    
    return tensor, original_size


def evaluate(args):
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    use_gpu = torch.cuda.is_available() and args.use_gpu
    print(f"Using GPU: {args.cuda}" if use_gpu else "Using CPU")

    # 1. 确定输入模式（单图或文件夹）
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input path does not exist: {args.input}")
        return
    
    is_single_image = input_path.is_file()
    
    # 获取图片列表
    if is_single_image:
        image_files = [input_path]
        print(f"Single image mode: {input_path}")
    else:
        # 支持的图片格式
        valid_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff']
        image_files = [f for f in input_path.iterdir() 
                      if f.suffix.lower() in valid_exts]
        image_files.sort()
        print(f"Folder mode: found {len(image_files)} images in {input_path}")

    # 2. 确定是否有真值mask（用于计算指标）
    has_ground_truth = args.ground_truth is not None
    gt_dir = Path(args.ground_truth) if has_ground_truth else None
    
    if has_ground_truth:
        if not gt_dir.exists():
            print(f"Warning: Ground truth path does not exist: {gt_dir}")
            has_ground_truth = False
        else:
            print(f"Ground truth mode enabled: {gt_dir}")
    
    # 3. 加载模型
    model = BiSeNet(args.num_classes, args.context_path)
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")

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

    # 验证权重是否正确加载（插入到 model.eval() 之前）
    # 获取第一个可训练参数
    first_param = None
    for name, param in (model.module if use_gpu else model).named_parameters():
        first_param = param.data.cpu().numpy()
        print(f"第一个参数名: {name}")
        break  # 只看第一个

    if first_param is not None:
        print(f"\n=== 权重检查 ===")
        print(f"权重均值: {first_param.mean():.6f}")
        print(f"权重标准差: {first_param.std():.6f}")

        # 关键判断：随机初始化的权重通常均值接近0，标准差较小（如0.02左右）
        if abs(first_param.mean()) < 0.001 and first_param.std() < 0.05:
            print("⚠️ 警告：权重看起来是随机初始化！模型加载可能失败了")
        else:
            print("✅ 权重看起来是训练过的")
        print(f"================\n")

    model.eval()

    # 4. 创建输出目录（如果指定了output）
    save_output = args.output is not None
    if save_output:
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        vis_dir = output_dir / 'visualizations'
        vis_dir.mkdir(exist_ok=True)
        print(f"Output directory: {output_dir}")
        print(f"Visualizations will be saved to: {vis_dir}")

    # 5. 开始推理
    per_image_results = []
    total_hist = np.zeros((args.num_classes, args.num_classes))
    total_inference_time = 0.0
    
    print(f"Start processing {len(image_files)} images...")

    with torch.no_grad():
        for idx, img_path in enumerate(tqdm.tqdm(image_files)):
            # 加载原图（用于可视化）
            original_image = Image.open(img_path).convert('RGB')
            orig_w, orig_h = original_image.size
            
            # 预处理
            input_tensor, _ = load_image(img_path, scale=(args.crop_height, args.crop_width))
            
            if use_gpu:
                input_tensor = input_tensor.cuda()

            # 对第一张图做 warmup，排除初始化开销，避免第一行计时结果异常
            if idx == 0:
                _ = model(input_tensor)
                if use_gpu:
                    torch.cuda.synchronize()

            # 正式推理计时
            if use_gpu:
                torch.cuda.synchronize()
            start_time = time.time()

            outputs = model(input_tensor)

            if use_gpu:
                torch.cuda.synchronize()
            inference_time = (time.time() - start_time) * 1000  # 转换为ms
            total_inference_time += inference_time

            # 处理输出
            if isinstance(outputs, (tuple, list)):
                main_output = outputs[0]
            else:
                main_output = outputs
            
            # 获取预测mask
            predict = main_output.squeeze(0)  # [C, H, W]
            predict = reverse_one_hot(predict)
            predict_np = predict.cpu().numpy()
            
            # 将预测mask调整回原始尺寸
            predict_pil = Image.fromarray((predict_np * 255).astype(np.uint8))
            predict_pil_resized = predict_pil.resize((orig_w, orig_h), Image.NEAREST)
            predict_fullres = np.array(predict_pil_resized) // 255  # 转回0/1

            # 保存可视化结果（红色透明蒙版）
            if save_output:
                # 水体为红色，非水体保持原图
                overlay_img = overlay_mask_on_image(
                    original_image, 
                    predict_fullres, 
                    alpha=args.alpha,
                    color=(255, 0, 0)  # 红色 (R, G, B)
                )
                vis_path = vis_dir / f"{img_path.stem}_result{img_path.suffix}"
                overlay_img.save(vis_path)

            # 如果有真值，计算指标
            img_metrics = None
            if has_ground_truth:
                gt_path = gt_dir / f"{img_path.stem}.png"  # 假设mask是png格式
                if not gt_path.exists():
                    # 尝试其他扩展名
                    for ext in ['.jpg', '.jpeg', '.png', '.bmp']:
                        gt_path = gt_dir / f"{img_path.stem}{ext}"
                        if gt_path.exists():
                            break
                
                if gt_path.exists():
                    gt_mask = Image.open(gt_path).convert('L')
                    gt_mask = gt_mask.resize((orig_w, orig_h), Image.NEAREST)
                    gt_np = np.array(gt_mask)
                    # 二值化（假设mask中>127为前景）
                    gt_np = (gt_np > 127).astype(np.uint8)
                    
                    # 确保预测和真值尺寸一致
                    if predict_fullres.shape != gt_np.shape:
                        predict_fullres = np.array(
                            Image.fromarray((predict_fullres * 255).astype(np.uint8))
                            .resize((gt_np.shape[1], gt_np.shape[0]), Image.NEAREST)
                        ) // 255
                    
                    # 计算单图指标
                    img_metrics, hist = compute_metrics_per_image(predict_fullres, gt_np, args.num_classes)
                    img_metrics['filename'] = img_path.name
                    img_metrics['inference_time_ms'] = inference_time
                    img_metrics['fps'] = 1000.0 / inference_time if inference_time > 0 else 0.0
                    img_metrics['acc'] = img_metrics['acc']  # 确保字段名一致
                    
                    per_image_results.append(img_metrics)
                    total_hist += hist
                else:
                    print(f"Warning: Ground truth not found for {img_path.name}")
                    # 仍然记录推理时间，但指标为NaN
                    per_image_results.append({
                        'filename': img_path.name,
                        'precision': float('nan'),
                        'recall': float('nan'),
                        'f1': float('nan'),
                        'miou': float('nan'),
                        'acc': float('nan'),
                        'inference_time_ms': inference_time,
                        'fps': 1000.0 / inference_time if inference_time > 0 else 0.0
                    })

    # 6. 计算整体指标（如果有真值）
    if has_ground_truth and len(per_image_results) > 0:
        # 过滤掉没有真值的记录（NaN）
        valid_results = [r for r in per_image_results if not np.isnan(r['precision'])]
        
        if len(valid_results) > 0:
            overall_metrics = {
                'precision': np.mean([r['precision'] for r in valid_results]),
                'recall': np.mean([r['recall'] for r in valid_results]),
                'f1': np.mean([r['f1'] for r in valid_results]),
                'miou': np.mean([r['miou'] for r in valid_results]),
                'acc': np.mean([r['acc'] for r in valid_results]),
                'inference_time_ms': total_inference_time / len(image_files),
                'fps': 1000.0 / (total_inference_time / len(image_files)) if total_inference_time > 0 else 0.0,
                'total_images': len(image_files),
                'test_time_total_s': total_inference_time / 1000
            }
            
            # 从混淆矩阵计算的整体指标（更准确）
            overall_metrics_hist = compute_metrics_from_hist(total_hist, args.num_classes)
            overall_metrics_miou = np.mean(per_class_iu(total_hist))
            
            # 使用混淆矩阵计算的指标更准确
            overall_metrics['precision'] = overall_metrics_hist['precision']
            overall_metrics['recall'] = overall_metrics_hist['recall']
            overall_metrics['f1'] = overall_metrics_hist['f1']
            overall_metrics['miou'] = overall_metrics_miou
            
            # 7. 输出结果
            print("-" * 50)
            print(f"Processed {len(image_files)} images")
            print(f"Valid images with GT: {len(valid_results)}")
            print(f"Overall Average Metrics:")
            print(f"  Precision:      {overall_metrics['precision']:.6f}")
            print(f"  Recall:         {overall_metrics['recall']:.6f}")
            print(f"  F1-Score:       {overall_metrics['f1']:.6f}")
            print(f"  mIoU:           {overall_metrics['miou']:.6f}")
            print(f"  Pixel Accuracy: {overall_metrics['acc']:.6f}")
            print(f"  Avg Inference:  {overall_metrics['inference_time_ms']:.4f} ms")
            print(f"  FPS:            {1000/overall_metrics['inference_time_ms']:.2f}")
            print("-" * 50)
            
            # 保存CSV
            if save_output:
                csv_path = output_dir / 'evaluation_results.csv'
                save_results_to_csv(per_image_results, overall_metrics, model_info, args, csv_path)
            
            # 打印逐图结果摘要（仅前5个和总结）
            print("\nPer-image Results (first 5):")
            for i, result in enumerate(per_image_results[:5]):
                print(f"  {result['filename']}: P={result['precision']:.4f}, R={result['recall']:.4f}, F1={result['f1']:.4f}, mIoU={result['miou']:.4f}")
            if len(per_image_results) > 5:
                print(f"  ... and {len(per_image_results)-5} more images")
        else:
            print("Warning: No valid ground truth masks found for evaluation")
    else:
        # 无真值模式，仅打印推理统计
        avg_time = total_inference_time / len(image_files) if len(image_files) > 0 else 0
        print("-" * 50)
        print(f"Processed {len(image_files)} images (no ground truth evaluation)")
        print(f"Average inference time: {avg_time:.4f} ms")
        print(f"FPS: {1000/avg_time:.2f}" if avg_time > 0 else "N/A")
        print("-" * 50)

    print("Evaluation Done!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate BiSeNet for Water Segmentation")

    parser.add_argument('--model_path', type=str, required=True, 
                        help='Path to the trained model (.pth)')

    # 新的统一输入参数
    parser.add_argument('--input', type=str, required=True,
                        help='输入路径：可以是单张图片路径，也可以是图片文件夹路径')

    # 真值mask路径（可选，用于计算指标）
    parser.add_argument('--ground_truth', type=str, default=None,
                        help='真值mask文件夹路径（可选）。如果提供，将计算Precision、Recall、F1、mIoU等指标')

    # 输出路径（可选，用于保存可视化结果和CSV）
    parser.add_argument('--output', type=str, default=None,
                        help='输出文件夹路径（可选）。如果提供，将保存可视化叠加图和评估CSV')

    # 可视化参数
    parser.add_argument('--alpha', type=float, default=0.5,
                        help='蒙版透明度 (0.0-1.0)，默认0.5')

    # 模型参数
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--context_path', type=str, default='resnet18')
    parser.add_argument('--crop_height', type=int, default=360)
    parser.add_argument('--crop_width', type=int, default=480)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()
    evaluate(args)