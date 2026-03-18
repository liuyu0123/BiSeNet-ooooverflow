import argparse
import os
import csv
import time
from pathlib import Path

import tqdm
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

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


def save_results_to_csv(metrics, model_info, args, save_path):
    """保存测试结果到 CSV（标准格式）"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    # 构建结果字典 - 使用真实计算的 loss
    result = {
        'model_path': args.model_path,
        'model_type': 'BiSeNet',
        'backbone': args.context_path,
        'test_images': args.test_images if args.test_images else os.path.join(args.data_root, args.split, 'images'),
        'test_masks': args.test_masks if args.test_masks else os.path.join(args.data_root, args.split, 'masks'),
        'crop_height': args.crop_height,
        'crop_width': args.crop_width,
        'num_classes': args.num_classes,
        'total_params': model_info['total_params'],
        'model_size_mb': f"{model_info['model_size_mb']:.2f}",
        'test_loss': f"{metrics['loss']:.6f}",  # 使用真实计算的 loss
        'test_precision': f"{metrics['precision']:.6f}",
        'test_recall': f"{metrics['recall']:.6f}",
        'test_f1': f"{metrics['f1']:.6f}",
        'test_miou': f"{metrics['miou']:.6f}",
        'test_acc': f"{metrics['acc']:.6f}",
        'inference_time_ms': f"{metrics['inference_time_ms']:.4f}",
        'fps': f"{metrics['fps']:.2f}",
        'total_images': metrics['total_images'],
    }

    # 写入 CSV（追加模式）
    header = list(result.keys())
    file_exists = save_path.exists()

    with open(save_path, 'a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=header)
        if not file_exists:
            writer.writeheader()
        writer.writerow(result)

    print(f"Results saved to {save_path}")

    # 同时保存详细文本报告
    report_path = save_path.parent / f"{save_path.stem}_report.txt"
    with open(report_path, 'a') as f:
        f.write(f"\n{'='*50}\n")
        f.write(f"Test Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"Model: {args.model_path}\n")
        f.write(f"Model Type: BiSeNet\n")
        f.write(f"Backbone: {args.context_path}\n")
        f.write(f"Test Images: {result['test_images']}\n")
        f.write(f"Test Masks: {result['test_masks']}\n\n")
        f.write(f"Total Parameters: {model_info['total_params']:,}\n")
        f.write(f"Model Size: {model_info['model_size_mb']:.2f} MB\n")
        f.write(f"Input Size: {args.crop_height}x{args.crop_width}\n")
        f.write(f"Number of Classes: {args.num_classes}\n\n")
        f.write(f"Test Set Size: {metrics['total_images']} images\n\n")
        f.write(f"Test Loss:      {metrics['loss']:.6f}\n")  # 添加 loss 到报告
        f.write(f"Global Pixel Accuracy: {metrics['acc']*100:.3f}%\n")
        f.write(f"mIoU:           {metrics['miou']:.6f}\n")
        f.write(f"Precision:      {metrics['precision']:.6f}\n")
        f.write(f"Recall:         {metrics['recall']:.6f}\n")
        f.write(f"F1-Score:       {metrics['f1']:.6f}\n")
        f.write(f"Inference Time: {metrics['inference_time_ms']:.4f} ms\n")
        f.write(f"FPS:            {metrics['fps']:.2f}\n")
        f.write(f"{'='*50}\n")

    print(f"Text report appended to {report_path}")


def evaluate(args):
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    use_gpu = torch.cuda.is_available() and args.use_gpu
    print(f"Using GPU: {args.cuda}" if use_gpu else "Using CPU")

    # 1. 准备数据路径
    use_direct_paths = args.test_images is not None or args.test_masks is not None

    if use_direct_paths:
        if args.test_images is None or args.test_masks is None:
            raise ValueError("--test-images 和 --test-masks 必须同时指定")

        img_dir = args.test_images
        mask_dir = args.test_masks
        split_name = "test"

        print("使用直接指定的测试集路径模式")
        print(f"测试图像: {img_dir}")
        print(f"测试掩码: {mask_dir}")
    else:
        img_dir = os.path.join(args.data_root, args.split, 'images')
        mask_dir = os.path.join(args.data_root, args.split, 'masks')
        split_name = args.split

        print(f"使用 data_root 模式: {args.data_root} (split: {args.split})")

    # 检查路径
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

    # 获取模型信息
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

    model.eval()

    # 3. 创建保存预测结果的目录
    save_pred_dir = os.path.join(args.save_pred_path, f'{split_name}_predictions')
    os.makedirs(save_pred_dir, exist_ok=True)
    print(f"Predictions will be saved to: {save_pred_dir}")

    # 4. 开始评估
    hist = np.zeros((args.num_classes, args.num_classes))
    precision_record = []

    # 测量推理时间
    inference_times = []
    total_images = 0

    # 新增：累计 loss
    total_loss = 0.0
    loss_count = 0

    print(f"Start evaluating on {len(dataset)} images...")

    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm.tqdm(dataloader)):
            if use_gpu:
                data = data.cuda()
                label = label.cuda()

            # 测量推理时间
            if use_gpu:
                torch.cuda.synchronize()
            batch_start = time.time()

            outputs = model(data)

            if use_gpu:
                torch.cuda.synchronize()
            batch_time = time.time() - batch_start
            inference_times.append(batch_time)

            # 新增：计算 loss（BiSeNet 返回多个输出）
            # outputs 通常是 [main_output, aux_output1, aux_output2] 的元组
            if isinstance(outputs, (tuple, list)):
                main_output = outputs[0]
                # 计算主输出的交叉熵损失
                loss = F.cross_entropy(main_output, label.squeeze(1).long(), ignore_index=255)
            else:
                # 单输出情况
                loss = F.cross_entropy(outputs, label.squeeze(1).long(), ignore_index=255)

            total_loss += loss.item()
            loss_count += 1

            predict = outputs[0] if isinstance(outputs, (tuple, list)) else outputs

            # 修复：处理 batch 维度，reverse_one_hot 期望 3D 输入 [C, H, W]
            # predict 当前是 [B, C, H, W]，batch_size=1，所以 squeeze(0) 得到 [C, H, W]
            if predict.dim() == 4:
                predict = predict.squeeze(0)  # [1, C, H, W] -> [C, H, W]

            # 后处理
            predict = reverse_one_hot(predict)
            predict_np = predict.cpu().numpy()
            label_np = label.squeeze().cpu().numpy()

            # 计算指标
            precision = compute_global_accuracy(predict_np, label_np)
            precision_record.append(precision)
            hist += fast_hist(label_np.flatten(), predict_np.flatten(), args.num_classes)

            # 保存预测图
            pred_img = (predict_np * 255).astype(np.uint8)
            pred_pil = Image.fromarray(pred_img)

            img_name = dataset.image_filenames[i]
            pred_pil.save(os.path.join(save_pred_dir, img_name))

            total_images += 1

    # 5. 计算最终指标
    test_time = sum(inference_times)
    mean_precision = np.mean(precision_record)
    miou_list = per_class_iu(hist)
    miou = np.mean(miou_list)

    # 从混淆矩阵计算详细指标
    detailed_metrics = compute_metrics_from_hist(hist, args.num_classes)

    # 计算推理时间
    avg_inference_time_ms = np.mean(inference_times) * 1000
    fps = total_images / test_time if test_time > 0 else 0

    # 计算平均 loss
    avg_loss = total_loss / loss_count if loss_count > 0 else 0.0

    # 构建指标字典 - 包含真实 loss
    metrics = {
        'loss': avg_loss,  # 新增：真实计算的 loss
        'acc': mean_precision,
        'miou': miou,
        'precision': detailed_metrics['precision'],
        'recall': detailed_metrics['recall'],
        'f1': detailed_metrics['f1'],
        'inference_time_ms': avg_inference_time_ms,
        'fps': fps,
        'total_images': total_images,
    }

    # 6. 输出结果
    print("-" * 50)
    if use_direct_paths:
        print(f"Test Images: {img_dir}")
        print(f"Test Masks:  {mask_dir}")
    else:
        print(f"Dataset: {args.data_root} ({args.split})")
    print(f"Total Images: {total_images}")
    print(f"Test Loss:      {avg_loss:.6f}")  # 新增：打印 loss
    print(f"Global Pixel Accuracy: {mean_precision*100:.3f}%")
    print(f"mIoU:           {miou:.6f}")
    print(f"Precision:      {detailed_metrics['precision']:.6f}")
    print(f"Recall:         {detailed_metrics['recall']:.6f}")
    print(f"F1-Score:       {detailed_metrics['f1']:.6f}")
    print(f"Inference Time: {avg_inference_time_ms:.4f} ms")
    print(f"FPS:            {fps:.2f}")
    print(f"Class IoUs:     {miou_list}")
    print("-" * 50)
    print("Evaluation Done!")

    # 7. 保存结果
    if args.output:
        output_path = args.output
    else:
        # 默认保存到模型目录
        model_dir = Path(args.model_path).parent
        output_path = model_dir / f"BiSeNet_{args.context_path}_test_results.csv"

    save_results_to_csv(metrics, model_info, args, output_path)

    return metrics


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

    parser.add_argument('--save_pred_path', type=str, default='./eval_results', 
                        help='Dir to save prediction images')
    # 新增：输出结果路径
    parser.add_argument('--output', type=str, default=None,
                        help='测试结果CSV保存路径（可选）')

    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--context_path', type=str, default='resnet18')
    parser.add_argument('--crop_height', type=int, default=360)
    parser.add_argument('--crop_width', type=int, default=480)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args()
    evaluate(args)