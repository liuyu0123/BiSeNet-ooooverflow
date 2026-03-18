import argparse
import os
import csv
import time
from pathlib import Path

import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from dataset.WaterDataset import WaterDataset
from model.build_BiSeNet import BiSeNet
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
from loss import DiceLoss


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


class MetricsLogger:
    """指标记录器，生成标准格式CSV"""
    
    def __init__(self, save_path, model_info, args):
        self.save_path = Path(save_path)
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model_info = model_info
        self.args = args
        
        self.header = [
            'epoch',
            'train_loss', 'train_precision', 'train_recall', 'train_f1', 'train_miou',
            'val_precision', 'val_recall', 'val_f1', 'val_miou',
            'inference_time_ms', 'fps', 'learning_rate'
        ]
        self.rows = []
        
    def log_epoch(self, epoch, train_loss, train_metrics, val_metrics, 
                  inference_time_ms, fps, lr):
        """记录一轮数据"""
        row = {
            'epoch': epoch,
            'train_loss': f"{train_loss:.6f}",
            'train_precision': f"{train_metrics.get('precision', 0):.6f}",
            'train_recall': f"{train_metrics.get('recall', 0):.6f}",
            'train_f1': f"{train_metrics.get('f1', 0):.6f}",
            'train_miou': f"{train_metrics.get('miou', 0):.6f}",
            'val_precision': f"{val_metrics.get('precision', 0):.6f}",
            'val_recall': f"{val_metrics.get('recall', 0):.6f}",
            'val_f1': f"{val_metrics.get('f1', 0):.6f}",
            'val_miou': f"{val_metrics.get('miou', 0):.6f}",
            'inference_time_ms': f"{inference_time_ms:.4f}",
            'fps': f"{fps:.2f}",
            'learning_rate': f"{lr:.8f}",
        }
        self.rows.append(row)
        
    def save(self):
        """保存CSV"""
        with open(self.save_path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
            writer.writerows(self.rows)
        print(f"\nTraining log saved to {self.save_path}")
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.save_path.parent / f"{self.save_path.stem}_model_info.txt"
        with open(info_path, 'w') as f:
            f.write(f"Model Type: BiSeNet\n")
            f.write(f"Backbone: {self.args.context_path}\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Input Size: {self.args.crop_height}x{self.args.crop_width}\n")
            f.write(f"Num Classes: {self.args.num_classes}\n")
            f.write(f"Epochs: {self.args.num_epochs}\n")
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Learning Rate: {self.args.learning_rate}\n")
            f.write(f"Optimizer: {self.args.optimizer}\n")
            f.write(f"Loss: {self.args.loss}\n")


def val(args, model, dataloader):
    """验证函数 - 完全按照成功版本"""
    print('Start Validation...')
    model.eval()
    precision_record = []
    hist = np.zeros((args.num_classes, args.num_classes))
    
    # 测量推理时间
    inference_times = []
    total_images = 0
    
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm.tqdm(dataloader)):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # 测量推理时间
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            start_time = time.time()
            
            outputs = model(data)
            
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            inference_times.append(time.time() - start_time)
            
            predict = outputs[0]
            
            # 使用 reverse_one_hot 处理 (batch_size=1，所以是3D)
            predict = reverse_one_hot(predict)
            predict = predict.cpu().numpy()
            label = label.squeeze().cpu().numpy()

            precision = compute_global_accuracy(predict, label)
            precision_record.append(precision)
            
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)
            total_images += data.size(0)

    # 计算指标
    precision = np.mean(precision_record)
    miou_list = per_class_iu(hist)
    miou = np.mean(miou_list)
    
    # 计算详细指标
    detailed_metrics = compute_metrics_from_hist(hist, args.num_classes)
    
    # 计算推理时间
    avg_inference_time_ms = np.mean(inference_times) * 1000
    fps = total_images / sum(inference_times) if sum(inference_times) > 0 else 0
    
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation mIoU: {miou:.4f}')
    print(f'Class IoUs: {miou_list}')
    print(f'Inference Time: {avg_inference_time_ms:.4f} ms, FPS: {fps:.2f}')
    
    model.train()
    
    # 返回完整指标
    metrics = {
        'precision': detailed_metrics['precision'],
        'recall': detailed_metrics['recall'],
        'f1': detailed_metrics['f1'],
        'miou': miou,
        'acc': precision,
    }
    
    return metrics, avg_inference_time_ms, fps


def train(args, model, optimizer, dataloader_train, dataloader_val, metrics_logger):
    writer = SummaryWriter(comment=f'_water_{args.optimizer}_{args.context_path}')
    
    # 选择损失函数
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255)
    else:
        raise ValueError("Unsupported loss type")

    max_miou = 0.0
    step = 0
    
    for epoch in range(args.num_epochs):
        # 学习率策略
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description(f'Epoch {epoch}, LR {lr:.6f}')
        
        loss_record = []
        train_hist = np.zeros((args.num_classes, args.num_classes))
        
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            outputs = model(data)
            output, output_sup1, output_sup2 = outputs
            
            # 计算三个损失的加权和
            loss1 = loss_func(output, label)
            loss2 = loss_func(output_sup1, label)
            loss3 = loss_func(output_sup2, label)
            
            loss = loss1 + loss2 + loss3
            
            tq.update(args.batch_size)
            tq.set_postfix(loss=f'{loss.item():.6f}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            writer.add_scalar('loss_step', loss.item(), step)
            loss_record.append(loss.item())
            
            # 收集训练指标 - 修复：遍历batch，对每个样本单独调用reverse_one_hot
            with torch.no_grad():
                # output shape: [B, C, H, W]
                batch_size = output.size(0)
                for b in range(batch_size):
                    # 取单个样本 [C, H, W]
                    single_output = output[b]  # [C, H, W]
                    single_label = label[b]    # [H, W] 或 [1, H, W]
                    
                    # 调用 reverse_one_hot (期望3D输入)
                    pred = reverse_one_hot(single_output)
                    pred = pred.cpu().numpy()
                    label_np = single_label.squeeze().cpu().numpy()
                    
                    train_hist += fast_hist(label_np.flatten(), pred.flatten(), args.num_classes)
            
        tq.close()
        
        avg_loss = np.mean(loss_record)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.6f}')
        writer.add_scalar('loss_epoch', avg_loss, epoch)
        
        # 计算训练指标
        train_metrics = compute_metrics_from_hist(train_hist, args.num_classes)

        # 定期验证
        if (epoch + 1) % args.validation_step == 0:
            val_metrics, inference_time_ms, fps = val(args, model, dataloader_val)
            writer.add_scalar('val_precision', val_metrics['acc'], epoch)
            writer.add_scalar('val_miou', val_metrics['miou'], epoch)
            
            # 记录到CSV
            metrics_logger.log_epoch(
                epoch + 1,
                avg_loss,
                train_metrics,
                val_metrics,
                inference_time_ms,
                fps,
                lr
            )
            
            if val_metrics['miou'] > max_miou:
                max_miou = val_metrics['miou']
                if not os.path.isdir(args.save_model_path):
                    os.makedirs(args.save_model_path)
                
                save_path = os.path.join(args.save_model_path, 'best_water_seg.pth')
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f'New Best Model Saved at {save_path} with mIoU: {val_metrics["miou"]:.4f}')
        else:
            # 不验证时也记录训练指标
            empty_metrics = {'precision': 0, 'recall': 0, 'f1': 0, 'miou': 0}
            metrics_logger.log_epoch(
                epoch + 1,
                avg_loss,
                train_metrics,
                empty_metrics,
                0,
                0,
                lr
            )

    # 保存训练日志
    metrics_logger.save()
    print(f"\nTraining complete! Best Val mIoU: {max_miou:.4f}")


def main(params=None):
    parser = argparse.ArgumentParser(description="Train BiSeNet for Water Segmentation")
    
    # 核心参数
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2, help='水 + 非水 = 2类')
    
    # 数据路径配置
    parser.add_argument('--images', type=str, default=None, help='训练图像目录路径')
    parser.add_argument('--masks', type=str, default=None, help='训练掩码目录路径')
    parser.add_argument('--val-images', type=str, default=None, dest='val_images', help='验证图像目录路径')
    parser.add_argument('--val-masks', type=str, default=None, dest='val_masks', help='验证掩码目录路径')
    
    # 旧方式
    parser.add_argument('--data_root', type=str, default='./my_water_data', 
                        help='数据集根目录 (仅当未直接指定 images/masks 时使用)')
    parser.add_argument('--crop_height', type=int, default=360)
    parser.add_argument('--crop_width', type=int, default=480)
    
    # 模型与训练配置
    parser.add_argument('--context_path', type=str, default='resnet18', choices=['resnet18', 'resnet101'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--loss', type=str, default='crossentropy', choices=['crossentropy', 'dice'])
    parser.add_argument('--validation_step', type=int, default=1)
    parser.add_argument('--save_model_path', type=str, default='./checkpoints_water')
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args(params)
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    print(f"Using GPU: {args.cuda}")

    # 构建数据集路径
    use_direct_paths = args.images is not None or args.masks is not None
    
    if use_direct_paths:
        if args.images is None or args.masks is None:
            parser.error("--images 和 --masks 必须同时指定")
        if args.val_images is None or args.val_masks is None:
            parser.error("--val-images 和 --val-masks 必须同时指定")
        
        train_img_dir = args.images
        train_mask_dir = args.masks
        val_img_dir = args.val_images
        val_mask_dir = args.val_masks
        
        print("使用直接指定的路径模式")
    else:
        train_img_dir = os.path.join(args.data_root, 'train', 'images')
        train_mask_dir = os.path.join(args.data_root, 'train', 'masks')
        val_img_dir = os.path.join(args.data_root, 'val', 'images')
        val_mask_dir = os.path.join(args.data_root, 'val', 'masks')
        
        print(f"使用 data_root 模式: {args.data_root}")

    # 检查路径
    paths_to_check = [
        (train_img_dir, "训练图像"),
        (train_mask_dir, "训练掩码"),
        (val_img_dir, "验证图像"),
        (val_mask_dir, "验证掩码")
    ]
    
    all_paths_exist = True
    for path, desc in paths_to_check:
        if not os.path.exists(path):
            print(f"Error: {desc}路径不存在: {path}")
            all_paths_exist = False
        else:
            print(f"✓ {desc}: {path}")
    
    if not all_paths_exist:
        if not use_direct_paths:
            print("\n请运行 split_dataset.py 生成 train/val 文件夹，或直接指定 --images/--masks 等参数")
        return

    print(f"\nLoading datasets...")
    
    dataset_train = WaterDataset(
        images_dir=train_img_dir,
        masks_dir=train_mask_dir,
        scale=(args.crop_height, args.crop_width),
        mode='train'
    )
    
    dataset_val = WaterDataset(
        images_dir=val_img_dir,
        masks_dir=val_mask_dir,
        scale=(args.crop_height, args.crop_width),
        mode='val'
    )

    print(f"Train samples: {len(dataset_train)}, Val samples: {len(dataset_val)}")

    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True
    )
    
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=1,
        shuffle=False, 
        num_workers=args.num_workers
    )

    # 构建模型
    model = BiSeNet(args.num_classes, args.context_path)
    
    # 获取模型信息并创建记录器
    model_info = get_model_info(model)
    print(f"Model: {model_info['total_params']:,} params, {model_info['model_size_mb']:.2f} MB")
    
    # 创建记录器
    log_filename = f"BiSeNet_{args.context_path}_training_log_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    metrics_logger = MetricsLogger(
        os.path.join(args.save_model_path, log_filename),
        model_info,
        args
    )
    metrics_logger.save_model_info()
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # 加载预训练权重
    if args.pretrained_model_path:
        print(f"Loading pretrained model from {args.pretrained_model_path}")
        if torch.cuda.is_available() and args.use_gpu:
            model.module.load_state_dict(torch.load(args.pretrained_model_path))
        else:
            model.load_state_dict(torch.load(args.pretrained_model_path, map_location='cpu'))

    # 优化器
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), args.learning_rate, momentum=0.9, weight_decay=1e-4)
    else:
        optimizer = torch.optim.RMSprop(model.parameters(), args.learning_rate)

    print("Starting Training...")
    train(args, model, optimizer, dataloader_train, dataloader_val, metrics_logger)


if __name__ == '__main__':
    main()