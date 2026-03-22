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


def save_checkpoint(model, save_dir, filename):
    """
    保存模型检查点
    自动处理 DataParallel 包装的情况
    """
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, filename)
    
    if isinstance(model, torch.nn.DataParallel):
        state_dict = model.module.state_dict()
    else:
        state_dict = model.state_dict()
    
    torch.save(state_dict, save_path)
    print(f'  模型已保存: {save_path}')
    return save_path


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
    """指标记录器，支持实时写入CSV（每个epoch写入一次）"""
    
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
        
        # 立即创建文件并写入表头（覆盖模式）
        with open(self.save_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writeheader()
        print(f"日志文件初始化: {self.save_path}")
        
    def log_epoch(self, epoch, train_loss, train_metrics, val_metrics, 
                  inference_time_ms, fps, lr):
        """记录一轮数据并立即写入文件（追加模式）"""
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
        # 立即追加写入CSV，不缓存
        with open(self.save_path, 'a', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=self.header)
            writer.writerow(row)
            
    def save(self):
        """保留此方法以保持兼容性（不再执行写入操作，因为已实时写入）"""
        print(f"训练日志已保存至: {self.save_path}")
        
    def save_model_info(self):
        """保存模型信息"""
        info_path = self.save_path.parent / f"{self.save_path.stem}_model_info.txt"
        with open(info_path, 'w', encoding='utf-8') as f:
            f.write(f"Model Type: BiSeNet\n")
            f.write(f"Backbone: {self.args.context_path}\n")
            f.write(f"Total Parameters: {self.model_info['total_params']:,}\n")
            f.write(f"Trainable Parameters: {self.model_info['trainable_params']:,}\n")
            f.write(f"Model Size: {self.model_info['model_size_mb']:.2f} MB\n")
            f.write(f"Input Size: {self.args.crop_height}x{self.args.crop_width}\n")
            f.write(f"Num Classes: {self.args.num_classes}\n")
            f.write(f"Epochs: {self.args.epochs}\n")
            f.write(f"Batch Size: {self.args.batch_size}\n")
            f.write(f"Learning Rate: {self.args.learning_rate}\n")
            f.write(f"Optimizer: {self.args.optimizer}\n")
            f.write(f"Loss: {self.args.loss}\n")
            f.write(f"Model Name: {self.args.model_name}\n")
            f.write(f"Save Interval: {self.args.save_interval}\n")


def val(args, model, dataloader):
    """验证函数"""
    print('开始验证...')
    model.eval()
    precision_record = []
    hist = np.zeros((args.num_classes, args.num_classes))
    
    # 测量推理时间
    inference_times = []
    total_images = 0
    
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm.tqdm(dataloader, desc='Val')):
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
            
            # 使用 reverse_one_hot 处理
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
    
    print(f'验证精度: {precision:.4f}, mIoU: {miou:.4f}')
    print(f'各类IoU: {miou_list}')
    print(f'推理时间: {avg_inference_time_ms:.2f} ms, FPS: {fps:.2f}')
    
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
    
    for epoch in range(args.epochs):
        # 学习率策略
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.epochs)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description(f'Epoch {epoch}')
        
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
            tq.set_postfix(loss=f'{loss.item():.6f}', lr=f'{lr:.6f}')
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            step += 1
            writer.add_scalar('loss_step', loss.item(), step)
            loss_record.append(loss.item())
            
            # 收集训练指标
            with torch.no_grad():
                batch_size = output.size(0)
                for b in range(batch_size):
                    single_output = output[b]
                    single_label = label[b]
                    
                    pred = reverse_one_hot(single_output)
                    pred = pred.cpu().numpy()
                    label_np = single_label.squeeze().cpu().numpy()
                    
                    train_hist += fast_hist(label_np.flatten(), pred.flatten(), args.num_classes)
            
        tq.close()
        
        avg_loss = np.mean(loss_record)
        print(f'\nEpoch {epoch} 平均损失: {avg_loss:.6f}, 学习率: {lr:.6f}')
        writer.add_scalar('loss_epoch', avg_loss, epoch)
        
        # 计算训练指标
        train_metrics = compute_metrics_from_hist(train_hist, args.num_classes)

        # 定期验证
        if (epoch + 1) % args.validation_step == 0:
            val_metrics, inference_time_ms, fps = val(args, model, dataloader_val)
            writer.add_scalar('val_precision', val_metrics['acc'], epoch)
            writer.add_scalar('val_miou', val_metrics['miou'], epoch)
            
            # 记录到CSV（实时写入）
            metrics_logger.log_epoch(
                epoch + 1,
                avg_loss,
                train_metrics,
                val_metrics,
                inference_time_ms,
                fps,
                lr
            )
            
            # 保存最佳模型
            if val_metrics['miou'] > max_miou:
                max_miou = val_metrics['miou']
                save_checkpoint(model, args.model_dir, f"{args.model_name}_best.pth")
                print(f'  >>> 新的最佳模型! mIoU: {val_metrics["miou"]:.4f}')
        else:
            # 不验证时也记录训练指标（验证指标为0）
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
        
        # 分步保存中间模型（按指定间隔）
        if args.save_interval > 0 and (epoch + 1) % args.save_interval == 0:
            save_checkpoint(model, args.model_dir, f"{args.model_name}_epoch{epoch+1}.pth")

    # 训练结束，保存最终模型（last）
    print(f'\n训练完成，保存最终模型...')
    save_checkpoint(model, args.model_dir, f"{args.model_name}_last.pth")
    
    # 保存训练日志（已实时写入，此调用仅为兼容性）
    metrics_logger.save()
    print(f"\n训练结束! 最佳验证mIoU: {max_miou:.4f}")
    print(f"模型保存目录: {args.model_dir}")
    print(f"日志保存路径: {args.log_dir}")


def main(params=None):
    parser = argparse.ArgumentParser(description="Train BiSeNet for Water Segmentation")
    
    # 核心参数
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--learning-rate', type=float, default=1e-3)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2, help='水 + 非水 = 2类')
    
    # 数据路径配置
    parser.add_argument('--images', type=str, default=None, help='训练图像目录路径')
    parser.add_argument('--masks', type=str, default=None, help='训练掩码目录路径')
    parser.add_argument('--val-images', type=str, default=None, dest='val_images', help='验证图像目录路径')
    parser.add_argument('--val-masks', type=str, default=None, dest='val_masks', help='验证掩码目录路径')
    
    # 旧方式（兼容）
    parser.add_argument('--data_root', type=str, default='./my_water_data', 
                        help='数据集根目录 (仅当未直接指定 images/masks 时使用)')
    parser.add_argument('--crop_height', type=int, default=360)
    parser.add_argument('--crop_width', type=int, default=480)
    
    # 模型与训练配置
    parser.add_argument('--context_path', type=str, default='resnet18', choices=['resnet18', 'resnet101'])
    parser.add_argument('--optimizer', type=str, default='adam', choices=['adam', 'sgd'])
    parser.add_argument('--loss', type=str, default='crossentropy', choices=['crossentropy', 'dice'])
    parser.add_argument('--validation_step', type=int, default=1, help='每N个epoch验证一次')
    
    # 新增：保存路径和文件名控制参数
    parser.add_argument('--save_model_path', type=str, default='./checkpoints_water',
                        help='默认保存目录（如果未指定--model-dir和--log-dir）')
    parser.add_argument('--model-dir', type=str, default=None,
                        help='模型保存目录（默认使用 --save_model_path）')
    parser.add_argument('--log-dir', type=str, default=None,
                        help='日志保存目录（默认使用 --save_model_path）')
    parser.add_argument('--model-name', type=str, default=None,
                        help='模型保存文件名前缀（默认：BiSeNet_{context_path}）')
    parser.add_argument('--log-name', type=str, default=None,
                        help='日志文件名前缀（默认：BiSeNet_{context_path}_training_log_时间戳）')
    parser.add_argument('--save-interval', type=int, default=0,
                        help='分步保存频率，每N个epoch保存一次中间模型，0为不保存（默认：0）')
    
    parser.add_argument('--pretrained_model_path', type=str, default=None)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--cuda', type=str, default='0')

    args = parser.parse_args(params)
    
    # 设置 GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    print(f"使用 GPU: {args.cuda}")

    # 设置默认路径和文件名
    if args.model_dir is None:
        args.model_dir = args.save_model_path
    if args.log_dir is None:
        args.log_dir = args.save_model_path
    if args.model_name is None:
        args.model_name = f"BiSeNet_{args.context_path}"
    if args.log_name is None:
        args.log_name = f"BiSeNet_{args.context_path}_training_log_{time.strftime('%Y%m%d_%H%M%S')}"
    
    # 确保目录存在
    os.makedirs(args.model_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

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
            print(f"错误: {desc}路径不存在: {path}")
            all_paths_exist = False
        else:
            print(f"✓ {desc}: {path}")
    
    if not all_paths_exist:
        if not use_direct_paths:
            print("\n请运行 split_dataset.py 生成 train/val 文件夹，或直接指定 --images/--masks 等参数")
        return

    print(f"\n加载数据集...")
    
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

    print(f"训练样本: {len(dataset_train)}, 验证样本: {len(dataset_val)}")

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
    print(f"\n模型信息: {model_info['total_params']:,} 参数, {model_info['model_size_mb']:.2f} MB")
    
    # 创建记录器 - 使用指定的日志路径和文件名
    log_filename = args.log_name if args.log_name.endswith('.csv') else f"{args.log_name}.csv"
    metrics_logger = MetricsLogger(
        os.path.join(args.log_dir, log_filename),
        model_info,
        args
    )
    metrics_logger.save_model_info()
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # 加载预训练权重
    if args.pretrained_model_path:
        print(f"加载预训练模型: {args.pretrained_model_path}")
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

    print(f"\n开始训练...")
    print(f"模型保存目录: {args.model_dir}")
    print(f"日志保存目录: {args.log_dir}")
    print(f"模型名称前缀: {args.model_name}")
    print(f"分步保存间隔: {args.save_interval} (0为不保存中间模型)")
    
    train(args, model, optimizer, dataloader_train, dataloader_val, metrics_logger)


if __name__ == '__main__':
    main()