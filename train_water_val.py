import argparse
import os
import tqdm
import torch
import numpy as np
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

# 引入自定义数据集
from dataset.WaterDataset import WaterDataset
# 引入模型构建 (假设 model/build_BiSeNet.py 存在且不变)
from model.build_BiSeNet import BiSeNet
# 引入工具函数 (假设 utils.py 存在且不变)
from utils import poly_lr_scheduler, reverse_one_hot, compute_global_accuracy, fast_hist, per_class_iu
# 引入损失函数 (假设 loss.py 存在)
from loss import DiceLoss

def val(args, model, dataloader):
    print('Start Validation...')
    model.eval()
    precision_record = []
    # 初始化混淆矩阵 (2x2)
    hist = np.zeros((args.num_classes, args.num_classes))
    
    with torch.no_grad():
        for i, (data, label) in enumerate(tqdm.tqdm(dataloader)):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda()

            # 模型前向传播
            # BiSeNet 返回 (out, out_sup1, out_sup2)
            outputs = model(data)
            predict = outputs[0] # 只取主输出进行验证
            
            # 将 logits 转为类别索引 [B, H, W]
            predict = reverse_one_hot(predict) 
            predict = predict.cpu().numpy()

            label = label.squeeze().cpu().numpy()

            # 计算精度和直方图
            precision = compute_global_accuracy(predict, label)
            precision_record.append(precision)
            
            # 更新混淆矩阵
            hist += fast_hist(label.flatten(), predict.flatten(), args.num_classes)

    precision = np.mean(precision_record)
    miou_list = per_class_iu(hist)
    miou = np.mean(miou_list)
    
    print(f'Validation Precision: {precision:.4f}')
    print(f'Validation mIoU: {miou:.4f}')
    print(f'Class IoUs: {miou_list}')
    
    model.train() # 切回训练模式
    return precision, miou

def train(args, model, optimizer, dataloader_train, dataloader_val):
    writer = SummaryWriter(comment=f'_water_{args.optimizer}_{args.context_path}')
    
    # 选择损失函数
    if args.loss == 'dice':
        loss_func = DiceLoss()
    elif args.loss == 'crossentropy':
        loss_func = torch.nn.CrossEntropyLoss(ignore_index=255) # 忽略可能的255背景，虽然我们已经二值化了
    else:
        raise ValueError("Unsupported loss type")

    max_miou = 0.0
    step = 0
    
    for epoch in range(args.num_epochs):
        # 学习率策略
        lr = poly_lr_scheduler(optimizer, args.learning_rate, iter=epoch, max_iter=args.num_epochs)
        # 手动更新优化器学习率
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
            
        model.train()
        tq = tqdm.tqdm(total=len(dataloader_train) * args.batch_size)
        tq.set_description(f'Epoch {epoch}, LR {lr:.6f}')
        
        loss_record = []
        
        for i, (data, label) in enumerate(dataloader_train):
            if torch.cuda.is_available() and args.use_gpu:
                data = data.cuda()
                label = label.cuda() # Shape: [B, H, W], Values: 0 or 1

            outputs = model(data)
            output, output_sup1, output_sup2 = outputs
            
            # 计算三个损失的加权和 (BiSeNet 经典做法)
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
            
        tq.close()
        
        avg_loss = np.mean(loss_record)
        print(f'Epoch {epoch} Average Loss: {avg_loss:.6f}')
        writer.add_scalar('loss_epoch', avg_loss, epoch)

        # 定期验证
        if (epoch + 1) % args.validation_step == 0:
            precision, miou = val(args, model, dataloader_val)
            writer.add_scalar('val_precision', precision, epoch)
            writer.add_scalar('val_miou', miou, epoch)
            
            if miou > max_miou:
                max_miou = miou
                if not os.path.isdir(args.save_model_path):
                    os.makedirs(args.save_model_path)
                
                save_path = os.path.join(args.save_model_path, 'best_water_seg.pth')
                # 保存模型状态 (如果是 DataParallel，保存 module.state_dict())
                if isinstance(model, torch.nn.DataParallel):
                    torch.save(model.module.state_dict(), save_path)
                else:
                    torch.save(model.state_dict(), save_path)
                print(f'New Best Model Saved at {save_path} with mIoU: {miou:.4f}')

def main(params=None):
    parser = argparse.ArgumentParser(description="Train BiSeNet for Water Segmentation")
    
    # --- 核心参数 ---
    parser.add_argument('--num_epochs', type=int, default=50)
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--batch_size', type=int, default=4) # 根据显存调整，4GB显存建议2或4
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--num_classes', type=int, default=2, help='水 + 非水 = 2类')
    
    # --- 数据路径配置 (新方式：直接指定路径) ---
    parser.add_argument('--images', type=str, default=None, help='训练图像目录路径')
    parser.add_argument('--masks', type=str, default=None, help='训练掩码目录路径')
    parser.add_argument('--val-images', type=str, default=None, dest='val_images', help='验证图像目录路径')
    parser.add_argument('--val-masks', type=str, default=None, dest='val_masks', help='验证掩码目录路径')
    
    # --- 数据路径配置 (旧方式：通过 data_root 自动构建) ---
    # 如果新方式未指定，则使用此方式
    parser.add_argument('--data_root', type=str, default='./my_water_data', 
                        help='数据集根目录 (仅当未直接指定 images/masks 时使用)')
    parser.add_argument('--crop_height', type=int, default=360)
    parser.add_argument('--crop_width', type=int, default=480)
    
    # --- 模型与训练配置 ---
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

    # --- 构建数据集路径 (支持新旧两种方式) ---
    # 检查是否使用了新方式（直接指定路径）
    use_direct_paths = args.images is not None or args.masks is not None
    
    if use_direct_paths:
        # 新方式：直接使用指定的路径
        # 检查必要参数是否齐全
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
        # 旧方式：通过 data_root 构建路径
        train_img_dir = os.path.join(args.data_root, 'train', 'images')
        train_mask_dir = os.path.join(args.data_root, 'train', 'masks')
        val_img_dir = os.path.join(args.data_root, 'val', 'images')
        val_mask_dir = os.path.join(args.data_root, 'val', 'masks')
        
        print(f"使用 data_root 模式: {args.data_root}")

    # 检查路径是否存在
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
    
    # 实例化数据集
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

    # 数据加载器
    dataloader_train = DataLoader(
        dataset_train, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers, 
        drop_last=True
    )
    
    dataloader_val = DataLoader(
        dataset_val, 
        batch_size=1, # 验证集通常 batch_size=1
        shuffle=False, 
        num_workers=args.num_workers
    )

    # 构建模型
    model = BiSeNet(args.num_classes, args.context_path)
    
    if torch.cuda.is_available() and args.use_gpu:
        model = torch.nn.DataParallel(model).cuda()

    # 加载预训练权重 (可选)
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
    train(args, model, optimizer, dataloader_train, dataloader_val)

if __name__ == '__main__':
    # 你可以在这里硬编码参数，或者直接在命令行运行
    # 命令行示例: python train_water.py --data_root ./my_water_data --num_epochs 50
    # 新方式示例: python train_water.py --images ./train/images --masks ./train/masks --val-images ./val/images --val-masks ./val/masks
    main()