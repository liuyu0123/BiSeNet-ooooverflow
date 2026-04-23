import argparse
import os
import torch
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from model.build_BiSeNet import BiSeNet
from utils import reverse_one_hot

def get_transform(scale):
    return transforms.Compose([
        transforms.Resize((scale[0], scale[1])),
        transforms.ToTensor(),
        # 如果训练时用了 Normalize，这里也要加上相同的参数
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

def infer_single_image(model, img_path, save_path, scale, use_gpu):
    # 读取图片
    image = Image.open(img_path).convert('RGB')
    original_w, original_h = image.size
    
    # 预处理
    transform = get_transform(scale)
    input_tensor = transform(image).unsqueeze(0) # [1, C, H, W]
    
    if use_gpu:
        input_tensor = input_tensor.cuda()
    
    # 推理
    model.eval()
    with torch.no_grad():
        outputs = model(input_tensor)
        predict = outputs[0]
        predict = reverse_one_hot(predict)
        predict_np = predict.squeeze().cpu().numpy() # [H, W] values 0 or 1
    
    # 后处理：将预测结果 Resize 回原图尺寸
    predict_img = Image.fromarray((predict_np * 255).astype(np.uint8))
    predict_img = predict_img.resize((original_w, original_h), resample=Image.NEAREST)
    predict_np_orig = np.array(predict_img)
    
    # 可视化：创建叠加图
    # 策略：原图作为背景，将预测为水(1)的区域覆盖上一层半透明红色
    overlay = image.copy()
    pixels = overlay.load()
    mask_pixels = predict_np_orig
    
    # 简单的叠加逻辑：如果是水，变红
    for y in range(original_h):
        for x in range(original_w):
            if mask_pixels[y, x] > 0: # 是水
                # 混合：50% 原色 + 50% 红色 (255, 0, 0)
                r, g, b = pixels[x, y][:3]
                pixels[x, y] = (int(r * 0.5 + 255 * 0.5), int(g * 0.5), int(b * 0.5))
    
    # 保存结果
    os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
    overlay.save(save_path)
    print(f"Saved result to: {save_path}")

def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.cuda
    use_gpu = torch.cuda.is_available() and args.use_gpu
    
    # 加载模型
    model = BiSeNet(args.num_classes, args.context_path)
    if use_gpu:
        model = torch.nn.DataParallel(model).cuda()
    
    print(f"Loading model from {args.model_path}...")
    if use_gpu:
        model.module.load_state_dict(torch.load(args.model_path))
    else:
        model.load_state_dict(torch.load(args.model_path, map_location='cpu'))
    
    model.eval()
    print("Model loaded successfully.")

    # 确定输入源
    input_paths = []
    if os.path.isfile(args.input_path):
        input_paths.append(args.input_path)
    elif os.path.isdir(args.input_path):
        valid_exts = ('.jpg', '.jpeg', '.png', '.bmp')
        input_paths = [os.path.join(args.input_path, f) for f in os.listdir(args.input_path) if f.lower().endswith(valid_exts)]
        print(f"Found {len(input_paths)} images in folder.")
    else:
        print(f"Error: Input path {args.input_path} does not exist.")
        return

    # 批量推理
    for img_path in input_paths:
        filename = os.path.basename(img_path)
        save_name = f"water_seg_{filename}"
        save_path = os.path.join(args.output_dir, save_name)
        
        print(f"Processing: {filename}...")
        try:
            infer_single_image(model, img_path, save_path, (args.crop_height, args.crop_width), use_gpu)
        except Exception as e:
            print(f"Failed to process {filename}: {e}")

    print("All done!")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Demo / Inference for Water Segmentation")
    
    parser.add_argument('--model_path', type=str, required=True, help='Path to trained model')
    parser.add_argument('--input_path', type=str, required=True, help='Path to a single image or a folder of images')
    parser.add_argument('--output_dir', type=str, default='./demo_results', help='Directory to save results')
    
    parser.add_argument('--num_classes', type=int, default=2)
    parser.add_argument('--context_path', type=str, default='resnet18')
    parser.add_argument('--crop_height', type=int, default=360)
    parser.add_argument('--crop_width', type=int, default=480)
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--cuda', type=str, default='0')
    
    args = parser.parse_args()
    main(args)