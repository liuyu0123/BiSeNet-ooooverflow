#模型训练
python train.py

#模型性能评估
python eval.py

#模型预测(单张图片推理)
python demo.py

######################## 水域分割 ########################
#Mask批量重命名
python ./tools/rename_mask_imgs.py ./dataset/water_seg2/masks

#数据集预处理(划分)
python ./tools/split_dataset.py `
    --root_dir ./dataset/water_seg  `
    --train_ratio 0.8  `
    --val_ratio 0.1

#数据集csv标注生成
python ./tools/generate_class_dict.py  --output_path ./dataset/water_seg/class_dict.csv

#模型训练
python train_water.py --data_root ./dataset/water_seg --num_epochs 5 --batch_size 4 --context_path resnet18
#使用数据集2(黑白的mask, 实际上使用黑红的mask就能够训练成功并且成功eval和demo)
python train_water.py --data_root ./dataset/water_seg2 --num_epochs 5 --batch_size 4 --context_path resnet18
#模型训练（水域分割，train与val分离）✅
python train_water_val.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_white_noSuffix `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_white_noSuffix `
    --num_epochs 5 `
    --batch_size 4 `
    --context_path resnet18

#模型训练(pro)
python train_water_val_pro.py `
    --images D:\Files\Data\IRWSB\train\images `
    --masks D:\Files\Data\IRWSB\train\masks_red `
    --val-images D:\Files\Data\IRWSB\val\images `
    --val-masks D:\Files\Data\IRWSB\val\masks_red `
    --epochs 5 `
    --batch-size 4 `
    --learning-rate 5e-4 `
    --model-dir checkpoints/experiment1 `
    --log-dir logs/experiment1 `
    --model-name experiment1 `
    --log-name experiment1 `
    --save-interval 0

#模型评估
python eval_water.py --model_path ./checkpoints_water/best_water_seg.pth --data_root ./dataset/water_seg --split val
#模型评估（水域分割，test路径输入）
python eval_water_val.py `
    --model_path ./checkpoints_water/best_water_seg.pth `
    --test-images D:\Files\Data\IRWSB\test\images `
    --test-masks D:\Files\Data\IRWSB\test\masks_white_noSuffix `
    --save_pred_path ./eval_results `
    --output checkpoints_water/test_result.csv

#模型预测(单张图片推理)
python demo_water.py --model_path ./checkpoints_water/best_water_seg.pth --input_path ./img/H05_1_0000000000.jpg --output_dir ./demo_results
#模型预测(整个文件夹)
python demo_water.py --model_path ./checkpoints_water/best_water_seg.pth --input_path ./img --output_dir ./demo_results


#模型推理pro（生成红色mask蒙版和csv评价指标）
#单图推理（无真值，仅保存可视化）：
python eval_water_val_pro.py `
    --model_path ./checkpoints/water_model.pth `
    --input ./test_images/sample1.jpg `
    --output ./eval_results_pro `
    --alpha 0.5

#文件夹批量推理（有真值，生成CSV）：
python eval_water_val_pro.py `
    --model_path "F:\AAA\9_bisenet_best\experiment1\experiment1_last.pth" `
    --input "D:\Files\Data\IRWSB\analyse\images" `
    --ground_truth "D:\Files\Data\IRWSB\analyse\masks_white_noSuffix" `
    --output ./eval_results_pro `
    --crop_height 360 `
    --crop_width 480

# 仅打印结果（不保存文件）：
python eval_water_val_pro.py `
    --model_path ./checkpoints/water_model.pth `
    --input "D:\Files\Data\IRWSB\analyse\images" `
    --ground_truth "D:\Files\Data\IRWSB\analyse\masks_white_noSuffix"
