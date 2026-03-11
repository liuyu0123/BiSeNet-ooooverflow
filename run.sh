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
python train_water.py --data_root ./dataset/water_seg2 --num_epochs 5 --batch_size 4 --context_path resnet18

#模型评估
python eval_water.py --model_path ./checkpoints_water/best_water_seg.pth --data_root ./dataset/water_seg --split val

#模型预测(单张图片推理)
python demo_water.py --model_path ./checkpoints_water/best_water_seg.pth --input_path ./img/H05_1_0000000000.jpg --output_dir ./demo_results
#模型预测(整个文件夹)
python demo_water.py --model_path ./checkpoints_water/best_water_seg.pth --input_path ./img --output_dir ./demo_results