#模型训练
python train.py

#模型性能评估
python eval.py

#模型预测(单张图片推理)
python demo.py

######################## 水域分割 ########################
#数据集预处理(划分)
python ./tools/split_dataset.py `
    --root_dir ./dataset/water_seg  `
    --train_ratio 0.8  `
    --val_ratio 0.1