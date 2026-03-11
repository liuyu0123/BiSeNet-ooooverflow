import csv
import argparse
from pathlib import Path

def generate_class_dict(output_path):
    # 定义类别
    # 格式: name, r, g, b, class_id (原代码中 class_11 列似乎用于标记是否参与计算，通常设为 1)
    # 注意：原 CamVid 代码逻辑中，最后一列通常是 'class' 或类似标记，这里根据你提供的上下文设为 1
    
    classes = [
        {"name": "non_water", "r": 0, "g": 0, "b": 0, "id": 1},   # 背景/非水 (黑色)
        {"name": "water", "r": 255, "g": 255, "b": 255, "id": 1}  # 水 (白色)
    ]
    
    # 如果你的 mask 是其他颜色，请修改上面的 r, g, b 值
    # 例如水是蓝色: {"name": "water", "r": 0, "g": 0, "b": 255, "id": 1}

    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', newline='') as csvfile:
        fieldnames = ['name', 'r', 'g', 'b', 'class_11']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for cls in classes:
            writer.writerow({
                'name': cls['name'],
                'r': cls['r'],
                'g': cls['g'],
                'b': cls['b'],
                'class_11': cls['id']
            })

    print(f"class_dict.csv 已生成至: {output_file}")
    print("内容预览:")
    print("name,r,g,b,class_11")
    for cls in classes:
        print(f"{cls['name']},{cls['r']},{cls['g']},{cls['b']},{cls['id']}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate class_dict.csv for BiSeNet")
    parser.add_argument('--output_path', type=str, default='./class_dict.csv', help='输出文件路径')
    args = parser.parse_args()
    
    generate_class_dict(args.output_path)