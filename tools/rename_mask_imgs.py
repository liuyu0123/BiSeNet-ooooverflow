#!/usr/bin/env python3
"""
批量重命名 mask 文件脚本
将 H05_1_0000000000_mask.png 重命名为 H05_1_0000000000.png
"""

import os
import re
import sys
from pathlib import Path


def rename_mask_files(directory):
    """
    将指定目录下所有 *_mask.png 文件重命名为 *.png
    
    Args:
        directory: 目标目录路径
    """
    directory = Path(directory).resolve()
    
    if not directory.exists():
        print(f"❌ 错误：目录不存在 {directory}")
        return False
    
    if not directory.is_dir():
        print(f"❌ 错误：{directory} 不是目录")
        return False
    
    # 查找所有 *_mask.png 文件
    pattern = re.compile(r'^(.*)_mask\.png$', re.IGNORECASE)
    mask_files = []
    
    for file_path in directory.iterdir():
        if file_path.is_file():
            match = pattern.match(file_path.name)
            if match:
                mask_files.append(file_path)
    
    if not mask_files:
        print(f"ℹ️ 在 {directory} 中未找到 *_mask.png 文件")
        return True
    
    print(f"📁 扫描目录: {directory}")
    print(f"🔍 找到 {len(mask_files)} 个需要重命名的文件\n")
    
    success_count = 0
    skip_count = 0
    error_count = 0
    
    for file_path in mask_files:
        old_name = file_path.name
        # 去掉 _mask 后缀
        new_name = old_name.replace('_mask.png', '.png')
        new_path = directory / new_name
        
        # 检查目标文件是否已存在
        if new_path.exists():
            print(f"⚠️ 跳过: {old_name} -> 目标文件 {new_name} 已存在")
            skip_count += 1
            continue
        
        try:
            file_path.rename(new_path)
            print(f"✅ 重命名: {old_name} -> {new_name}")
            success_count += 1
        except Exception as e:
            print(f"❌ 失败: {old_name} -> 错误: {e}")
            error_count += 1
    
    print(f"\n📊 完成统计:")
    print(f"   成功: {success_count}")
    print(f"   跳过: {skip_count}")
    print(f"   失败: {error_count}")
    
    return error_count == 0


def main():
    """主函数"""
    # 获取命令行参数或使用交互式输入
    if len(sys.argv) > 1:
        target_dir = sys.argv[1]
    else:
        target_dir = input("请输入目标文件夹路径: ").strip().strip('"')
    
    # 移除可能存在的引号
    target_dir = target_dir.strip("'\"")
    
    if not target_dir:
        print("❌ 错误：未提供目录路径")
        print("用法: python rename_mask.py <目录路径>")
        sys.exit(1)
    
    # 执行重命名
    success = rename_mask_files(target_dir)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()