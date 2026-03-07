"""
Description: 递归遍历源目录下所有子文件夹，拷贝所有PDF文件到目标目录
Author: [Your Name]
Date: 2024-10-24

*** 不要轻易运行这个文件，请先确认源目录和目标目录路径正确，并且有足够的权限进行文件操作。运行前建议备份重要数据，以防止意外情况发生。***
"""


import os
import shutil
from pathlib import Path

def copy_all_pdfs(source_dir, target_dir):
    """
    递归遍历源目录下所有子文件夹，拷贝所有PDF文件到目标目录
    
    Args:
        source_dir (str): 源目录路径（包含多层子文件夹的PDF目录）
        target_dir (str): 目标目录路径（存放所有PDF的文件夹）
    """
    # 确保目标目录存在，不存在则创建
    Path(target_dir).mkdir(parents=True, exist_ok=True)
    
    # 统计变量
    total_files = 0  # 总处理文件数
    success_files = 0  # 成功拷贝数
    error_files = []  # 拷贝失败的文件列表
    
    # 递归遍历所有文件
    for root, dirs, files in os.walk(source_dir):
        for file in files:
            # 只处理PDF文件（不区分大小写）
            if file.lower().endswith('.pdf'):
                total_files += 1
                # 源文件完整路径
                source_file = os.path.join(root, file)
                # 目标文件路径（先默认同名）
                target_file = os.path.join(target_dir, file)
                
                # 处理文件名重复：如果已存在，自动加序号（如 file_1.pdf, file_2.pdf）
                file_name, file_ext = os.path.splitext(file)
                counter = 1
                while os.path.exists(target_file):
                    target_file = os.path.join(target_dir, f"{file_name}_{counter}{file_ext}")
                    counter += 1
                
                try:
                    # 拷贝文件（保留文件元数据）
                    shutil.copy2(source_file, target_file)
                    success_files += 1
                    # 每处理100个文件打印进度，避免刷屏
                    if success_files % 100 == 0:
                        print(f"已成功拷贝 {success_files} 个PDF文件...")
                except Exception as e:
                    # 捕获所有异常，记录失败文件并继续
                    error_info = f"文件 {source_file} 拷贝失败：{str(e)}"
                    error_files.append(error_info)
                    print(error_info)
    
    # 打印最终统计结果
    print("\n=== 拷贝完成 ===")
    print(f"总扫描到PDF文件数：{total_files}")
    print(f"成功拷贝数：{success_files}")
    print(f"失败数：{len(error_files)}")
    if error_files:
        print("\n失败文件列表：")
        for err in error_files:
            print(err)

if __name__ == "__main__":
    # 配置源目录和目标目录（请确认路径正确）
    SOURCE_DIR = r"G:\ref-endnote\subrefs-2022-2026.Data\PDF"
    TARGET_DIR = r"G:\ref-endnote\refsPDF2022-2026"
    
    # 验证源目录是否存在
    if not os.path.exists(SOURCE_DIR):
        print(f"错误：源目录 {SOURCE_DIR} 不存在！")
    else:
        print(f"开始从 {SOURCE_DIR} 拷贝所有PDF到 {TARGET_DIR}...")
        copy_all_pdfs(SOURCE_DIR, TARGET_DIR)