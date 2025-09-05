import os
import shutil
from argparse import ArgumentParser

def copy_images(source_dir, target_dir="data/input", max_files=None):
    """
    将图像文件从源目录复制到目标目录
    
    Args:
        source_dir: 源图像目录路径
        target_dir: 目标目录路径(默认data/input)
        max_files: 最大复制文件数量(None表示全部)
    """
    # 确保目标目录存在
    os.makedirs(target_dir, exist_ok=True)
    
    # 获取所有图像文件
    image_exts = ('.jpg', '.jpeg', '.png')
    files = [f for f in os.listdir(source_dir) 
            if f.lower().endswith(image_exts)]
    
    # 限制文件数量
    if max_files is not None:
        files = files[:max_files]
    
    # 复制文件
    for i, filename in enumerate(files, 1):
        for i in range(1, max_files):
            src = os.path.join(source_dir, filename)
            dst = os.path.join(target_dir, filename.split(".")[0] + "_" + str(i)+"."+filename.split(".")[1])
            shutil.copy2(src, dst)
            print(f"[{i}/{len(files)}] Copied {filename} to {filename.split('.')[0]}_{i}.{filename.split('.')[1]}")

        

if __name__ == '__main__':
    parser = ArgumentParser(description="图像文件复制工具")
    parser.add_argument("source_dir", help="源图像目录路径")
    parser.add_argument("--target_dir", default="data/input", 
                      help="目标目录路径(默认data/input)")
    parser.add_argument("--max_files", type=int, default=None,
                      help="最大复制文件数量")
    
    args = parser.parse_args()
    copy_images(args.source_dir, args.target_dir, args.max_files)
