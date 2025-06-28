import cv2
import os
from natsort import natsorted

def images_to_video(img_folder, output_path, fps=10):
    # 获取所有jpg图片并排序
    img_files = [f for f in os.listdir(img_folder) if f.endswith('.jpg')]
    img_files = natsorted(img_files)  # 按文件名自然排序

    if not img_files:
        print("没有找到jpg图片")
        return

    # 读取第一张图片确定尺寸
    first_img = cv2.imread(os.path.join(img_folder, img_files[0]))
    height, width, layers = first_img.shape

    # 定义视频编码器和输出文件
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    for img_file in img_files:
        img_path = os.path.join(img_folder, img_file)
        img = cv2.imread(img_path)
        if img is None:
            print(f"跳过无法读取的图片: {img_file}")
            continue
        video.write(img)

    video.release()
    print(f"视频已保存到: {output_path}")


# 用法示例
path = '../results_oppo/LMs-deepseek-chat/run_1/0/top_down_image/'
images_to_video(path, 'output1.mp4', fps=60)