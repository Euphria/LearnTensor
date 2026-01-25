##############################################
#  Image decomposition and reconstruction
##############################################

import os
import shutil
import torch
from PIL import Image
from torchvision import transforms

class ImgProcess:
    def __init__(self, img_path, output_dir, target_size=(1024, 1024)):
        """
        初始化图片处理器
        :param img_path: 原始图片路径
        :param output_dir: 输出目录 (会自动创建)
        :param target_size: 目标矩阵大小 (height, width)
        """
        self.img_path = img_path
        self.output_dir = output_dir
        self.target_size = target_size
        
        # 1. 创建输出目录
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"[Info] Created directory: {output_dir}")

        # 2. 定义转换管线 (Transforms)
        # 逻辑：转灰度 -> 中心裁剪(或填充)到目标尺寸 -> 转Tensor并归一化
        self.transform_pipeline = transforms.Compose([
            # 强制转为灰度图 (1通道)
            transforms.Grayscale(num_output_channels=1),
            
            # 中心裁剪：如果图片比 target_size 大，就切中间部分
            # 如果图片比 target_size 小，torchvision 会自动填充黑色背景 (padding)
            transforms.CenterCrop(target_size),
            
            # 转为 Tensor：数值会自动从 0-255 缩放到 0.0-1.0
            transforms.ToTensor() 
        ])

    def img_to_matrix(self):
        """
        将图片转换为 PyTorch 矩阵
        :return: shape 为 (H, W) 的 torch.Tensor，数值范围 [0, 1]
        """
        img = Image.open(self.img_path)
        
        # 执行转换
        img_tensor = self.transform_pipeline(img)
        
        # transform 出来的形状是 (C, H, W)，即 (1, 1024, 1024)
        # 因为你要的是二维矩阵，我们把通道维度 squeeze 掉
        matrix = img_tensor.squeeze(0)

        # 保存处理后的输入图片
        print(f"[Info] Saving processed input image...")
        self.matrix_to_img(matrix, save_name="input.png")
        
        print(f"[Process] Image -> Matrix done. Shape: {matrix.shape}, Range: [{matrix.min():.2f}, {matrix.max():.2f}]")
        return matrix

    def matrix_to_img(self, matrix, save_name="output.png"):
        """
        将矩阵还原为图片并保存
        :param matrix: torch.Tensor, shape (H, W)
        :param save_name: 保存的文件名，默认 'output.png'
        """
        # 1. 维度检查与恢复
        # 如果是 (H, W)，需要变回 (1, H, W) 才能转图片
        if matrix.ndim == 2:
            matrix = matrix.unsqueeze(0)
            
        # 2. 数值安全截断 (Clamp)
        # 防止计算过程中产生略小于0或略大于1的数导致报错
        matrix = torch.clamp(matrix, 0.0, 1.0)
        
        # 3. 转换为 PIL 图片
        to_pil = transforms.ToPILImage()
        img = to_pil(matrix) # 会自动把 0-1 映射回 0-255
        
        # 4. 保存
        save_path = os.path.join(self.output_dir, save_name)
        img.save(save_path)
        print(f"[Process] Matrix -> Image saved to: {save_path}")

# --- 使用示例 ---
if __name__ == "__main__":
    # 假设你当前目录下有一张测试图 'test.jpg'，没有的话请替换成真实路径
    # 如果找不到图片，可以先随便截个屏保存为 test.jpg
    Image_PATH = r"D:\GitStore\LearnTensor\my_script\TensorRecon\dataset\test.jpg"
    output_directory = r"D:\GitStore\LearnTensor\my_script\TensorRecon\results\test"
    # 1. 初始化
    processor = ImgProcess(
        img_path=Image_PATH,       # 替换你的图片路径
        output_dir=output_directory, # 结果保存文件夹
        target_size=(1024, 1024)   # 目标维度
    )

    # 2. 图片转矩阵 (你可以在这里进行张量分解的学习操作)
    data_matrix = processor.img_to_matrix()
    print(f"[Process] Matrix shape: {data_matrix.shape}")
    
    # [模拟训练过程] 假设这里经过了一系列张量操作...
    # 这里我们直接把原矩阵当做结果
    reconstructed_matrix = data_matrix 

    # 3. 矩阵转图片
    processor.matrix_to_img(reconstructed_matrix)