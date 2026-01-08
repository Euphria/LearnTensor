import re
import matplotlib.pyplot as plt
from pathlib import Path

def fingure_loss(save_dir: Path,
                log_path: str = "output.log"):
    """
    从指定的 save_dir 目录中读取 tmp.log 并生成 loss.pdf
    :param save_dir: 保存实验结果的目录 (pathlib.Path 或 str)
    :param log_path: 日志文件路径，默认为 "output.log"
    """
    log_path = save_dir / log_path  # 匹配 train.py 中定义的 log_path 名
    pdf_path = save_dir / "loss.pdf"
    
    if not log_path.exists():
        print(f"[Post Error] 在目录中找不到日志文件: {log_path}")
        return

    iterations = []
    losses = []
    
    # 读取日志内容
    with open(log_path, 'r', encoding='utf-8') as f:
        content = f.read()
        
    # 正则表达式匹配 Iteration 和 loss 值 (兼容科学计数法)
    pattern = r"Iteration\s+(\d+):\s+loss\s+=\s+([+-]?\d*\.?\d+(?:[eE][+-]?\d+)?)"
    matches = re.findall(pattern, content)
    
    for it, loss in matches:
        iterations.append(int(it))
        losses.append(float(loss))
        
    if not iterations:
        print(f"[Post Warning] 日志中未提取到 Loss 数据，请检查日志格式。")
        return

    # 绘图逻辑
    plt.figure(figsize=(8, 5))
    plt.plot(iterations, losses, marker='o', markersize=3, label='Training Loss', color='#D62728')
    
    plt.xlabel('Iteration')
    plt.ylabel('Loss Value')
    plt.title(f'Training Loss Curve ({save_dir.parent.name})')
    plt.grid(True, linestyle=':', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    
    # 保存图像
    plt.savefig(pdf_path)
    plt.close()
    print(f"[Post] Loss 曲线已保存至: {pdf_path}")

if __name__ == "__main__":
    log_path = "output.log"
    save_dir = Path("D:/GitStore/LearnTensor/my_script/my_train_test/assets/data_10_10_3/20260108_1605")  # 替换为实际的保存目录
    fingure_loss(log_path, save_dir)