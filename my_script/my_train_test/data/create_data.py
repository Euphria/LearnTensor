import h5py
import numpy as np

# 核心生成函数（可传参，参数灵活配置）
def create_10x10x3_hdf5(file_path, ds_name, shape, min_val, max_val):
    # 1. 生成指定维度的随机整数
    random_data = np.random.randint(min_val, max_val + 1, size=shape, dtype=np.int32)
    # 2. 创建并写入HDF5文件
    with h5py.File(file_path, 'w') as f:
        f.create_dataset(ds_name, data=random_data)
    print(f"✅ HDF5数据集创建完成！\n📂 文件路径：{file_path}\n📊 数据集名：{ds_name}\n📐 数据维度：{shape}")

# 主程序入口：定义参数 + 调用函数
if __name__ == "__main__":
    # ===================== 在这里配置所有参数 =====================
    SAVE_FILE_PATH = r"D:\GitStore\LearnTensor\my_script\my_train_test\data\data_10_10_3.h5"      # 输出h5文件路径
    DATASET_NAME = "train_data"             # 数据集名称
    DATA_SHAPE = (10, 10, 3)                # 核心维度10*10*3
    RAND_INT_MIN = 0                        # 随机整数最小值
    RAND_INT_MAX = 1                        # 随机整数最大值
    # ==============================================================
    
    # 调用函数，传入上述所有参数
    create_10x10x3_hdf5(SAVE_FILE_PATH, DATASET_NAME, DATA_SHAPE, RAND_INT_MIN, RAND_INT_MAX)