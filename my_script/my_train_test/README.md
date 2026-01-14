# LearnTensor 项目

本项目基于量子计算和张量网络实现量子态学习。

## 依赖项目

该项目依赖于 [tneq-qc](https://github.com/minogame/tneq-qc) 项目。

## 运行方法

训练模型:
```bash
python .\src\train.py
```

## 项目结构

```
tneq_train/
├── .gitignore                  # Git 忽略文件配置 
├── README.md                   # 项目说明文档
├── config/                     # 配置文件目录
│   └── config.yaml             # 主配置文件
├── data/                       # 数据集目录
│   ├── create_data.py          # 数据生成脚本
│   └── data_10_10_3.h5         # 训练数据文件
├── datamodules/                # 数据模块
│   ├── data_preprocess.py      # 数据预处理
│   ├── load_data.py            # 数据加载逻辑
│   └── __init__.py
├── post_processing/            # 后处理模块
│   ├── fingure_loss.py         # 损失函数可视化
│   ├── sample.py               # 采样
│   └── __init__.py
└── src/                        # 源代码
    ├── circuit_states.py       # 电路状态处理
    ├── load_config.py          # 配置加载工具
    └── train.py                # 核心训练逻辑
```

## 功能模块

- **data**:             数据集
- **config**:           配置文件
- **datamodules**:      数据加载和预处理
- **post_processing**:  训练结果可视化和后处理
- **src**:              核心训练逻辑和电路状态处理
- **assets**:           存储训练过程中的模型权重、损失曲线等结果
