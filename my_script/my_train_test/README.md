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
LearnTensor/
├── .gitignore
├── config.yaml                 # 主配置文件
├── README.md
├── assets/                     # 训练结果存储目录
├── data/                       # 数据集目录
│   ├── create_data.py
│   └── data_10_10_3.h5
├── datamodules/                # 数据模块
│   ├── data_preprocess.py
│   ├── load_data.py
│   └── __init__.py
├── post_processing/            # 后处理模块
│   ├── fingure_loss.py
│   ├── general.py
│   └── __init__.py
└── src/                        # 源代码
    ├── circuit_states.py
    ├── load_config.py
    ├── train.py
    └── __init__.py
```

## 功能模块

- **datamodules**: 数据加载和预处理
- **post_processing**: 训练结果可视化和后处理
- **src**: 核心训练逻辑和电路状态处理
- **assets**: 存储训练过程中的模型权重、损失曲线等结果
