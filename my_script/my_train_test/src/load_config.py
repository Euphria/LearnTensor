import yaml
import os

def load_config(config_path: str) -> dict:
    """
    从 YAML 文件加载实验配置。
    """
    if not os.path.exists(config_path):
        raise ValueError(f"找不到配置文件: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    print(f"[Config] 成功加载配置文件: \t{os.path.abspath(config_path)}")
    return config

if __name__ == "__main__":
    # 测试代码
    config_path = "config.yaml"
    cfg = load_config(config_path=config_path)
    print(f"数据路径为: {cfg['data_path']}")