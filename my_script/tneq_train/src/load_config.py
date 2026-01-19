import yaml
import os
import re

def convert_scientific_notation(data):
    """
    递归遍历字典或列表，将匹配科学计数法格式的字符串（如 1e-6）转换为浮点数。
    """
    if isinstance(data, dict):
        return {k: convert_scientific_notation(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [convert_scientific_notation(i) for i in data]
    elif isinstance(data, str):
        # 使用正则表达式匹配科学计数法格式: 如 1e-6, 1.0E+5, -2.3e-4
        if re.fullmatch(r"^[+-]?\d+(\.\d+)?[eE][+-]?\d+$", data.strip()):
            try:
                return float(data)
            except ValueError:
                return data
    return data
def load_config(config_path: str) -> dict:
    """
    从 YAML 文件加载实验配置。
    """
    if not os.path.exists(config_path):
        raise ValueError(f"找不到配置文件: {config_path}")
        
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
        
    config = convert_scientific_notation(config)
        
    print(f"[Config] 成功加载配置文件: \t{os.path.abspath(config_path)}")
    print(f'[Config] 配置文件内容: \n{config["training_params"]["lr_schedule"]}')
    return config

if __name__ == "__main__":
    # 测试代码
    config_path = "config.yaml"
    cfg = load_config(config_path=config_path)
    print(f"数据路径为: {cfg['data_path']}")