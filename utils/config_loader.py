"""
CrossTracker 配置加载器
用于加载和管理 YAML 配置文件
"""

import yaml
import os
from pathlib import Path
from typing import Any, Dict


class ConfigSection:
    """配置节类 - 支持动态属性访问"""
    
    def __init__(self, config_dict: Dict[str, Any]):
        self._config = config_dict
    
    def __getattr__(self, name: str) -> Any:
        """动态获取属性"""
        # 先尝试原始键名
        if name in self._config:
            value = self._config[name]
            # 如果值是字典，返回新的 ConfigSection 实例
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        # 尝试大写键名（向后兼容）
        upper_name = name.upper()
        if upper_name in self._config:
            value = self._config[upper_name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        # 尝试小写键名
        lower_name = name.lower()
        if lower_name in self._config:
            value = self._config[lower_name]
            if isinstance(value, dict):
                return ConfigSection(value)
            return value
        
        raise AttributeError(f"配置项 '{name}' 不存在")
    
    def get(self, key: str, default: Any = None) -> Any:
        """安全获取配置值"""
        try:
            return getattr(self, key)
        except AttributeError:
            return default


class Config:
    """配置管理类"""
    
    def __init__(self, config_path: str = "config.yaml"):
        """
        初始化配置
        
        Args:
            config_path: 配置文件路径
        """
        self.config_path = config_path
        self._config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """加载配置文件"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"配置文件未找到: {self.config_path}")
        
        with open(self.config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def reload(self):
        """重新加载配置文件"""
        self._config = self._load_config()
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        获取配置值，支持点号路径
        
        Args:
            key_path: 配置路径，如 'input_handler.buffer_size'
            default: 默认值
            
        Returns:
            配置值
        """
        keys = key_path.split('.')
        value = self._config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default
    
    def __getattr__(self, name: str) -> ConfigSection:
        """动态获取配置节"""
        if name in self._config:
            return ConfigSection(self._config[name])
        raise AttributeError(f"配置节 '{name}' 不存在")

# 全局配置实例


#main
if __name__ == "__main__":
    config = Config('config_camera1.yaml')
    # from config_loader import config

    # 测试配置访问
    print("=== 新的配置访问方式 ===")

    # 1. 直接访问配置节
    print(f"输入源: {config.input_handler.input_source}")
    print(f"缓冲区大小: {config.input_handler.buffer_size}")
    print(f"模型路径: {config.ai.model_path}")
    print(f"置信度阈值: {config.ai.confidence_threshold}")

    # 2. 访问嵌套配置
    print(f"视频保存启用: {config.output_handler.video_save.enable}")
    print(f"视频保存路径: {config.output_handler.video_save.path}")
    print(f"RTSP推流启用: {config.output_handler.rtsp_push.enable}")
    print(f"RTSP输出URL: {config.output_handler.rtsp_push.output_url}")

    # 3. 使用 get 方法安全访问（支持默认值）
    print(f"不存在的配置项: {config.input_handler.get('non_existent', 'default_value')}")

    # 4. 使用点号路径访问
    print(f"使用点号路径: {config.get('input_handler.input_source')}")
    print(f"使用点号路径（嵌套）: {config.get('output_handler.video_save.enable')}")

    print("\n=== 优势对比 ===")
    print("✅ 添加新配置项时，无需修改代码")
    print("✅ 支持任意深度的嵌套配置")
    print("✅ 自动类型推断")
    print("✅ 支持默认值")
    print("✅ 向后兼容（支持大小写变换）")
