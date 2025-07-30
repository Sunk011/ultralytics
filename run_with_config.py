# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import argparse
import os
import sys

# 添加CrossTracker目录到系统路径
sys.path.append(os.path.abspath("./ultralytics"))
from config_loader import Config

from predict_detect import run


def main():
    # 解析命令行参数获取配置文件路径
    parser = argparse.ArgumentParser(description="Run prediction with specified config file")
    parser.add_argument("config_path", help="Path to the configuration YAML file")
    args = parser.parse_args()

    # 加载配置文件
    config = Config(args.config_path)

    # 从配置中读取参数
    model_path = config.model.path
    input_source = config.input.source
    target_class = config.detection.target_class
    save_directory = config.output.save_directory
    confidence_threshold = config.detection.confidence_threshold
    display_window = config.output.display_window
    device = config.model.device

    # 调用原有run函数并传入参数
    run(
        MODEL_PATH=model_path,
        INPUT_SOURCE=input_source,
        TARGET_CLASS_TO_CROP=target_class,
        CROP_SAVE_DIRECTORY=save_directory,
        CONFIDENCE_THRESHOLD=confidence_threshold,
        DISPLAY_REALTIME_WINDOW=display_window,
        device=device,
    )


if __name__ == "__main__":
    main()
