# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license

import torch

from ultralytics import YOLO


def verify_environment_and_train():
    """一个简单的脚本，用于验证Ultralytics环境并运行一个快速的训练任务。."""
    print("--- 环境验证开始 ---")

    # 1. 验证PyTorch和CUDA
    try:
        print(f"PyTorch 版本: {torch.__version__}")
        cuda_available = torch.cuda.is_available()
        print(f"CUDA 是否可用: {cuda_available}")
        if cuda_available:
            device_count = torch.cuda.device_count()
            print(f"找到 {device_count} 个GPU设备。")
            current_device = torch.cuda.current_device()
            print(f"当前设备: {torch.cuda.get_device_name(current_device)}")
            print(f"CUDA 版本 (PyTorch内置): {torch.version.cuda}")
        print("PyTorch 和 CUDA 验证完毕。")
    except Exception as e:
        print(f"!!! PyTorch或CUDA验证失败: {e}")
        return

    # 2. 验证Ultralytics库的导入
    try:
        # 这一步已经隐式地测试了OpenCV的导入
        model = YOLO("/home/sk/project/ultralytics/yolo11m-obj365-640-Pretrain.pt")

        # 3. 运行一个非常短的训练任务
        # results = model.train(
        #     data='/home/sk/project/datasets/car-detection/car-detection.yaml',
        #     epochs=150,
        #     imgsz=640,
        #     project='jg_project',
        #     name='v001',
        #     exist_ok=True,  # 允许覆盖之前的运行结果
        #     device='4, 5, 6, 7'
        # )

        model.train(
            # 基础参数
            # data='/home/sk/project/datasets/car-detection/car-detection.yaml',  # 数据集配置文件
            # data='VisDrone.yaml',  # 数据集配置文件
            data="/home/sk/project/datasets/car_vis/car_vis.yaml",  # 数据集配置文件
            # model='/home/sk/project/ultralytics/yolo11m-obj365-640-Pretrain.pt',  # 预训练权重（推荐使用较大模型，4090算力足够支撑）
            epochs=150,  # 训练轮次（结合早停机制，无需过大）
            imgsz=1024,  # 输入图像尺寸（YOLO默认，兼顾精度和速度）
            # device='4,5,6,7',  # 指定4张GPU
            device="0,5,6,7",  # 指定4张GPU
            # 批处理参数（关键优化）
            batch=32,  # 总batch_size（4张卡均分，单卡16，24G显存足够）
            # 若出现OOM，可降至32（单卡8），但64更高效
            # 学习率与优化器（多卡适配）
            # lr0=0.01,  # 初始学习率（4卡训练可适当提高，默认0.01适合单卡，多卡可保持或略增）
            # lrf=0.01,  # 最终学习率因子（lr0 * lrf）
            # optimizer='AdamW',  # 优化器（AdamW在小样本上更稳定，SGD适合大样本）
            # 数据增强（根据数据集复杂度调整）
            augment=True,  # 启用默认增强策略
            # hsv_h=0.015,  # HSV色调增强幅度（0-1）
            # hsv_s=0.7,    # 饱和度增强幅度
            # hsv_v=0.4,    # 明度增强幅度
            # degrees=10.0,  # 旋转角度（0-180）
            flipud=0,  # 上下翻转概率
            fliplr=0,  # 左右翻转概率
            mosaic=1.0,  # 马赛克增强概率（1.0表示启用）
            erasing=0.5,
            iou=0.5,
            # 正则化与早停（防止过拟合）
            # weight_decay=0.0005,  # 权重衰减
            patience=0,  # 早停轮次（15轮无提升则停止）
            save_period=10,  # 每10轮保存一次模型
            # 输出配置
            project="jg_project",  # 项目文件夹
            name="car_vis/epoch120_v1",  # 实验名称
            # exist_ok=True,  # 允许覆盖现有结果
            save=True,  # 保存模型
            val=True,  # 每轮训练后验证
            verbose=True,  # 打印详细日志
        )

        print("\n--- 训练任务成功完成！ ---")
        # print(f"训练结果保存在: {results.save_dir}")

    except Exception as e:
        print(f"\n!!! 训练过程中发生错误: {e}")
        # 打印更详细的追溯信息，便于调试
        import traceback

        traceback.print_exc()


if __name__ == "__main__":
    verify_environment_and_train()
