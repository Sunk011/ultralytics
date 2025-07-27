# from ultralytics import YOLO

# # Load a model
# # model = YOLO("yolo11n-cls.yaml")  # build a new model from YAML
# # model = YOLO("yolo11n-cls.pt")  # load a pretrained model (recommended for training)
# model = YOLO("yolo11n-cls.yaml").load("/home/sk/project/jg_project/v001/weights/best.pt")  # build from YAML and transfer weights

# print("loaded model pth")
# # Train the model
# # results = model.train(data="mnist160", epochs=100, imgsz=64)


from ultralytics import YOLO
import torch

def main():
    # 检查GPU
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")

    # 1. 加载一个预训练的【分类】模型
    #    注意：后缀是 '-cls.pt'，代表这是分类模型！
    #    从分类预训练模型开始，效果远好于从检测模型开始。
    # model = YOLO('yolov8s-cls.pt')
    model = YOLO("yolo11m-cls.yaml").load("/home/sk/project/jg_project/car_vis/epoch120_v1/weights/best.pt")  # build from YAML and transfer weights
    

    # 2. 开始训练分类模型
    print("开始训练分类模型...")
    results = model.train(
        data='/home/sk/project/datasets/yolo_dataset_bus_cls_dataset', # 【核心】直接指向数据集的根目录！
        epochs=100,                  # 分类任务通常收敛更快，可以从50轮开始
        imgsz=224,                  # 分类模型常用尺寸，如 224x224
        batch=256,                   # 可以设置得比检测任务更大
        project='jg_project',  # 项目文件夹
        name='cls/test_yolo11m',  # 实验名称
        
        verbose=True,  # 打印详细日志
        patience=0,  # 早停轮次（15轮无提升则停止）
        
        flipud=0,   # 上下翻转概率
        fliplr=0.8,   # 左右翻转概率
        mosaic=1.0,   # 马赛克增强概率（1.0表示启用）
        
        # device= '4, 5, 6, 7'
        # device= '0, 5, 6, 7'
        device= '6, 7'
    )
    print("分类模型训练完成！")
    
    # # 3. (可选) 在验证集上评估模型
    # # 训练结束后会自动评估，这里是手动调用的方式
    # metrics = model.val()
    # print(f"Top-1 准确率: {metrics.top1}")
    # print(f"Top-5 准确率: {metrics.top5}")

if __name__ == '__main__':
    main()