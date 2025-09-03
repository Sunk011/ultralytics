# YOLO目标检测评价工具

## 📖 简介

这是一个专门用于评价YOLO模型推理结果的Python工具，支持归一化标签格式，能够通过对比预测框与真实标注框来计算mAP等性能指标。

## ✨ 主要特性

- ✅ **支持归一化标签**: 自动处理YOLO格式的归一化坐标 (0-1范围)
- ✅ **原始图像尺寸支持**: 从原始图像中读取真实尺寸进行坐标转换
- ✅ **完整评价指标**: mAP、Precision、Recall、F1-Score、AP等
- ✅ **COCO格式mAP**: 支持计算多IoU阈值下的mAP@[0.5:0.95]
- ✅ **可视化图表**: 自动生成AP柱状图、PR曲线等
- ✅ **中文支持**: 图表和输出完全支持中文显示
- ✅ **详细日志**: 提供详细的加载和评价过程信息
- ✅ **错误处理**: 智能处理各种边缘情况和数据格式问题

## 📁 数据格式要求

### 1. 预测文件格式 (`pred_dir`)
每行格式: `class_id x_center y_center width height confidence`

```
0 0.5 0.3 0.2 0.15 0.85
1 0.7 0.6 0.1 0.2 0.92
```

### 2. 真实标注格式 (`gt_dir`)  
每行格式: `class_id x_center y_center width height`

```
0 0.52 0.28 0.18 0.16
1 0.68 0.62 0.12 0.18
```

### 3. 图像目录 (`img_dir`)
包含原始图像文件，支持格式：`.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.webp`

### 4. 类别名称文件 (可选)
JSON格式，用于显示类别名称：

```json
{
  "0": "人",
  "1": "车",
  "2": "自行车"
}
```

## 🚀 快速开始

### 安装依赖

```bash
pip install numpy matplotlib pillow opencv-python
```

### 基本使用

```bash
python yolo_evaluator.py \
    --pred_dir ./predictions \
    --gt_dir ./ground_truths \
    --img_dir ./images \
    --conf_threshold 0.25 \
    --iou_threshold 0.5 \
    --output_dir ./results \
    --plot_charts \
    --coco_map
```

### Python API使用

```python
from yolo_evaluator import YOLOEvaluator

# 创建评价器
evaluator = YOLOEvaluator(
    pred_dir="predictions",
    gt_dir="ground_truths", 
    img_dir="images",
    conf_threshold=0.25,
    iou_threshold=0.5,
    class_names={0: "人", 1: "车"}
)

# 运行评价
metrics = evaluator.run_evaluation()

# 保存结果和生成图表
evaluator.save_results("results")
evaluator.plot_class_ap_chart("results")
evaluator.plot_precision_recall_curve("results")

# 计算COCO格式mAP
coco_map = evaluator.calculate_map_at_multiple_ious()
print(f"COCO mAP: {coco_map['mAP@[0.5:0.95]']:.4f}")
```

## 📊 输出结果

### 1. 控制台输出
```
================================================================================
YOLO目标检测性能评价结果
================================================================================
IoU阈值: 0.5
置信度阈值: 0.25
总类别数: 3
mAP@0.5: 0.7234

各类别详细指标:
--------------------------------------------------------------------------------
类别ID     类别名           Precision    Recall       F1-Score     AP          
--------------------------------------------------------------------------------
0        人               0.8500       0.7800       0.8133       0.7845      
1        车               0.9200       0.8600       0.8889       0.8456      
2        自行车           0.6800       0.5400       0.6027       0.5401      
--------------------------------------------------------------------------------
平均                       0.8167       0.7267       0.7683       0.7234      
================================================================================
```

### 2. 生成的文件

- `evaluation_results.txt`: 详细的文本格式结果
- `results.json`: JSON格式的完整结果数据
- `class_ap_chart.png`: 各类别AP柱状图
- `precision_recall_curve.png`: PR曲线图
- `coco_map.json`: COCO格式的多IoU阈值mAP结果

### 3. 可视化图表

#### AP柱状图
显示每个类别的平均精度，便于对比不同类别的检测性能。

#### PR曲线
显示精确率与召回率的关系，每个类别一条曲线，包含AP数值。

## ⚙️ 参数说明

| 参数 | 说明 | 默认值 | 必需 |
|------|------|--------|------|
| `--pred_dir` | 预测结果目录路径 | - | ✅ |
| `--gt_dir` | 真实标注目录路径 | - | ✅ |
| `--img_dir` | 原始图像目录路径 | - | ✅ |
| `--conf_threshold` | 置信度阈值 | 0.25 | ❌ |
| `--iou_threshold` | IoU阈值 | 0.5 | ❌ |
| `--output_dir` | 结果保存目录 | evaluation_results | ❌ |
| `--class_names` | 类别名称JSON文件路径 | None | ❌ |
| `--plot_charts` | 生成可视化图表 | False | ❌ |
| `--coco_map` | 计算COCO格式mAP | False | ❌ |

## 🧪 测试

运行测试套件：

```bash
python test_yolo_evaluator.py
```

测试包括：
- ✅ 基本功能测试（使用模拟数据）
- ✅ 边缘情况测试（空文件、低置信度等）
- ✅ 真实数据测试（需要提供真实数据路径）

## 📁 目录结构示例

```
project/
├── images/                 # 原始图像
│   ├── img_001.jpg
│   ├── img_002.jpg
│   └── ...
├── predictions/            # 预测结果（归一化 + 置信度）
│   ├── img_001.txt
│   ├── img_002.txt
│   └── ...
├── ground_truths/          # 真实标注（归一化）
│   ├── img_001.txt
│   ├── img_002.txt
│   └── ...
├── class_names.json        # 类别名称（可选）
└── results/                # 评价结果
    ├── evaluation_results.txt
    ├── results.json
    ├── class_ap_chart.png
    ├── precision_recall_curve.png
    └── coco_map.json
```

## 🔧 故障排除

### 常见问题

1. **所有指标都是0**
   - 检查置信度阈值是否过高
   - 检查图像尺寸是否正确读取
   - 验证标签格式是否正确

2. **找不到图像文件**
   - 确保图像目录路径正确
   - 检查文件扩展名是否支持
   - 验证文件名是否与标签文件匹配

3. **坐标转换错误**
   - 确认标签是否为归一化坐标 (0-1范围)
   - 检查图像尺寸读取是否正确

4. **中文显示问题**
   - 工具会自动寻找系统中的中文字体
   - 如果没有中文字体，会使用英文显示并给出警告

### 调试技巧

```python
# 启用详细日志
evaluator = YOLOEvaluator(...)
evaluator.run_evaluation()

# 查看加载的数据统计
print(f"预测数量: {sum(len(preds) for preds in evaluator.predictions.values())}")
print(f"真实框数量: {sum(len(gts) for gts in evaluator.ground_truths.values())}")
print(f"图像数量: {len(evaluator.image_sizes)}")
```

## 🤝 贡献

欢迎提交Issue和Pull Request来改进这个工具！

## 📄 许可证

MIT License

## 👥 作者

- GitHub Copilot
- 日期: 2025-08-22

---

如有问题或建议，请随时联系！
