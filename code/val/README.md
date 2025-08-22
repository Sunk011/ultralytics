# YOLO目标检测评价工具

## 功能说明
本工具用于评价YOLO模型的推理结果，通过对比预测框与真实标注框来计算mAP等性能指标。

## 支持的指标
- **mAP (mean Average Precision)**: 平均精度均值
- **Precision**: 精确率
- **Recall**: 召回率 
- **F1-Score**: F1分数
- **AP (Average Precision)**: 各类别的平均精度

## 新增功能 ✨
- **中文可视化支持**: 图表标题、标签、图例完全支持中文显示
- **多种图表类型**: AP柱状图、PR关系图、雷达图、类别对比图
- **HTML报告**: 生成美观的中文HTML评价报告
- **自定义类别名称**: 支持加载中文类别名称文件

## 输入格式

### 预测结果格式 (YOLO格式)
每个txt文件对应一张图像，格式为：
```
class_id x_center y_center width height confidence
```

### 真实标注格式 (YOLO格式)
每个txt文件对应一张图像，格式为：
```
class_id x_center y_center width height
```

**注意**: 坐标为相对坐标 (0-1之间的浮点数)

## 使用方法

### 1. 命令行使用
```bash
# 基础评价
python evaluate.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truths \
    --conf_threshold 0.25 \
    --iou_threshold 0.5 \
    --output results.txt

# 生成可视化图表
python evaluate.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truths \
    --visualize \
    --output_dir visualization_results

# 生成中文HTML报告（推荐）
python evaluate.py \
    --pred_dir /path/to/predictions \
    --gt_dir /path/to/ground_truths \
    --html_report \
    --class_names class_names.txt \
    --output_dir evaluation_report
```

### 2. Python代码使用
```python
from evaluate import DetectionEvaluator

# 创建评价器
evaluator = DetectionEvaluator(
    pred_dir="/path/to/predictions",
    gt_dir="/path/to/ground_truths", 
    conf_threshold=0.25,
    iou_threshold=0.5
)

# 运行评价
metrics = evaluator.run_evaluation()

# 保存结果
evaluator.save_results("evaluation_results.txt")

# 生成可视化图表（支持中文）
class_names = {0: "人员", 1: "车辆", 2: "动物"}
evaluator.save_visualization_charts("charts", class_names)

# 生成HTML报告
evaluator.generate_detailed_report("report", class_names)
```

### 3. 类别名称文件格式
创建 `class_names.txt` 文件，每行一个类别名称：
```
人员
车辆
动物
建筑物
交通标志
```

## 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| pred_dir | 预测结果目录路径 | 必填 |
| gt_dir | 真实标注目录路径 | 必填 |
| conf_threshold | 置信度阈值 | 0.25 |
| iou_threshold | IoU阈值 | 0.5 |
| output | 结果保存路径 | evaluation_results.txt |
| visualize | 生成可视化图表 | False |
| output_dir | 输出目录（包含图表） | evaluation_output |
| html_report | 生成HTML格式报告 | False |
| class_names | 类别名称文件路径 | None |

## 输出结果

### 控制台输出
```
================================================================================
YOLO目标检测性能评价结果
================================================================================
IoU阈值: 0.5
置信度阈值: 0.25
总类别数: 3
mAP@0.5: 0.7543

各类别详细指标:
--------------------------------------------------------------------------------
类别ID   Precision    Recall       F1-Score     AP          
--------------------------------------------------------------------------------
0        0.8234       0.7891       0.8059       0.7823      
1        0.7453       0.6982       0.7211       0.6834      
2        0.8901       0.8234       0.8553       0.7972      
--------------------------------------------------------------------------------
平均     0.8196       0.7702       0.7941       0.7543      
================================================================================

各类别AP排名:
----------------------------------------
 1. 类别2: AP=0.7972
 2. 类别0: AP=0.7823
 3. 类别1: AP=0.6834
----------------------------------------
```

### 可视化图表
使用 `--visualize` 参数可生成以下图表：
- **AP柱状图**: 各类别AP对比
- **PR关系图**: 精确率-召回率关系
- **雷达图**: 整体性能指标
- **类别对比图**: 各类别多指标对比

### HTML报告
使用 `--html_report` 参数可生成包含以下内容的HTML报告：
- 📊 交互式数据表格
- 📈 高质量可视化图表
- 📋 详细的指标说明
- 🎨 美观的中文界面

## 注意事项

1. **文件对应**: 预测结果文件名必须与真实标注文件名一致
2. **坐标格式**: 默认假设输入为相对坐标，内部会转换为绝对坐标
3. **图像尺寸**: 当前默认使用640x640，实际使用时可能需要根据具体情况调整
4. **置信度过滤**: 只有置信度大于等于阈值的预测框才会参与评价

## 扩展功能

### 批量评价不同IoU阈值
```python
iou_thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for iou_thresh in iou_thresholds:
    evaluator = DetectionEvaluator(pred_dir, gt_dir, iou_threshold=iou_thresh)
    metrics = evaluator.run_evaluation()
    print(f"mAP@{iou_thresh}: {metrics['mAP']:.4f}")
```

### 自定义可视化
```python
# 使用自定义类别名称
class_names = {
    0: "人员",
    1: "车辆", 
    2: "动物",
    3: "建筑物"
}

# 生成单个图表
evaluator.plot_class_ap_chart("ap_chart.png", class_names)
evaluator.plot_precision_recall_chart("pr_chart.png", class_names)
evaluator.plot_metrics_radar_chart("radar_chart.png")

# 生成完整报告
evaluator.generate_detailed_report("my_report", class_names)
```

### 快速演示
```bash
# 运行中文可视化演示
python demo_chinese.py
```

## 常见问题

1. **Q**: 如何处理不同图像尺寸？
   **A**: 需要修改代码中的图像尺寸设置，或者传入图像尺寸信息

2. **Q**: 支持COCO格式的标注吗？
   **A**: 当前只支持YOLO格式，如需支持其他格式需要添加转换函数

3. **Q**: 如何添加自定义类别名称？
   **A**: 创建类别名称文件或在代码中使用class_names字典

4. **Q**: 中文字符无法显示怎么办？
   **A**: 确保系统安装了中文字体，工具会自动检测并使用可用的中文字体

5. **Q**: 如何安装依赖？
   **A**: 运行 `pip install numpy matplotlib pillow` 安装必要依赖

6. **Q**: HTML报告无法打开怎么办？
   **A**: 使用现代浏览器（Chrome、Firefox、Edge）打开HTML文件

## 依赖要求
```bash
pip install numpy matplotlib pillow
```

## 字体支持
工具支持以下中文字体（自动检测）：
- SimHei (黑体)
- Microsoft YaHei (微软雅黑)
- PingFang SC (苹方)
- WenQuanYi Micro Hei (文泉驿微米黑)
- Noto Sans CJK SC (思源黑体)
