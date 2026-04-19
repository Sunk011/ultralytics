"""
训练结果CSV可视化工具

该模块提供了一个类，用于可视化CSV文件中的训练结果。
每个指标都单独绘制，横轴为epoch。
"""

import pandas as pd
import matplotlib.pyplot as plt
import os
from pathlib import Path


class ResultsVisualizer:
    """
    用于可视化CSV文件中训练结果的类。
    每一列（除了epoch）都单独可视化为一张图。
    """
    
    def __init__(self, csv_path, output_dir=None):
        """
        使用CSV文件路径初始化可视化器。
        
        参数:
            csv_path (str): results.csv文件的路径
            output_dir (str, optional): 保存图表的目录。如果为None，则保存在CSV文件旁边
        """
        self.csv_path = csv_path
        self.df = None
        self.epochs = None
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = Path(csv_path).parent / 'visualizations'
        else:
            self.output_dir = Path(output_dir)
        
        # 如果输出目录不存在，则创建
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # 加载数据
        self._load_data()
    
    def _load_data(self):
        """加载CSV数据并提取epoch列。"""
        try:
            self.df = pd.read_csv(self.csv_path)
            self.epochs = self.df['epoch'].values
            print(f"✓ 成功从以下路径加载数据: {self.csv_path}")
            print(f"✓ 发现 {len(self.df)} 个epochs和 {len(self.df.columns)-1} 个指标")
        except Exception as e:
            raise ValueError(f"加载CSV文件出错: {e}")
    
    def _save_plot(self, column_name, fig):
        """
        保存图表到文件。
        
        参数:
            column_name (str): 正在绘制的列名
            fig: Matplotlib图形对象
        """
        # 清理列名以用作文件名（将/替换为_）
        safe_name = column_name.replace('/', '_').replace('(', '').replace(')', '')
        output_path = self.output_dir / f"{safe_name}.png"
        fig.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"✓ 已保存: {output_path}")
    
    def plot_time(self):
        """绘制训练时间随epoch的变化。"""
        if 'time' not in self.df.columns:
            print("⚠ 未找到'time'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['time'], linewidth=2, color='#2E86AB')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('时间 (秒)', fontsize=12)
        ax.set_title('训练时间 vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('time', fig)
    
    def plot_train_box_loss(self):
        """绘制训练box损失随epoch的变化。"""
        if 'train/box_loss' not in self.df.columns:
            print("⚠ 未找到'train/box_loss'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['train/box_loss'], linewidth=2, color='#A23B72')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Box损失', fontsize=12)
        ax.set_title('训练Box损失 vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('train/box_loss', fig)
    
    def plot_train_cls_loss(self):
        """绘制训练分类损失随epoch的变化。"""
        if 'train/cls_loss' not in self.df.columns:
            print("⚠ 未找到'train/cls_loss'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['train/cls_loss'], linewidth=2, color='#F18F01')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('分类损失', fontsize=12)
        ax.set_title('训练分类损失 vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('train/cls_loss', fig)
    
    def plot_train_dfl_loss(self):
        """绘制训练DFL损失随epoch的变化。"""
        if 'train/dfl_loss' not in self.df.columns:
            print("⚠ 未找到'train/dfl_loss'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['train/dfl_loss'], linewidth=2, color='#C73E1D')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('DFL损失', fontsize=12)
        ax.set_title('训练DFL损失 vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('train/dfl_loss', fig)
    
    def plot_metrics_precision(self):
        """绘制精确率指标随epoch的变化。"""
        if 'metrics/precision(B)' not in self.df.columns:
            print("⚠ 未找到'metrics/precision(B)'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['metrics/precision(B)'], linewidth=2, color='#06A77D')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('精确率', fontsize=12)
        ax.set_title('精确率指标(B) vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('metrics/precision(B)', fig)
    
    def plot_metrics_recall(self):
        """绘制召回率指标随epoch的变化。"""
        if 'metrics/recall(B)' not in self.df.columns:
            print("⚠ 未找到'metrics/recall(B)'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['metrics/recall(B)'], linewidth=2, color='#005F73')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('召回率', fontsize=12)
        ax.set_title('召回率指标(B) vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('metrics/recall(B)', fig)
    
    def plot_metrics_map50(self):
        """绘制mAP50指标随epoch的变化。"""
        if 'metrics/mAP50(B)' not in self.df.columns:
            print("⚠ 未找到'metrics/mAP50(B)'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['metrics/mAP50(B)'], linewidth=2, color='#0A9396')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP50', fontsize=12)
        ax.set_title('mAP50指标(B) vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('metrics/mAP50(B)', fig)
    
    def plot_metrics_map50_95(self):
        """绘制mAP50-95指标随epoch的变化。"""
        if 'metrics/mAP50-95(B)' not in self.df.columns:
            print("⚠ 未找到'metrics/mAP50-95(B)'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['metrics/mAP50-95(B)'], linewidth=2, color='#94D2BD')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('mAP50-95', fontsize=12)
        ax.set_title('mAP50-95指标(B) vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('metrics/mAP50-95(B)', fig)
    
    def plot_val_box_loss(self):
        """绘制验证box损失随epoch的变化。"""
        if 'val/box_loss' not in self.df.columns:
            print("⚠ 未找到'val/box_loss'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['val/box_loss'], linewidth=2, color='#E9D8A6')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('Box损失', fontsize=12)
        ax.set_title('验证Box损失 vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('val/box_loss', fig)
    
    def plot_val_cls_loss(self):
        """绘制验证分类损失随epoch的变化。"""
        if 'val/cls_loss' not in self.df.columns:
            print("⚠ 未找到'val/cls_loss'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['val/cls_loss'], linewidth=2, color='#EE9B00')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('分类损失', fontsize=12)
        ax.set_title('验证分类损失 vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('val/cls_loss', fig)
    
    def plot_val_dfl_loss(self):
        """绘制验证DFL损失随epoch的变化。"""
        if 'val/dfl_loss' not in self.df.columns:
            print("⚠ 未找到'val/dfl_loss'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['val/dfl_loss'], linewidth=2, color='#CA6702')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('DFL损失', fontsize=12)
        ax.set_title('验证DFL损失 vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('val/dfl_loss', fig)
    
    def plot_lr_pg0(self):
        """绘制学习率pg0随epoch的变化。"""
        if 'lr/pg0' not in self.df.columns:
            print("⚠ 未找到'lr/pg0'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['lr/pg0'], linewidth=2, color='#BB3E03')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('学习率', fontsize=12)
        ax.set_title('学习率(pg0) vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('lr/pg0', fig)
    
    def plot_lr_pg1(self):
        """绘制学习率pg1随epoch的变化。"""
        if 'lr/pg1' not in self.df.columns:
            print("⚠ 未找到'lr/pg1'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['lr/pg1'], linewidth=2, color='#AE2012')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('学习率', fontsize=12)
        ax.set_title('学习率(pg1) vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('lr/pg1', fig)
    
    def plot_lr_pg2(self):
        """绘制学习率pg2随epoch的变化。"""
        if 'lr/pg2' not in self.df.columns:
            print("⚠ 未找到'lr/pg2'列，跳过...")
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(self.epochs, self.df['lr/pg2'], linewidth=2, color='#9B2226')
        ax.set_xlabel('Epoch', fontsize=12)
        ax.set_ylabel('学习率', fontsize=12)
        ax.set_title('学习率(pg2) vs Epoch', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        self._save_plot('lr/pg2', fig)
    
    def visualize_all(self):
        """生成所有可视化图表。"""
        print(f"\n{'='*60}")
        print(f"开始可视化处理...")
        print(f"输出目录: {self.output_dir}")
        print(f"{'='*60}\n")
        
        # 调用所有绘图方法
        self.plot_time()
        self.plot_train_box_loss()
        self.plot_train_cls_loss()
        self.plot_train_dfl_loss()
        self.plot_metrics_precision()
        self.plot_metrics_recall()
        self.plot_metrics_map50()
        self.plot_metrics_map50_95()
        self.plot_val_box_loss()
        self.plot_val_cls_loss()
        self.plot_val_dfl_loss()
        self.plot_lr_pg0()
        self.plot_lr_pg1()
        self.plot_lr_pg2()
        
        print(f"\n{'='*60}")
        print(f"✓ 所有可视化图表已成功生成！")
        print(f"✓ 图表保存至: {self.output_dir}")
        print(f"{'='*60}\n")


def main():
    """命令行使用的主函数。"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='从CSV文件可视化训练结果'
    )
    parser.add_argument(
        'csv_path',
        type=str,
        help='results.csv文件的路径'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='保存图表的目录（默认：在CSV文件旁创建"visualizations"文件夹）'
    )
    
    args = parser.parse_args()
    
    # 创建可视化器并生成所有图表
    visualizer = ResultsVisualizer(args.csv_path, args.output_dir)
    visualizer.visualize_all()


if __name__ == '__main__':
    main()
