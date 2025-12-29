#!/usr/bin/env python3
"""
统计所有训练模型的验证集准确率并绘制直方图
只统计最佳准确率在前500个epoch内出现的模型
"""
import os
import re
import json
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
from collections import defaultdict

# 使用英文字体，确保所有文字都能正常显示
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False


def extract_best_accuracy_from_log(log_file):
    """
    从training.log、train_info.json或best_model.pth中提取最佳验证准确率和出现的epoch
    
    Returns:
        (best_accuracy, best_epoch) 或 (None, None) 如果无法提取
    """
    log_path = Path(log_file)
    log_dir = log_path.parent
    
    best_accuracy = None
    best_epoch = None
    
    # 方法1: 从train_info.json读取（最可靠）
    json_file = log_dir / 'train_info.json'
    if json_file.exists():
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                info = json.load(f)
                best_accuracy = info.get('best_val_accuracy')
        except:
            pass
    
    # 方法2: 从best_model.pth读取epoch信息（最准确）
    model_file = log_dir / 'best_model.pth'
    if model_file.exists():
        try:
            import torch
            checkpoint = torch.load(str(model_file), map_location='cpu')
            # 优先从checkpoint读取epoch
            if 'epoch' in checkpoint:
                best_epoch = checkpoint.get('epoch')
            # 如果checkpoint中有准确率，也使用它（更准确）
            if 'val_accuracy' in checkpoint:
                checkpoint_acc = checkpoint.get('val_accuracy') * 100  # 转换为百分比
                if best_accuracy is None or abs(checkpoint_acc - best_accuracy) < 0.1:
                    best_accuracy = checkpoint_acc
        except Exception as e:
            # 如果加载失败，忽略错误继续
            pass
    
    # 方法3: 从training.log读取
    if best_accuracy is None and log_path.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
            # 查找"最佳验证准确率"行
            for line in reversed(lines):
                if '最佳验证准确率' in line:
                    # 提取准确率数值
                    match = re.search(r'(\d+\.?\d*)%', line)
                    if match:
                        best_accuracy = float(match.group(1))
                        break
    
    if best_accuracy is None:
        return None, None
    
    # 如果还没找到epoch，从日志中查找最佳准确率第一次出现的epoch
    if best_epoch is None and log_path.exists():
        with open(log_file, 'r', encoding='utf-8') as f:
            for line in f:
                # 匹配格式: Epoch [123/3000], ..., Val Acc: 54.42%
                match = re.search(r'Epoch\s+\[(\d+)/\d+\].*?Val Acc:\s*(\d+\.?\d*)%', line)
                if match:
                    epoch = int(match.group(1))
                    val_acc = float(match.group(2))
                    
                    # 检查是否达到最佳准确率（允许小的浮点误差）
                    if abs(val_acc - best_accuracy) < 0.1:
                        best_epoch = epoch
                        break
    
    return best_accuracy, best_epoch


def analyze_all_models(logs_dir):
    """分析所有训练模型"""
    logs_path = Path(logs_dir)
    if not logs_path.exists():
        print(f"错误: 日志目录不存在: {logs_dir}")
        return
    
    accuracies = []
    skipped_count = 0
    error_count = 0
    processed_count = 0
    
    # 遍历所有训练日志文件夹
    for log_folder in sorted(logs_path.iterdir()):
        if not log_folder.is_dir():
            continue
        
        training_log = log_folder / 'training.log'
        if not training_log.exists():
            continue
        
        processed_count += 1
        stock_code = log_folder.name.split('_')[-2] if '_' in log_folder.name else 'unknown'
        
        try:
            best_acc, best_epoch = extract_best_accuracy_from_log(str(training_log))
            
            if best_acc is None:
                error_count += 1
                print(f"⚠ 无法提取准确率: {log_folder.name}")
                continue
            
            if best_epoch is None:
                # 如果找不到epoch，尝试从train_info.json获取
                json_file = log_folder / 'train_info.json'
                if json_file.exists():
                    try:
                        with open(json_file, 'r', encoding='utf-8') as f:
                            info = json.load(f)
                            total_epochs = info.get('total_epochs', 3000)
                            # 如果总epochs <= 500，说明最佳准确率在前500内，跳过
                            if total_epochs <= 500:
                                skipped_count += 1
                                print(f"⚠ 跳过 {stock_code}: 总epochs={total_epochs} (前500个epoch内)")
                                continue
                            else:
                                # 无法确定具体epoch，但总epochs>500，假设最佳准确率在500之后
                                best_epoch = total_epochs
                    except:
                        skipped_count += 1
                        print(f"⚠ 跳过 {stock_code}: 无法读取train_info.json")
                        continue
                else:
                    skipped_count += 1
                    print(f"⚠ 跳过 {stock_code}: 无法确定最佳准确率出现的epoch")
                    continue
            
            # 检查是否超过500个epoch（只有超过500的才纳入统计）
            if best_epoch <= 500:
                skipped_count += 1
                print(f"⚠ 跳过 {stock_code}: 最佳准确率出现在epoch {best_epoch} (前500个epoch内)")
                continue
            
            accuracies.append(best_acc)
            print(f"✓ {stock_code}: 准确率={best_acc:.2f}%, epoch={best_epoch}")
            
        except Exception as e:
            error_count += 1
            print(f"✗ 处理失败 {log_folder.name}: {e}")
    
    print(f"\n统计完成:")
    print(f"  处理总数: {processed_count}")
    print(f"  纳入统计: {len(accuracies)} (最佳准确率出现在500个epoch之后)")
    print(f"  跳过(前500个epoch内): {skipped_count}")
    print(f"  错误/无法提取: {error_count}")
    
    return accuracies


def plot_histogram(accuracies, output_file='accuracy_histogram.png'):
    """绘制准确率直方图"""
    if not accuracies:
        print("错误: 没有可用的准确率数据")
        return
    
    accuracies = np.array(accuracies)
    
    # 创建图表
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 绘制直方图
    n, bins, patches = ax.hist(accuracies, bins=30, edgecolor='black', alpha=0.7)
    
    # 计算统计信息
    mean_acc = np.mean(accuracies)
    median_acc = np.median(accuracies)
    std_acc = np.std(accuracies)
    min_acc = np.min(accuracies)
    max_acc = np.max(accuracies)
    
    # 添加统计信息文本（英文）
    stats_text = f'Samples: {len(accuracies)}\n'
    stats_text += f'Mean: {mean_acc:.2f}%\n'
    stats_text += f'Median: {median_acc:.2f}%\n'
    stats_text += f'Std: {std_acc:.2f}%\n'
    stats_text += f'Min: {min_acc:.2f}%\n'
    stats_text += f'Max: {max_acc:.2f}%'
    
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
            fontsize=10)
    
    # 添加平均值和中位数线（英文）
    ax.axvline(mean_acc, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_acc:.2f}%')
    ax.axvline(median_acc, color='green', linestyle='--', linewidth=2, label=f'Median: {median_acc:.2f}%')
    
    # 设置标签和标题（英文）
    ax.set_xlabel('Validation Accuracy (%)', fontsize=12)
    ax.set_ylabel('Number of Models', fontsize=12)
    ax.set_title('Validation Accuracy Distribution of All Trained Models\n(Only models with best accuracy after 500 epochs)', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"\n直方图已保存到: {output_file}")
    
    # 保存统计数据到输出文件所在目录
    output_path = Path(output_file)
    output_dir = output_path.parent
    stats_filename = output_path.stem + '_stats.json'
    stats_file = output_dir / stats_filename
    
    stats = {
        'total_models': len(accuracies),
        'mean': float(mean_acc),
        'median': float(median_acc),
        'std': float(std_acc),
        'min': float(min_acc),
        'max': float(max_acc),
        'percentiles': {
            '25%': float(np.percentile(accuracies, 25)),
            '50%': float(np.percentile(accuracies, 50)),
            '75%': float(np.percentile(accuracies, 75)),
            '90%': float(np.percentile(accuracies, 90)),
            '95%': float(np.percentile(accuracies, 95)),
        }
    }
    
    # 确保目录存在
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(stats_file, 'w', encoding='utf-8') as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    
    print(f"统计数据已保存到: {stats_file}")
    
    plt.show()


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='统计所有训练模型的验证集准确率')
    parser.add_argument('--logs-dir', type=str, default='logs',
                       help='日志目录路径 (默认: logs)')
    parser.add_argument('--output', type=str, default='accuracy_histogram.png',
                       help='输出图片文件名或路径 (默认: accuracy_histogram.png)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='输出目录路径 (如果指定，所有结果将保存到此目录)')
    
    args = parser.parse_args()
    
    # 处理输出路径：确保所有结果保存到对应文件夹
    output_path = Path(args.output)
    script_dir = Path(__file__).parent
    
    if args.output_dir:
        # 如果指定了输出目录，所有文件保存到该目录
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # 如果输出路径只是文件名，使用指定的输出目录
        if output_path.parent == Path('.') or str(output_path.parent) == '.':
            output_path = output_dir / output_path.name
        else:
            # 如果输出路径包含目录，但指定了输出目录，使用输出目录
            output_path = output_dir / output_path.name
        final_output_dir = output_dir
    else:
        # 如果没有指定输出目录
        if output_path.parent == Path('.') or str(output_path.parent) == '.':
            # 如果只是文件名，保存到脚本所在目录下的 results 文件夹
            final_output_dir = script_dir / 'results'
            final_output_dir.mkdir(parents=True, exist_ok=True)
            output_path = final_output_dir / output_path.name
        else:
            # 如果输出路径包含目录，使用该目录
            if output_path.is_absolute():
                final_output_dir = output_path.parent
            else:
                # 相对路径，相对于脚本目录
                output_path = script_dir / output_path
                final_output_dir = output_path.parent
            final_output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("开始分析训练模型准确率")
    print("=" * 60)
    print(f"日志目录: {args.logs_dir}")
    print(f"输出目录: {final_output_dir}")
    print(f"输出文件: {output_path}")
    print("=" * 60)
    print()
    
    # 分析所有模型
    accuracies = analyze_all_models(args.logs_dir)
    
    if accuracies:
        # 绘制直方图（使用处理后的完整路径）
        plot_histogram(accuracies, str(output_path))
    else:
        print("错误: 没有找到可用的准确率数据")


if __name__ == '__main__':
    main()

