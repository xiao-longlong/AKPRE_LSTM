"""
LSTM模型推理模块
模拟 inference.py 实现推理脚本
"""
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import os
import json
from datetime import datetime
import logging
import pickle


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class LSTMModel(nn.Module):
    """LSTM模型定义（与训练脚本一致）"""
    def __init__(self, input_size=2, hidden_size=128, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        batch_size, seq_len, hidden = lstm_out1.shape
        lstm_out1_reshaped = lstm_out1.reshape(-1, hidden)
        lstm_out1_bn = self.bn1(lstm_out1_reshaped)
        lstm_out1 = lstm_out1_bn.reshape(batch_size, seq_len, hidden)
        
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        lstm_out3, _ = self.lstm3(lstm_out2)
        out = lstm_out3[:, -1, :]
        out = self.dropout3(out)
        
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout4(out)
        
        out = self.fc2(out)
        
        return out


class GoldPriceDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, sequences):
        self.sequences = torch.FloatTensor(sequences)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx]


def load_scaler_from_checkpoint(checkpoint):
    """从checkpoint加载scaler"""
    try:
        price_scaler = checkpoint['price_scaler']
        volume_scaler = checkpoint['volume_scaler']
        # 如果是bytes，使用pickle加载
        if isinstance(price_scaler, bytes):
            price_scaler = pickle.loads(price_scaler)
            volume_scaler = pickle.loads(volume_scaler)
        return price_scaler, volume_scaler
    except Exception as e:
        raise ValueError(f"无法加载scaler: {e}")


def inference_model(model_path, data_path, stock_code, end_date, config, log_dir=None):
    """
    模型推理
    
    Args:
        model_path: 模型文件路径
        data_path: 数据文件路径（CSV格式）
        stock_code: 股票代码
        end_date: 截止日期
        config: 配置字典
        log_dir: 日志目录
    
    Returns:
        dict: 推理结果信息
    """
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    print(f'Using device: {device}')
    
    # 创建日志目录
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"inference_{stock_code}_{end_date}"
    log_folder = os.path.join(log_dir, f"{timestamp}_{task_name}")
    os.makedirs(log_folder, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(log_folder, 'inference.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始推理: {stock_code}, 模型: {model_path}")
    logger.info(f"数据文件: {data_path}")
    
    # 检查文件是否存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f"数据文件不存在: {data_path}")
    
    # 加载模型
    logger.info("加载模型...")
    checkpoint = torch.load(model_path, map_location=device)
    model_config = checkpoint['model_config']
    
    # 创建模型
    model = LSTMModel(
        input_size=model_config['input_size'],
        hidden_size=model_config['hidden_size'],
        num_classes=model_config['num_classes']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    logger.info(f"模型架构: {model_config.get('architecture', 'Unknown')}")
    
    # 加载scaler
    logger.info("加载数据标准化器...")
    price_scaler, volume_scaler = load_scaler_from_checkpoint(checkpoint)
    
    # 加载数据
    logger.info("加载数据...")
    df = pd.read_csv(data_path, encoding='utf-8-sig')
    
    # 解析序列数据
    logger.info("解析序列数据...")
    price_sequences = []
    volume_sequences = []
    dates = []
    
    for idx, row in df.iterrows():
        dates.append(row['起始日期'])
        
        price_str = str(row['收盘价序列']).strip('"')
        price_seq = np.array([float(x) for x in price_str.split(',')])
        
        volume_str = str(row['成交量序列']).strip('"')
        volume_seq = np.array([float(x) for x in volume_str.split(',')])
        
        price_sequences.append(price_seq)
        volume_sequences.append(volume_seq)
    
    price_sequences = np.array(price_sequences)
    volume_sequences = np.array(volume_sequences)
    
    logger.info(f"总样本数: {len(dates)}")
    logger.info(f"价格序列形状: {price_sequences.shape}")
    logger.info(f"成交量序列形状: {volume_sequences.shape}")
    
    # 提取标签（如果存在）
    has_labels = '标签' in df.columns
    if has_labels:
        true_labels = df['标签'].values
        logger.info(f"标签分布: {np.bincount(true_labels)}")
    else:
        true_labels = None
        logger.info("数据中没有标签，仅进行预测")
    
    # 数据预处理
    logger.info("数据预处理...")
    price_scaled = price_scaler.transform(price_sequences.reshape(-1, 1)).reshape(price_sequences.shape)
    volume_scaled = volume_scaler.transform(volume_sequences.reshape(-1, 1)).reshape(volume_sequences.shape)
    
    # 组合特征
    X = np.stack([price_scaled, volume_scaled], axis=2)
    logger.info(f"最终数据形状: {X.shape}")
    
    # 创建数据加载器
    batch_size = config.get('batch_size', 64)
    dataset = GoldPriceDataset(X)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    
    # 推理
    logger.info("运行推理...")
    all_predictions = []
    all_probabilities = []
    
    with torch.no_grad():
        for sequences in dataloader:
            sequences = sequences.to(device)
            outputs = model(sequences)
            
            _, predicted = torch.max(outputs.data, 1)
            all_predictions.extend(predicted.cpu().numpy())
            
            probabilities = torch.softmax(outputs, dim=1)
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    logger.info(f"预测完成，预测分布: {np.bincount(all_predictions)}")
    
    # 保存结果
    logger.info("保存结果...")
    results_data = {
        '日期': dates,
        '预测结果': all_predictions,
        '预测概率_跌': all_probabilities[:, 0],
        '预测概率_涨': all_probabilities[:, 1]
    }
    
    if has_labels:
        results_data['真值标签'] = true_labels
    
    results_df = pd.DataFrame(results_data)
    results_path = os.path.join(log_folder, 'inference_results.csv')
    results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
    logger.info(f"推理结果已保存到: {results_path}")
    
    # 计算统计信息（如果有标签）
    if has_labels:
        accuracy = accuracy_score(true_labels, all_predictions)
        cm = confusion_matrix(true_labels, all_predictions)
        
        logger.info(f"\n{'='*60}")
        logger.info("推理统计信息")
        logger.info(f"{'='*60}")
        logger.info(f"总体准确率: {accuracy*100:.2f}%")
        logger.info(f"\n混淆矩阵:\n{cm}")
        logger.info(f"\n分类报告:\n{classification_report(true_labels, all_predictions, target_names=['Down (0)', 'Up (1)'])}")
        
        # 可视化
        logger.info("生成可视化图表...")
        plt.figure(figsize=(15, 10))
        
        # 1. 混淆矩阵
        plt.subplot(2, 3, 1)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=['Down (0)', 'Up (1)'],
                    yticklabels=['Down (0)', 'Up (1)'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        # 2. 滚动准确率
        plt.subplot(2, 3, 2)
        window_size = 30
        rolling_accuracy = []
        for i in range(len(true_labels)):
            start = max(0, i - window_size + 1)
            end = i + 1
            acc = accuracy_score(true_labels[start:end], all_predictions[start:end])
            rolling_accuracy.append(acc)
        
        plt.plot(rolling_accuracy, alpha=0.7)
        plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall: {accuracy:.3f}')
        plt.title(f'Rolling Accuracy (window={window_size})')
        plt.xlabel('Sample Index')
        plt.ylabel('Accuracy')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 3. 预测分布
        plt.subplot(2, 3, 3)
        pred_counts = np.bincount(all_predictions)
        true_counts = np.bincount(true_labels)
        x = np.arange(2)
        width = 0.35
        plt.bar(x - width/2, true_counts, width, label='True', alpha=0.8)
        plt.bar(x + width/2, pred_counts, width, label='Predicted', alpha=0.8)
        plt.xlabel('Class')
        plt.ylabel('Count')
        plt.title('Label Distribution')
        plt.xticks(x, ['Down (0)', 'Up (1)'])
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 4. 预测概率分布
        plt.subplot(2, 3, 4)
        plt.hist(all_probabilities[:, 0], bins=50, alpha=0.5, label='P(Down)', color='red')
        plt.hist(all_probabilities[:, 1], bins=50, alpha=0.5, label='P(Up)', color='green')
        plt.xlabel('Probability')
        plt.ylabel('Frequency')
        plt.title('Prediction Probability Distribution')
        plt.legend()
        plt.grid(True, alpha=0.3, axis='y')
        
        # 5. 正确/错误预测
        plt.subplot(2, 3, 5)
        correct = (true_labels == all_predictions).astype(int)
        rolling_correct = []
        for i in range(len(correct)):
            start = max(0, i - window_size + 1)
            end = i + 1
            rolling_correct.append(np.mean(correct[start:end]))
        
        plt.plot(rolling_correct, alpha=0.7, color='green', label='Rolling Correct Rate')
        plt.axhline(y=accuracy, color='r', linestyle='--', label=f'Overall: {accuracy:.3f}')
        plt.title(f'Rolling Correct Rate (window={window_size})')
        plt.xlabel('Sample Index')
        plt.ylabel('Correct Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        # 6. 按类别准确率
        plt.subplot(2, 3, 6)
        class_accuracy = []
        for label in [0, 1]:
            mask = true_labels == label
            if np.sum(mask) > 0:
                acc = accuracy_score(true_labels[mask], all_predictions[mask])
                class_accuracy.append(acc)
            else:
                class_accuracy.append(0)
        
        plt.bar(['Down (0)', 'Up (1)'], class_accuracy, alpha=0.8, color=['red', 'green'])
        plt.ylabel('Accuracy')
        plt.title('Accuracy by Class')
        plt.ylim([0, 1])
        plt.grid(True, alpha=0.3, axis='y')
        for i, v in enumerate(class_accuracy):
            plt.text(i, v + 0.02, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        viz_path = os.path.join(log_folder, 'inference_analysis.png')
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"可视化图表已保存到: {viz_path}")
        
        # 保存统计信息
        stats = {
            'stock_code': stock_code,
            'end_date': end_date,
            'overall_accuracy': float(accuracy),
            'class_accuracy': [float(ca) for ca in class_accuracy],
            'confusion_matrix': cm.tolist(),
            'total_samples': len(true_labels),
            'predictions_distribution': np.bincount(all_predictions).tolist(),
            'true_distribution': np.bincount(true_labels).tolist()
        }
        
        stats_path = os.path.join(log_folder, 'inference_statistics.json')
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        
        logger.info(f"统计信息已保存到: {stats_path}")
    else:
        logger.info("无标签数据，跳过统计信息计算")
    
    # 保存推理信息
    inference_info = {
        'stock_code': stock_code,
        'end_date': end_date,
        'model_path': model_path,
        'data_path': data_path,
        'has_labels': has_labels,
        'total_samples': len(dates),
        'results_path': results_path,
        'log_folder': log_folder
    }
    
    if has_labels:
        inference_info['accuracy'] = float(accuracy)
    
    info_path = os.path.join(log_folder, 'inference_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(inference_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"推理信息已保存到: {info_path}")
    logger.info("推理完成!")
    
    return {
        'log_folder': log_folder,
        'results_path': results_path,
        'accuracy': accuracy if has_labels else None
    }


if __name__ == "__main__":
    # 测试
    config = {
        'batch_size': 64,
        'use_gpu': True
    }
    
    model_path = "../checkpoint/best_model.pth"
    data_path = "../data/processed/518880_20251201/val.csv"
    
    if os.path.exists(model_path) and os.path.exists(data_path):
        result = inference_model(model_path, data_path, "518880", "20251201", config)
        print(f"推理完成，结果保存在: {result['log_folder']}")

