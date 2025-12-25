"""
LSTM模型训练模块
模拟 train_lstm.py 实现训练任务函数
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import os
import json
from datetime import datetime
import logging


# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class GoldPriceDataset(Dataset):
    """自定义数据集类"""
    def __init__(self, sequences, labels):
        self.sequences = torch.FloatTensor(sequences)
        self.labels = torch.LongTensor(labels)
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]


class LSTMModel(nn.Module):
    """LSTM模型定义"""
    def __init__(self, input_size=2, hidden_size=128, num_classes=2):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        
        # Layer 1: LSTM(128, return_sequences=True)
        self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
        self.dropout1 = nn.Dropout(0.2)
        self.bn1 = nn.BatchNorm1d(hidden_size)
        
        # Layer 2: LSTM(128, return_sequences=True)
        self.lstm2 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout2 = nn.Dropout(0.1)
        
        # Layer 3: LSTM(128, return_sequences=False)
        self.lstm3 = nn.LSTM(hidden_size, hidden_size, batch_first=True)
        self.dropout3 = nn.Dropout(0.2)
        
        # Layer 4: Dense(32, activation='relu')
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout4 = nn.Dropout(0.2)
        
        # Layer 5: Dense(2)
        self.fc2 = nn.Linear(32, num_classes)
        
    def forward(self, x):
        # x shape: (batch_size, seq_len, input_size)
        
        # Layer 1
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        # BatchNormalization
        batch_size, seq_len, hidden = lstm_out1.shape
        lstm_out1_reshaped = lstm_out1.reshape(-1, hidden)
        lstm_out1_bn = self.bn1(lstm_out1_reshaped)
        lstm_out1 = lstm_out1_bn.reshape(batch_size, seq_len, hidden)
        
        # Layer 2
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Layer 3
        lstm_out3, _ = self.lstm3(lstm_out2)
        out = lstm_out3[:, -1, :]
        out = self.dropout3(out)
        
        # Layer 4
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout4(out)
        
        # Layer 5
        out = self.fc2(out)
        
        return out


class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=30, monitor='val_accuracy', mode='max', verbose=1):
        self.patience = patience
        self.monitor = monitor
        self.mode = mode
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_epoch = 0
        
    def __call__(self, score, epoch):
        if self.best_score is None:
            self.best_score = score
            self.best_epoch = epoch
        elif (self.mode == 'max' and score < self.best_score) or \
             (self.mode == 'min' and score > self.best_score):
            self.counter += 1
            if self.verbose and self.counter >= self.patience:
                print(f'\n{"="*60}')
                print(f'Early Stopping Triggered!')
                print(f'Reason: {self.monitor} did not improve for {self.patience} consecutive epochs.')
                print(f'Best {self.monitor}: {self.best_score:.4f} at epoch {self.best_epoch+1}')
                print(f'Current {self.monitor}: {score:.4f} at epoch {epoch+1}')
                print(f'{"="*60}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            improvement = score - self.best_score if self.mode == 'max' else self.best_score - score
            if self.verbose and self.counter > 0:
                print(f'✓ {self.monitor} improved: {self.best_score:.4f} -> {score:.4f} '
                      f'(+{improvement:.4f}). Counter reset.')
            self.best_score = score
            self.best_epoch = epoch
            self.counter = 0


def train_lstm_model(data_folder, stock_code, end_date, config, log_dir=None):
    """
    训练LSTM模型
    
    Args:
        data_folder: 数据文件夹路径（包含train.csv和val.csv）
        stock_code: 股票代码
        end_date: 截止日期
        config: 配置字典
        log_dir: 日志目录
    
    Returns:
        str: 保存的模型路径
    """
    # 设置随机种子
    torch.manual_seed(config.get('random_seed', 42))
    np.random.seed(config.get('random_seed', 42))
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() and config.get('use_gpu', True) else 'cpu')
    print(f'Using device: {device}')
    
    # 创建日志目录
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'logs')
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    task_name = f"train_{stock_code}_{end_date}"
    log_folder = os.path.join(log_dir, f"{timestamp}_{task_name}")
    os.makedirs(log_folder, exist_ok=True)
    
    # 设置日志
    log_file = os.path.join(log_folder, 'training.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    logger = logging.getLogger(__name__)
    
    logger.info(f"开始训练: {stock_code}, 数据目录: {data_folder}")
    logger.info(f"配置: {json.dumps(config, indent=2, ensure_ascii=False)}")
    
    # 加载数据
    logger.info("加载数据...")
    train_df = pd.read_csv(os.path.join(data_folder, 'train.csv'), encoding='utf-8-sig')
    val_df = pd.read_csv(os.path.join(data_folder, 'val.csv'), encoding='utf-8-sig')
    
    # 解析序列数据
    logger.info("解析序列数据...")
    def parse_sequences(df):
        price_sequences = []
        volume_sequences = []
        labels = []
        
        for idx, row in df.iterrows():
            price_str = str(row['收盘价序列']).strip('"')
            price_seq = np.array([float(x) for x in price_str.split(',')])
            
            volume_str = str(row['成交量序列']).strip('"')
            volume_seq = np.array([float(x) for x in volume_str.split(',')])
            
            price_sequences.append(price_seq)
            volume_sequences.append(volume_seq)
            labels.append(int(row['标签']))
        
        return np.array(price_sequences), np.array(volume_sequences), np.array(labels)
    
    train_price, train_volume, train_labels = parse_sequences(train_df)
    val_price, val_volume, val_labels = parse_sequences(val_df)
    
    logger.info(f"训练集: {len(train_price)} 个样本")
    logger.info(f"验证集: {len(val_price)} 个样本")
    logger.info(f"标签分布 - 训练集: {np.bincount(train_labels)}")
    logger.info(f"标签分布 - 验证集: {np.bincount(val_labels)}")
    
    # 数据标准化
    logger.info("数据标准化...")
    price_scaler = StandardScaler()
    volume_scaler = StandardScaler()
    
    # 使用训练集拟合scaler
    price_scaler.fit(train_price.reshape(-1, 1))
    volume_scaler.fit(train_volume.reshape(-1, 1))
    
    # 标准化
    train_price_scaled = price_scaler.transform(train_price.reshape(-1, 1)).reshape(train_price.shape)
    train_volume_scaled = volume_scaler.transform(train_volume.reshape(-1, 1)).reshape(train_volume.shape)
    
    val_price_scaled = price_scaler.transform(val_price.reshape(-1, 1)).reshape(val_price.shape)
    val_volume_scaled = volume_scaler.transform(val_volume.reshape(-1, 1)).reshape(val_volume.shape)
    
    # 组合特征
    X_train = np.stack([train_price_scaled, train_volume_scaled], axis=2)
    X_val = np.stack([val_price_scaled, val_volume_scaled], axis=2)
    
    logger.info(f"训练数据形状: {X_train.shape}")
    logger.info(f"验证数据形状: {X_val.shape}")
    
    # 创建数据加载器
    batch_size = config.get('batch_size', 32)
    train_dataset = GoldPriceDataset(X_train, train_labels)
    val_dataset = GoldPriceDataset(X_val, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # 创建模型
    hidden_size = config.get('hidden_size', 128)
    model = LSTMModel(input_size=2, hidden_size=hidden_size, num_classes=2)
    model = model.to(device)
    
    logger.info(f"模型参数总数: {sum(p.numel() for p in model.parameters()):,}")
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    learning_rate = config.get('learning_rate', 0.001)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # 学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.0005, patience=3,
        min_lr=0.00001, verbose=True
    )
    
    # 早停机制
    early_stopping = EarlyStopping(
        patience=config.get('early_stopping_patience', 30),
        monitor='val_accuracy',
        mode='max',
        verbose=1
    )
    
    # 训练
    num_epochs = config.get('num_epochs', 100)
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []
    
    logger.info("开始训练...")
    best_val_acc = 0.0
    best_model_path = None
    
    for epoch in range(num_epochs):
        # 训练阶段
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0
        
        for sequences, labels in train_loader:
            sequences = sequences.to(device)
            labels = labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(sequences)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()
        
        train_accuracy = 100 * correct_train / total_train
        avg_train_loss = train_loss / len(train_loader)
        
        # 验证阶段
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0
        
        with torch.no_grad():
            for sequences, labels in val_loader:
                sequences = sequences.to(device)
                labels = labels.to(device)
                
                outputs = model(sequences)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs.data, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()
        
        val_accuracy = 100 * correct_val / total_val
        avg_val_loss = val_loss / len(val_loader)
        
        # 记录
        train_losses.append(avg_train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        # 学习率调度
        scheduler.step(avg_val_loss)
        
        # 早停检查
        early_stopping(val_accuracy, epoch)
        
        # 保存最佳模型
        if val_accuracy > best_val_acc:
            best_val_acc = val_accuracy
            best_model_path = os.path.join(log_folder, 'best_model.pth')
            torch.save({
                'model_state_dict': model.state_dict(),
                'price_scaler': price_scaler,
                'volume_scaler': volume_scaler,
                'model_config': {
                    'input_size': 2,
                    'hidden_size': hidden_size,
                    'num_classes': 2,
                    'architecture': '3xLSTM(128) + Dense(32) + Dense(2)',
                    'dropout_rates': [0.2, 0.1, 0.2, 0.2],
                    'features': ['price', 'volume']
                },
                'stock_code': stock_code,
                'end_date': end_date,
                'epoch': epoch + 1,
                'val_accuracy': val_accuracy
            }, best_model_path)
        
        # 打印信息
        if (epoch + 1) % 5 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            logger.info(f'Epoch [{epoch+1}/{num_epochs}], '
                       f'Train Loss: {avg_train_loss:.4f}, Train Acc: {train_accuracy:.2f}%, '
                       f'Val Loss: {avg_val_loss:.4f}, Val Acc: {val_accuracy:.2f}%, '
                       f'LR: {current_lr:.6f}')
        
        # 早停
        if early_stopping.early_stop:
            logger.info(f"早停触发于 epoch {epoch+1}")
            break
    
    logger.info("训练完成!")
    logger.info(f"最佳验证准确率: {best_val_acc:.2f}%")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Train Accuracy')
    plt.plot(val_accuracies, label='Val Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    
    plt.tight_layout()
    curve_path = os.path.join(log_folder, 'training_curves.png')
    plt.savefig(curve_path)
    logger.info(f"训练曲线已保存到: {curve_path}")
    
    # 保存训练信息
    train_info = {
        'stock_code': stock_code,
        'end_date': end_date,
        'config': config,
        'best_val_accuracy': float(best_val_acc),
        'total_epochs': epoch + 1,
        'train_size': len(train_price),
        'val_size': len(val_price),
        'model_path': best_model_path,
        'log_folder': log_folder
    }
    
    info_path = os.path.join(log_folder, 'train_info.json')
    with open(info_path, 'w', encoding='utf-8') as f:
        json.dump(train_info, f, indent=2, ensure_ascii=False)
    
    logger.info(f"训练信息已保存到: {info_path}")
    
    return best_model_path


if __name__ == "__main__":
    # 测试
    config = {
        'batch_size': 32,
        'num_epochs': 100,
        'learning_rate': 0.001,
        'hidden_size': 128,
        'early_stopping_patience': 30,
        'random_seed': 42
    }
    
    data_folder = "../data/processed/518880_20251201"
    if os.path.exists(data_folder):
        model_path = train_lstm_model(data_folder, "518880", "20251201", config)
        print(f"模型已保存到: {model_path}")

