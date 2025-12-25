"""
使用LSTM模型推理今天的股票涨跌结果
"""
import torch
import torch.nn as nn
import numpy as np
import os
import pickle


# LSTM模型定义（与训练脚本一致）
class LSTMModel(nn.Module):
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
        # Layer 1: LSTM(128, return_sequences=True)
        lstm_out1, _ = self.lstm1(x)
        lstm_out1 = self.dropout1(lstm_out1)
        # BatchNormalization
        batch_size, seq_len, hidden = lstm_out1.shape
        lstm_out1_reshaped = lstm_out1.reshape(-1, hidden)
        lstm_out1_bn = self.bn1(lstm_out1_reshaped)
        lstm_out1 = lstm_out1_bn.reshape(batch_size, seq_len, hidden)
        
        # Layer 2: LSTM(128, return_sequences=True)
        lstm_out2, _ = self.lstm2(lstm_out1)
        lstm_out2 = self.dropout2(lstm_out2)
        
        # Layer 3: LSTM(128, return_sequences=False)
        lstm_out3, _ = self.lstm3(lstm_out2)
        out = lstm_out3[:, -1, :]
        out = self.dropout3(out)
        
        # Layer 4: Dense(32, activation='relu')
        out = self.fc1(out)
        out = self.relu(out)
        out = self.dropout4(out)
        
        # Layer 5: Dense(2)
        out = self.fc2(out)
        
        return out


def load_model(model_path):
    """
    加载训练好的LSTM模型
    
    Args:
        model_path: str，模型文件路径
        
    Returns:
        tuple: (model, price_scaler, volume_scaler, device)
    """
    # 检查是否有GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file '{model_path}' not found. Please check the path.")
    
    print("Loading model...")
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
    model.eval()  # 设置为评估模式
    
    print(f"Model loaded: {model_config.get('architecture', 'LSTM')}")
    print(f"Input features: {model_config.get('features', ['price', 'volume'])}")
    
    # 加载scaler
    try:
        price_scaler = checkpoint['price_scaler']
        volume_scaler = checkpoint['volume_scaler']
        # 如果是bytes，使用pickle加载
        if isinstance(price_scaler, bytes):
            price_scaler = pickle.loads(price_scaler)
            volume_scaler = pickle.loads(volume_scaler)
        print("Scalers loaded from checkpoint")
    except (KeyError, TypeError) as e:
        raise ValueError(f"无法加载scaler: {e}")
    
    return model, price_scaler, volume_scaler, device


def predict_today(close_seq, volume_seq, model_path):
    """
    使用模型预测今天的股票涨跌
    
    Args:
        close_seq: numpy.array，收盘价归一化序列（60个值）
        volume_seq: numpy.array，成交量归一化序列（60个值）
        model_path: str，模型文件路径
        
    Returns:
        dict: 包含预测结果的字典
            - prediction: int，预测结果（0=跌，1=涨）
            - probability_up: float，预测为"涨"的概率
            - probability_down: float，预测为"跌"的概率
    """
    # 加载模型
    model, price_scaler, volume_scaler, device = load_model(model_path)
    
    # 数据预处理（使用保存的scaler）
    print("\nPreprocessing sequence data...")
    
    # 对序列进行scaler变换（需要reshape为(-1, 1)然后reshape回来）
    close_scaled = price_scaler.transform(close_seq.reshape(-1, 1)).reshape(close_seq.shape)
    volume_scaled = volume_scaler.transform(volume_seq.reshape(-1, 1)).reshape(volume_seq.shape)
    
    # 组合特征 (1, 60, 2)
    # close_scaled和volume_scaled都是(60,)的一维数组，需要组合成(60, 2)
    X = np.column_stack([close_scaled, volume_scaled])  # (60, 2)
    X = X.reshape(1, len(close_seq), 2)  # (1, 60, 2) - 添加batch维度
    print(f"Input sequence shape: {X.shape}")
    
    # 转换为tensor
    X_tensor = torch.FloatTensor(X).to(device)
    
    # 推理
    print("Running inference...")
    with torch.no_grad():
        outputs = model(X_tensor)
        
        # 获取预测类别
        _, predicted = torch.max(outputs.data, 1)
        prediction = predicted.cpu().numpy()[0]
        
        # 获取概率（使用softmax）
        probabilities = torch.softmax(outputs, dim=1)
        probs = probabilities.cpu().numpy()[0]
        
        probability_down = float(probs[0])  # 跌的概率
        probability_up = float(probs[1])    # 涨的概率
    
    result = {
        'prediction': int(prediction),
        'probability_up': probability_up,
        'probability_down': probability_down
    }
    
    print(f"预测结果: {'涨' if prediction == 1 else '跌'}")
    print(f"涨的概率: {probability_up:.4f} ({probability_up*100:.2f}%)")
    print(f"跌的概率: {probability_down:.4f} ({probability_down*100:.2f}%)")
    
    return result


if __name__ == "__main__":
    # 测试函数
    import sys
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from get_stock_data import get_stock_data
    from process_and_create_sequence import process_and_get_today_sequence
    
    # 获取数据
    df = get_stock_data("518880")
    
    # 处理并获取序列
    close_seq, volume_seq, last_date = process_and_get_today_sequence(df)
    
    # 模型路径（需要指定）
    model_path = "../../logs/20251225_175236_train_518880_20251225/best_model.pth"
    
    # 预测
    result = predict_today(close_seq, volume_seq, model_path)
    print(f"\n预测日期: {last_date}")
    print(f"预测结果: {result}")

