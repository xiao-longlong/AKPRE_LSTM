"""
股票数据预处理和序列化模块
合并预处理和序列化步骤，添加train/val数据划分功能
"""
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split


def process_stock_data(raw_data_path, stock_code, end_date, sequence_length=60, 
                      train_ratio=0.8, val_ratio=0.2, output_dir=None):
    """
    处理股票数据：预处理、序列化、划分train/val
    
    Args:
        raw_data_path: 原始数据文件路径
        stock_code: 股票代码
        end_date: 截止日期
        sequence_length: 序列长度，默认60
        train_ratio: 训练集比例，默认0.8
        val_ratio: 验证集比例，默认0.2 (train_ratio + val_ratio 应该 <= 1.0)
        output_dir: 输出目录，如果为None则使用默认路径
    
    Returns:
        dict: 包含处理后的数据信息
    """
    print(f"开始处理股票数据: {stock_code}")
    
    # 读取原始数据
    df = pd.read_csv(raw_data_path, encoding='utf-8-sig')
    print(f"读取数据: {len(df)} 条记录")
    
    # 提取日期、收盘价和成交量
    # 兼容不同的列名
    date_col = None
    close_col = None
    volume_col = None
    
    for col in df.columns:
        if '日期' in col or 'date' in col.lower():
            date_col = col
        if '收盘' in col or 'close' in col.lower():
            close_col = col
        if '成交量' in col or 'volume' in col.lower():
            volume_col = col
    
    if date_col is None or close_col is None or volume_col is None:
        raise ValueError(f"数据文件缺少必要列。可用列: {df.columns.tolist()}")
    
    # 滤除成交量小于平均值30%的天数
    volumes_raw = df[volume_col].astype(float)
    volume_mean = volumes_raw.mean()
    volume_threshold = volume_mean * 0.1
    df = df[volumes_raw >= volume_threshold].copy()
    print(f"过滤成交量小于平均值30%的数据后: {len(df)} 条记录 (平均值: {volume_mean:.2f}, 阈值: {volume_threshold:.2f})")
    
    dates = df[date_col].values
    close_prices = df[close_col].values.astype(float)
    volumes = df[volume_col].values.astype(float)
    
    # 计算后一天相对前一天的百分比变化
    price_pct_change = []
    volume_pct_change = []
    
    for i in range(len(close_prices) - 1):
        price_pct = (close_prices[i+1] - close_prices[i]) / close_prices[i] if close_prices[i] != 0 else 0
        volume_pct = (volumes[i+1] - volumes[i]) / volumes[i] if volumes[i] != 0 else 0
        price_pct_change.append(price_pct)
        volume_pct_change.append(volume_pct)
    
    # 最后一行没有后一天，用NaN填充
    price_pct_change.append(np.nan)
    volume_pct_change.append(np.nan)
    
    # 转换为numpy数组
    price_pct_change = np.array(price_pct_change)
    volume_pct_change = np.array(volume_pct_change)
    
    # 列归一化（使用min-max归一化）
    price_valid = price_pct_change[~np.isnan(price_pct_change)]
    volume_valid = volume_pct_change[~np.isnan(volume_pct_change)]
    
    if len(price_valid) > 0:
        price_min, price_max = price_valid.min(), price_valid.max()
        if price_max != price_min:
            price_normalized = (price_pct_change - price_min) / (price_max - price_min)
        else:
            price_normalized = np.zeros_like(price_pct_change)
    else:
        price_normalized = price_pct_change
    
    if len(volume_valid) > 0:
        volume_min, volume_max = volume_valid.min(), volume_valid.max()
        if volume_max != volume_min:
            volume_normalized = (volume_pct_change - volume_min) / (volume_max - volume_min)
        else:
            volume_normalized = np.zeros_like(volume_pct_change)
    else:
        volume_normalized = volume_pct_change
    
    # 计算后一天相对于当前行的涨跌标签（基于涨跌幅的线性映射）
    # 规则：
    # - 涨跌幅 > 3%：标签 = 1
    # - 涨跌幅 < -3%：标签 = 0
    # - 涨跌幅在 -3% 到 3% 之间：线性映射到 0~1，不涨不跌(0%) = 0.5
    up_down = []
    threshold = 0.03  # 3%阈值
    
    for i in range(len(close_prices) - 1):
        pct_change = price_pct_change[i]  # 已经计算好的涨跌幅
        
        if pct_change > threshold:
            # 涨跌幅 > 3%，标签 = 1
            label = 1.0
        elif pct_change < -threshold:
            # 涨跌幅 < -3%，标签 = 0
            label = 0.0
        else:
            # 涨跌幅在 -3% 到 3% 之间，线性映射到 0~1
            # 公式：label = 0.5 + (pct_change / threshold) * 0.5
            # 当 pct_change = 0 时，label = 0.5
            # 当 pct_change = threshold 时，label = 1.0
            # 当 pct_change = -threshold 时，label = 0.0
            label = 0.5 + (pct_change / threshold) * 0.5
            # 确保在 [0, 1] 范围内
            label = max(0.0, min(1.0, label))
        
        up_down.append(label)
    
    # 最后一行没有后一天，用NaN填充
    up_down.append(np.nan)
    
    # 创建处理后的DataFrame
    processed_df = pd.DataFrame({
        '日期': dates,
        '收盘价归一化': price_normalized,
        '成交量归一化': volume_normalized,
        '涨跌': up_down
    })
    
    # 过滤掉涨跌为NaN的行（最后一行）
    processed_df = processed_df.dropna(subset=['涨跌'])
    
    print(f"预处理完成，有效数据: {len(processed_df)} 条")
    
    # 提取数据用于序列化
    dates_clean = processed_df['日期'].values
    close_prices_norm = processed_df['收盘价归一化'].values
    volumes_norm = processed_df['成交量归一化'].values
    up_down_clean = processed_df['涨跌'].values.astype(float)  # 改为float，因为标签现在是0~1之间的连续值
    
    # 创建序列数据
    sequences_close = []
    sequences_volume = []
    labels = []
    sequence_dates = []
    
    # 滑动窗口创建序列
    for i in range(len(processed_df) - sequence_length - 1):
        # 提取sequence_length天的收盘价序列
        close_seq = close_prices_norm[i:i+sequence_length]
        # 提取sequence_length天的成交量序列
        volume_seq = volumes_norm[i:i+sequence_length]
        # 第sequence_length天的涨跌（索引i+sequence_length-1的下一天）
        label = up_down_clean[i+sequence_length]
        # 起始日期（序列结束的日期，即预测目标日期）
        start_date = dates_clean[i+sequence_length]
        
        sequences_close.append(close_seq)
        sequences_volume.append(volume_seq)
        labels.append(label)
        sequence_dates.append(start_date)
    
    # 转换为numpy数组
    sequences_close = np.array(sequences_close)
    sequences_volume = np.array(sequences_volume)
    labels = np.array(labels)
    
    print(f"共创建 {len(sequences_close)} 个序列")
    print(f"每个序列长度: {sequence_length}")
    
    # 统计标签分布（基于阈值分类）
    labels_binary = (labels > 0.5).astype(int)  # 用于统计，>0.5视为涨
    up_count = np.sum(labels_binary == 1)
    down_count = np.sum(labels_binary == 0)
    print(f"标签分布统计: 涨(>0.5): {up_count} ({up_count/len(labels)*100:.2f}%), "
          f"跌(<=0.5): {down_count} ({down_count/len(labels)*100:.2f}%)")
    print(f"标签范围: [{labels.min():.4f}, {labels.max():.4f}], 平均值: {labels.mean():.4f}")
    
    # 数据平衡：将涨跌数据调整成各50%（基于>0.5和<=0.5的分类）
    print("\n进行数据平衡处理...")
    up_indices = np.where(labels_binary == 1)[0]
    down_indices = np.where(labels_binary == 0)[0]
    
    print(f"平衡前 - 涨(>0.5): {len(up_indices)}, 跌(<=0.5): {len(down_indices)}")
    
    # 确定目标数量（取两者中的较小值，使两类数量相等）
    target_count = min(len(up_indices), len(down_indices))
    
    if len(up_indices) > len(down_indices):
        # 涨的多，随机下采样涨的样本
        np.random.seed(42)
        selected_up_indices = np.random.choice(up_indices, size=target_count, replace=False)
        balanced_indices = np.concatenate([selected_up_indices, down_indices])
        print(f"下采样涨的样本: {len(up_indices)} -> {target_count}")
    elif len(down_indices) > len(up_indices):
        # 跌的多，随机下采样跌的样本
        np.random.seed(42)
        selected_down_indices = np.random.choice(down_indices, size=target_count, replace=False)
        balanced_indices = np.concatenate([up_indices, selected_down_indices])
        print(f"下采样跌的样本: {len(down_indices)} -> {target_count}")
    else:
        # 已经平衡
        balanced_indices = np.concatenate([up_indices, down_indices])
        print("数据已经平衡，无需调整")
    
    # 打乱顺序
    np.random.seed(42)
    np.random.shuffle(balanced_indices)
    
    # 使用平衡后的索引提取数据
    sequences_close = sequences_close[balanced_indices]
    sequences_volume = sequences_volume[balanced_indices]
    labels = labels[balanced_indices]
    sequence_dates = [sequence_dates[i] for i in balanced_indices]
    
    # 重新计算统计信息
    labels_binary_balanced = (labels > 0.5).astype(int)
    up_count_balanced = np.sum(labels_binary_balanced == 1)
    down_count_balanced = np.sum(labels_binary_balanced == 0)
    print(f"平衡后 - 涨(>0.5): {up_count_balanced} ({up_count_balanced/len(labels)*100:.2f}%), "
          f"跌(<=0.5): {down_count_balanced} ({down_count_balanced/len(labels)*100:.2f}%)")
    print(f"平衡后标签范围: [{labels.min():.4f}, {labels.max():.4f}], 平均值: {labels.mean():.4f}")
    print(f"平衡后总样本数: {len(sequences_close)}")
    
    # 划分训练集和验证集
    # 确保比例合理
    if train_ratio + val_ratio > 1.0:
        val_ratio = 1.0 - train_ratio
    
    test_ratio = 1.0 - train_ratio - val_ratio
    
    # 创建索引数组
    indices = np.arange(len(sequences_close))
    
    # 为了分层采样，基于标签的二分类（>0.5 vs <=0.5）
    labels_binary_stratify = (labels > 0.5).astype(int)
    
    if test_ratio > 0:
        # 先分出测试集
        indices_temp, indices_test = train_test_split(
            indices,
            test_size=test_ratio, random_state=42, shuffle=True, stratify=labels_binary_stratify
        )
        # 再从剩余数据中分出训练集和验证集
        val_size = val_ratio / (train_ratio + val_ratio)
        train_indices, val_indices = train_test_split(
            indices_temp,
            test_size=val_size, random_state=42, shuffle=True, stratify=labels_binary_stratify[indices_temp]
        )
    else:
        # 直接划分训练集和验证集
        train_indices, val_indices = train_test_split(
            indices,
            test_size=val_ratio, random_state=42, shuffle=True, stratify=labels_binary_stratify
        )
        indices_test = np.array([])
    
    # 根据索引提取数据
    train_close = sequences_close[train_indices]
    train_volume = sequences_volume[train_indices]
    train_labels = labels[train_indices]
    train_dates = [sequence_dates[i] for i in train_indices]
    
    val_close = sequences_close[val_indices]
    val_volume = sequences_volume[val_indices]
    val_labels = labels[val_indices]
    val_dates = [sequence_dates[i] for i in val_indices]
    
    print(f"训练集: {len(train_close)} 个序列")
    print(f"验证集: {len(val_close)} 个序列")
    if len(indices_test) > 0:
        test_close = sequences_close[indices_test]
        test_volume = sequences_volume[indices_test]
        test_labels = labels[indices_test]
        test_dates = [sequence_dates[i] for i in indices_test]
        print(f"测试集: {len(test_close)} 个序列")
    else:
        test_close = None
        test_volume = None
        test_labels = None
        test_dates = None
    
    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), 
                                 'data', 'processed')
    
    # 创建以<股票码_截止日期>命名的文件夹
    folder_name = f"{stock_code}_{end_date}"
    output_folder = os.path.join(output_dir, folder_name)
    os.makedirs(output_folder, exist_ok=True)
    
    # 保存训练集
    train_data = []
    for i in range(len(train_close)):
        close_str = ','.join([str(x) for x in train_close[i]])
        volume_str = ','.join([str(x) for x in train_volume[i]])
        train_data.append({
            '起始日期': train_dates[i],
            '收盘价序列': close_str,
            '成交量序列': volume_str,
            '标签': train_labels[i]
        })
    
    train_df = pd.DataFrame(train_data)
    train_path = os.path.join(output_folder, 'train.csv')
    train_df.to_csv(train_path, index=False, encoding='utf-8-sig')
    print(f"训练集已保存到: {train_path}")
    
    # 保存验证集
    val_data = []
    for i in range(len(val_close)):
        close_str = ','.join([str(x) for x in val_close[i]])
        volume_str = ','.join([str(x) for x in val_volume[i]])
        val_data.append({
            '起始日期': val_dates[i],
            '收盘价序列': close_str,
            '成交量序列': volume_str,
            '标签': val_labels[i]
        })
    
    val_df = pd.DataFrame(val_data)
    val_path = os.path.join(output_folder, 'val.csv')
    val_df.to_csv(val_path, index=False, encoding='utf-8-sig')
    print(f"验证集已保存到: {val_path}")
    
    # 同时保存为numpy格式（便于模型训练使用）
    np.savez(os.path.join(output_folder, 'train.npz'),
             close_sequences=train_close,
             volume_sequences=train_volume,
             labels=train_labels,
             start_dates=train_dates)
    
    np.savez(os.path.join(output_folder, 'val.npz'),
             close_sequences=val_close,
             volume_sequences=val_volume,
             labels=val_labels,
             start_dates=val_dates)
    
    if test_close is not None:
        test_data = []
        for i in range(len(test_close)):
            close_str = ','.join([str(x) for x in test_close[i]])
            volume_str = ','.join([str(x) for x in test_volume[i]])
            test_data.append({
                '起始日期': test_dates[i],
                '收盘价序列': close_str,
                '成交量序列': volume_str,
                '标签': test_labels[i]
            })
        
        test_df = pd.DataFrame(test_data)
        test_path = os.path.join(output_folder, 'test.csv')
        test_df.to_csv(test_path, index=False, encoding='utf-8-sig')
        print(f"测试集已保存到: {test_path}")
        
        np.savez(os.path.join(output_folder, 'test.npz'),
                 close_sequences=test_close,
                 volume_sequences=test_volume,
                 labels=test_labels,
                 start_dates=test_dates)
    
    return {
        'output_folder': output_folder,
        'train_path': train_path,
        'val_path': val_path,
        'train_size': len(train_close),
        'val_size': len(val_close),
        'sequence_length': sequence_length
    }


if __name__ == "__main__":
    # 测试
    raw_path = "../data/raw/518880_20251201.csv"
    if os.path.exists(raw_path):
        result = process_stock_data(raw_path, "518880", "20251201")
        print(f"处理完成，输出目录: {result['output_folder']}")

