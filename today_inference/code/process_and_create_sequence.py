"""
处理股票数据并生成今天的序列
使用与训练时相同的预处理方法
"""
import pandas as pd
import numpy as np


def process_stock_data(df):
    """
    处理股票数据，计算百分比变化并归一化
    使用与训练时相同的预处理方法
    
    Args:
        df: pandas.DataFrame，包含日期、收盘、成交量等列
        
    Returns:
        pandas.DataFrame: 处理后的数据，包含日期、收盘价归一化、成交量归一化、涨跌
    """
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
    
    # 最后一行没有后一天，所以用NaN填充
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
    
    # 计算后一天相对于当前行的涨跌情况（涨为1，跌为0）
    up_down = []
    for i in range(len(close_prices) - 1):
        if close_prices[i+1] > close_prices[i]:
            up_down.append(1)
        else:
            up_down.append(0)
    
    # 最后一行没有后一天，用NaN填充
    up_down.append(np.nan)
    
    # 创建新的DataFrame
    result_df = pd.DataFrame({
        '日期': dates,
        '收盘价归一化': price_normalized,
        '成交量归一化': volume_normalized,
        '涨跌': up_down
    })
    
    return result_df


def create_today_sequence(processed_df, window_size=60):
    """
    创建到今天为止的最后一个window_size天序列
    用于预测今天（最后一行数据的日期）的涨跌
    
    Args:
        processed_df: pandas.DataFrame，处理后的数据（包含日期、收盘价归一化、成交量归一化、涨跌）
        window_size: int，序列窗口大小，默认60天
        
    Returns:
        tuple: (close_seq, volume_seq, last_date)
            - close_seq: numpy.array，收盘价归一化序列（window_size个值）
            - volume_seq: numpy.array，成交量归一化序列（window_size个值）
            - last_date: str，序列对应的日期（预测目标的日期，即"今天"）
    """
    # 提取所有数据（包括最后一行NaN）
    dates_all = processed_df['日期'].values
    close_prices_all = processed_df['收盘价归一化'].values
    volumes_all = processed_df['成交量归一化'].values
    
    # 过滤掉涨跌为NaN的行，用于确定有效数据的数量
    df_valid = processed_df.dropna(subset=['涨跌']).copy()
    num_valid = len(df_valid)
    
    if num_valid < window_size:
        raise ValueError(f"有效数据量不足，需要至少 {window_size} 条数据，当前只有 {num_valid} 条")
    
    # 提取最后window_size个有效数据点作为序列（用于预测今天）
    # 注意：实测发现能爬到今天的数据，所以序列要减去1，取昨天之前的数据预测今天
    # 序列包含从索引 num_valid - window_size - 1 到 num_valid - 1 的数据
    close_seq = close_prices_all[num_valid - window_size - 1 : num_valid - 1]
    volume_seq = volumes_all[num_valid - window_size - 1 : num_valid - 1]
    
    # 预测目标的日期：processed_df的最后一行日期（即"今天"）
    last_date = dates_all[-1]
    
    # 转换为字符串格式（如果是datetime对象）
    if hasattr(last_date, 'strftime'):
        last_date = last_date.strftime('%Y-%m-%d')
    else:
        last_date = str(last_date)
    
    print(f"成功创建序列，序列长度: {len(close_seq)}")
    if num_valid >= window_size + 1:
        start_idx = num_valid - window_size - 1
        end_idx = num_valid - 1
        if hasattr(dates_all[start_idx], 'strftime'):
            start_date_str = dates_all[start_idx].strftime('%Y-%m-%d')
            end_date_str = dates_all[end_idx].strftime('%Y-%m-%d')
        else:
            start_date_str = str(dates_all[start_idx])
            end_date_str = str(dates_all[end_idx])
        print(f"序列日期范围: {start_date_str} 到 {end_date_str}")
    print(f"预测目标日期: {last_date}（今天）")
    
    return close_seq, volume_seq, last_date


def process_and_get_today_sequence(df, sequence_length=60):
    """
    完整流程：处理数据并获取今天的序列
    
    Args:
        df: pandas.DataFrame，原始股票数据
        sequence_length: int，序列长度，默认60
        
    Returns:
        tuple: (close_seq, volume_seq, last_date)
            - close_seq: numpy.array，收盘价归一化序列
            - volume_seq: numpy.array，成交量归一化序列
            - last_date: str，序列对应的日期
    """
    # 步骤1：处理数据
    print("正在处理数据...")
    processed_df = process_stock_data(df)
    
    # 步骤2：创建今天的序列
    print("正在创建今天的序列...")
    close_seq, volume_seq, last_date = create_today_sequence(processed_df, window_size=sequence_length)
    
    return close_seq, volume_seq, last_date


if __name__ == "__main__":
    # 测试函数
    from get_stock_data import get_stock_data
    
    df = get_stock_data("518880")
    close_seq, volume_seq, last_date = process_and_get_today_sequence(df)
    
    print(f"\n收盘价序列前5个值: {close_seq[:5]}")
    print(f"成交量序列前5个值: {volume_seq[:5]}")
    print(f"预测日期: {last_date}")

