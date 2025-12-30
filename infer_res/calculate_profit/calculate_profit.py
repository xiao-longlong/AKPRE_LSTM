"""
基于模型预测的股票交易利润计算脚本
模拟交易策略：根据模型预测结果进行买入/卖出操作
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path

# 添加项目路径
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(project_root, 'today_inference', 'code'))

from get_stock_data import get_stock_data
from process_and_create_sequence import process_stock_data
from predict_today import load_model


def load_train_info(log_folder_path):
    """
    从训练日志目录加载训练信息
    
    Args:
        log_folder_path: 训练日志目录路径
    
    Returns:
        dict: 训练信息，包含股票代码、模型路径等
    """
    train_info_path = os.path.join(log_folder_path, 'train_info.json')
    
    if not os.path.exists(train_info_path):
        raise FileNotFoundError(f"训练信息文件不存在: {train_info_path}")
    
    with open(train_info_path, 'r', encoding='utf-8') as f:
        train_info = json.load(f)
    
    # 获取模型路径
    model_path = train_info.get('model_path')
    if model_path is None:
        model_path = os.path.join(log_folder_path, 'best_model.pth')
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 获取股票代码
    stock_code = train_info.get('stock_code')
    if stock_code is None:
        raise ValueError("训练信息中缺少股票代码")
    
    # 获取序列长度
    sequence_length = 60  # 默认值
    
    return {
        'stock_code': stock_code,
        'model_path': model_path,
        'sequence_length': sequence_length,
        'train_info': train_info
    }


def get_trading_dates(df, start_date=None, num_trading_days=30):
    """
    获取交易日列表
    
    Args:
        df: 股票数据DataFrame
        start_date: 起始日期（字符串格式 YYYY-MM-DD），如果为None则从最后一天往前推
        num_trading_days: 交易日数量
    
    Returns:
        list: 交易日列表（DataFrame索引）
    """
    # 确保日期列存在
    date_col = None
    for col in df.columns:
        if '日期' in col or 'date' in col.lower():
            date_col = col
            break
    
    if date_col is None:
        raise ValueError("数据中找不到日期列")
    
    # 确保日期是datetime类型
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values(by=date_col).reset_index(drop=True)
    
    # 确定结束日期（最后一天）
    end_date = df[date_col].max()
    
    # 确定起始日期
    if start_date is None:
        # 从最后一天往前推num_trading_days个交易日
        # 简单实现：取最后num_trading_days条数据
        start_idx = max(0, len(df) - num_trading_days - 1)
        start_date = df.iloc[start_idx][date_col]
    else:
        start_date = pd.to_datetime(start_date)
    
    # 筛选日期范围
    mask = (df[date_col] >= start_date) & (df[date_col] <= end_date)
    trading_df = df[mask].copy()
    
    if len(trading_df) < num_trading_days:
        print(f"警告: 请求 {num_trading_days} 个交易日，但只有 {len(trading_df)} 个交易日可用")
    
    return trading_df, date_col


def create_sequence_for_date(processed_df, target_date_idx, window_size=60):
    """
    为指定日期创建预测序列
    
    Args:
        processed_df: 处理后的数据DataFrame
        target_date_idx: 目标日期在processed_df中的索引
        window_size: 序列窗口大小
    
    Returns:
        tuple: (close_seq, volume_seq) 或 None（如果数据不足）
    """
    # 需要target_date_idx之前至少有window_size个有效数据点
    if target_date_idx < window_size:
        return None
    
    # 提取序列（从target_date_idx - window_size 到 target_date_idx - 1）
    # 注意：这里使用的是归一化后的百分比变化数据
    close_seq = processed_df['收盘价归一化'].iloc[target_date_idx - window_size:target_date_idx].values
    volume_seq = processed_df['成交量归一化'].iloc[target_date_idx - window_size:target_date_idx].values
    
    # 检查是否有NaN值
    if np.isnan(close_seq).any() or np.isnan(volume_seq).any():
        return None
    
    return close_seq, volume_seq


def simulate_trading(trading_df, predictions, date_col, initial_capital=10000):
    """
    模拟交易过程
    
    Args:
        trading_df: 交易日数据DataFrame
        predictions: 预测结果列表，每个元素是 {'date': str, 'prediction': int, 'probability_up': float}
        date_col: 日期列名
        initial_capital: 初始资金
    
    Returns:
        dict: 交易结果信息
    """
    # 找到开盘价和收盘价列
    open_col = None
    close_col = None
    
    for col in trading_df.columns:
        if '开盘' in col or 'open' in col.lower():
            open_col = col
        if '收盘' in col or 'close' in col.lower():
            close_col = col
    
    if open_col is None or close_col is None:
        raise ValueError(f"找不到开盘价或收盘价列。可用列: {trading_df.columns.tolist()}")
    
    # 创建预测字典，方便查找
    pred_dict = {pred['date']: pred for pred in predictions}
    
    # 初始化状态
    cash = initial_capital  # 现金
    shares = 0  # 持有股数
    position = 0  # 持仓状态：0=空仓，1=持仓
    total_trades = 0  # 交易次数
    
    # 交易记录
    trade_records = []
    
    # 按日期遍历
    for idx, row in trading_df.iterrows():
        date = row[date_col]
        
        # 转换为字符串用于查找
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
        else:
            date_str = str(date)
        
        # 获取该日期的预测（预测的是第二天的涨跌）
        if date_str not in pred_dict:
            # 如果没有预测，说明该日期因为数据不足被跳过了
            # 这种情况下，保持当前状态（不进行任何操作）
            if position == 0:
                # 空仓，继续空仓（记录但不改变状态）
                trade_records.append({
                    'date': date_str,
                    'action': '无预测-空仓',
                    'price': close_price,
                    'shares': 0,
                    'cash': cash,
                    'position_value': 0,
                    'total_value': cash,
                    'prediction': '无预测',
                    'probability_up': None
                })
            else:
                # 持仓，继续持有（以收盘价计算持仓价值）
                trade_records.append({
                    'date': date_str,
                    'action': '无预测-持有',
                    'price': close_price,
                    'shares': shares,
                    'cash': cash,
                    'position_value': shares * close_price,
                    'total_value': cash + shares * close_price,
                    'prediction': '无预测',
                    'probability_up': None
                })
            continue
        
        prediction = pred_dict[date_str]
        pred_value = prediction['prediction']  # 0=跌，1=涨
        open_price = float(row[open_col])
        close_price = float(row[close_col])
        
        # 根据预测决定操作
        # 如果预测第二天涨（prediction==1），买入或持有
        # 如果预测第二天跌（prediction==0），卖出
        
        if pred_value == 1:  # 预测涨
            if position == 0:  # 空仓，买入
                # 全仓买入
                shares = cash / open_price
                cash = 0
                position = 1
                total_trades += 1
                trade_records.append({
                    'date': date_str,
                    'action': '买入',
                    'price': open_price,
                    'shares': shares,
                    'cash': cash,
                    'position_value': shares * close_price,
                    'total_value': cash + shares * close_price,
                    'prediction': '涨',
                    'probability_up': prediction['probability_up']
                })
            else:  # 持仓，继续持有
                # 更新持仓价值（以收盘价计算）
                trade_records.append({
                    'date': date_str,
                    'action': '持有',
                    'price': close_price,
                    'shares': shares,
                    'cash': cash,
                    'position_value': shares * close_price,
                    'total_value': cash + shares * close_price,
                    'prediction': '涨',
                    'probability_up': prediction['probability_up']
                })
        else:  # 预测跌
            if position == 1:  # 持仓，卖出
                # 全仓卖出
                cash = shares * open_price
                shares = 0
                position = 0
                total_trades += 1
                trade_records.append({
                    'date': date_str,
                    'action': '卖出',
                    'price': open_price,
                    'shares': 0,
                    'cash': cash,
                    'position_value': 0,
                    'total_value': cash,
                    'prediction': '跌',
                    'probability_up': prediction['probability_up']
                })
            else:  # 空仓，继续空仓
                trade_records.append({
                    'date': date_str,
                    'action': '空仓',
                    'price': close_price,
                    'shares': 0,
                    'cash': cash,
                    'position_value': 0,
                    'total_value': cash,
                    'prediction': '跌',
                    'probability_up': prediction['probability_up']
                })
    
    # 最后一天如果还有持仓，以收盘价卖出
    if position == 1:
        last_row = trading_df.iloc[-1]
        last_date = last_row[date_col]
        if hasattr(last_date, 'strftime'):
            last_date_str = last_date.strftime('%Y-%m-%d')
        else:
            last_date_str = str(last_date)
        
        last_close = float(last_row[close_col])
        cash = shares * last_close
        shares = 0
        total_trades += 1
        
        trade_records.append({
            'date': last_date_str,
            'action': '最后卖出',
            'price': last_close,
            'shares': 0,
            'cash': cash,
            'position_value': 0,
            'total_value': cash,
            'prediction': '结束',
            'probability_up': 0
        })
    
    # 计算最终结果
    final_value = cash + shares * float(trading_df.iloc[-1][close_col]) if position == 1 else cash
    profit = final_value - initial_capital
    profit_rate = (profit / initial_capital) * 100
    
    return {
        'initial_capital': initial_capital,
        'final_value': final_value,
        'profit': profit,
        'profit_rate': profit_rate,
        'total_trades': total_trades,
        'trade_records': trade_records
    }


def calculate_profit(log_folder_path, num_trading_days=30, start_date=None, initial_capital=10000):
    """
    计算利润的主函数
    
    Args:
        log_folder_path: 训练日志目录路径
        num_trading_days: 交易日数量
        start_date: 起始日期（字符串格式 YYYY-MM-DD），如果为None则从最后一天往前推
        initial_capital: 初始资金
    
    Returns:
        dict: 计算结果
    """
    print("="*60)
    print("股票交易利润计算系统")
    print("="*60)
    
    # 步骤1：加载训练信息
    print("\n[步骤1] 加载训练信息")
    try:
        train_info = load_train_info(log_folder_path)
        print(f"股票代码: {train_info['stock_code']}")
        print(f"模型路径: {train_info['model_path']}")
        print(f"序列长度: {train_info['sequence_length']}")
    except Exception as e:
        print(f"加载训练信息失败: {e}")
        return None
    
    # 步骤2：获取股票数据
    print(f"\n[步骤2] 获取股票 {train_info['stock_code']} 的历史数据")
    try:
        stock_df = get_stock_data(train_info['stock_code'])
        print(f"成功获取 {len(stock_df)} 条历史数据")
        
        # 滤除成交量小于平均值30%的天数
        # 找到成交量列
        volume_col = None
        for col in stock_df.columns:
            if '成交量' in col or 'volume' in col.lower():
                volume_col = col
                break
        
        if volume_col is not None:
            volumes_raw = stock_df[volume_col].astype(float)
            volume_mean = volumes_raw.mean()
            volume_threshold = volume_mean * 0.3
            stock_df = stock_df[volumes_raw >= volume_threshold].copy()
            print(f"过滤成交量小于平均值30%的数据后: {len(stock_df)} 条记录 (平均值: {volume_mean:.2f}, 阈值: {volume_threshold:.2f})")
        else:
            print("警告: 未找到成交量列，跳过成交量过滤")
    except Exception as e:
        print(f"获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤3：处理数据
    print(f"\n[步骤3] 处理股票数据")
    try:
        processed_df = process_stock_data(stock_df)
        print(f"数据处理完成，共 {len(processed_df)} 条记录")
    except Exception as e:
        print(f"处理数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤4：获取交易日范围
    print(f"\n[步骤4] 确定交易日范围（{num_trading_days} 个交易日）")
    try:
        trading_df, date_col = get_trading_dates(stock_df, start_date, num_trading_days)
        print(f"交易日范围: {trading_df[date_col].min()} 到 {trading_df[date_col].max()}")
        print(f"共 {len(trading_df)} 个交易日")
        
        # 重要：检查processed_df中是否有足够的历史数据
        # 需要确保第一个交易日对应的索引至少是sequence_length（60）
        # processed_df已经在步骤3中处理过了，这里直接使用
        
        first_trading_date = trading_df[date_col].min()
        first_date_dt = pd.to_datetime(first_trading_date)
        
        # 在processed_df中找到第一个交易日的索引
        mask = processed_df['日期'] == first_date_dt
        if not mask.any():
            if hasattr(processed_df['日期'].iloc[0], 'date'):
                mask = processed_df['日期'].dt.date == first_date_dt.date()
            else:
                date_str_only = first_trading_date.strftime('%Y-%m-%d') if hasattr(first_trading_date, 'strftime') else str(first_trading_date).split(' ')[0]
                mask = processed_df['日期'].astype(str).str.startswith(date_str_only)
        
        if mask.any():
            first_trading_idx = processed_df[mask].index[0]
            sequence_length = train_info['sequence_length']
            
            if first_trading_idx < sequence_length:
                print(f"\n⚠️  警告: 第一个交易日（索引={first_trading_idx}）之前的历史数据不足（需要{sequence_length}天）")
                print(f"   这会导致前 {sequence_length - first_trading_idx} 个交易日无法进行预测")
                print(f"   建议：调整起始日期，确保有足够的历史数据")
                
                # 自动调整起始日期，确保第一个交易日有足够的历史数据
                if start_date is None:
                    # 如果用户没有指定起始日期，自动调整
                    min_required_idx = sequence_length
                    if len(processed_df) > min_required_idx:
                        # 找到第一个可以预测的日期
                        adjusted_start_date = processed_df.iloc[min_required_idx]['日期']
                        print(f"   自动调整起始日期为: {adjusted_start_date.strftime('%Y-%m-%d')}")
                        
                        # 重新获取交易日范围
                        trading_df, date_col = get_trading_dates(stock_df, adjusted_start_date.strftime('%Y-%m-%d'), num_trading_days)
                        print(f"   调整后的交易日范围: {trading_df[date_col].min()} 到 {trading_df[date_col].max()}")
                        print(f"   调整后共 {len(trading_df)} 个交易日")
                    else:
                        print(f"   ❌ 错误: 历史数据总量不足（只有{len(processed_df)}天，需要至少{sequence_length + num_trading_days}天）")
                        return None
                else:
                    print(f"   ⚠️  请手动调整 --start-date 参数，确保起始日期有足够的历史数据")
        else:
            print(f"⚠️  警告: 无法在processed_df中找到第一个交易日 {first_trading_date}")
            
    except Exception as e:
        print(f"获取交易日失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤5：加载模型
    print(f"\n[步骤5] 加载模型")
    try:
        model, price_scaler, volume_scaler, device = load_model(train_info['model_path'])
        print("模型加载成功")
    except Exception as e:
        print(f"加载模型失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤6：对每一天进行预测
    print(f"\n[步骤6] 对每个交易日进行预测")
    predictions = []
    
    # 确保processed_df已经准备好（在步骤4中已经处理过）
    # processed_df已经在步骤4中处理过了，这里不需要重复处理
    
    # 导入torch用于预测
    import torch
    
    # 对序列进行scaler变换并预测的辅助函数
    def predict_with_model(close_seq, volume_seq, model, price_scaler, volume_scaler, device):
        """使用已加载的模型进行预测"""
        # 数据预处理
        close_scaled = price_scaler.transform(close_seq.reshape(-1, 1)).reshape(close_seq.shape)
        volume_scaled = volume_scaler.transform(volume_seq.reshape(-1, 1)).reshape(volume_seq.shape)
        
        # 组合特征
        X = np.column_stack([close_scaled, volume_scaled])
        X = X.reshape(1, len(close_seq), 2)
        
        # 转换为tensor
        X_tensor = torch.FloatTensor(X).to(device)
        
        # 推理
        with torch.no_grad():
            outputs = model(X_tensor)
            _, predicted = torch.max(outputs.data, 1)
            prediction = predicted.cpu().numpy()[0]
            
            probabilities = torch.softmax(outputs, dim=1)
            probs = probabilities.cpu().numpy()[0]
            
            probability_down = float(probs[0])
            probability_up = float(probs[1])
        
        return {
            'prediction': int(prediction),
            'probability_up': probability_up,
            'probability_down': probability_down
        }
    
    for idx, row in trading_df.iterrows():
        date = row[date_col]
        if hasattr(date, 'strftime'):
            date_str = date.strftime('%Y-%m-%d')
            date_dt = pd.to_datetime(date)
        else:
            date_str = str(date)
            date_dt = pd.to_datetime(date)
        
        # 在processed_df中找到对应的索引
        # 使用日期匹配，考虑可能的时区或时间部分差异
        try:
            # 先尝试精确匹配
            mask = processed_df['日期'] == date_dt
            if not mask.any():
                # 如果精确匹配失败，尝试只比较日期部分（忽略时间）
                if hasattr(processed_df['日期'].iloc[0], 'date'):
                    mask = processed_df['日期'].dt.date == date_dt.date()
                else:
                    # 尝试字符串匹配
                    date_str_only = date_str.split(' ')[0]  # 只取日期部分
                    mask = processed_df['日期'].astype(str).str.startswith(date_str_only)
            
            if not mask.any():
                print(f"警告: 在processed_df中找不到日期 {date_str}")
                continue
        except Exception as e:
            print(f"警告: 日期匹配失败 {date_str}: {e}")
            continue
        
        target_idx = processed_df[mask].index[0]
        
        # 创建序列
        seq_result = create_sequence_for_date(processed_df, target_idx, train_info['sequence_length'])
        if seq_result is None:
            print(f"⚠️  警告: 日期 {date_str} (索引={target_idx}) 数据不足，无法创建序列（需要至少{train_info['sequence_length']}天的历史数据）")
            print(f"   该日期将被跳过，不会进行预测和交易")
            continue
        
        close_seq, volume_seq = seq_result
        
        # 进行预测（使用已加载的模型）
        try:
            result = predict_with_model(close_seq, volume_seq, model, price_scaler, volume_scaler, device)
            predictions.append({
                'date': date_str,
                'prediction': result['prediction'],
                'probability_up': result['probability_up'],
                'probability_down': result['probability_down']
            })
            if (idx + 1) % 10 == 0 or idx == len(trading_df) - 1:
                print(f"  进度: {idx + 1}/{len(trading_df)} - {date_str}: {'涨' if result['prediction'] == 1 else '跌'} (概率: {result['probability_up']:.2%})")
        except Exception as e:
            print(f"警告: 日期 {date_str} 预测失败: {e}")
            continue
    
    print(f"\n共完成 {len(predictions)} 个交易日的预测")
    
    # 步骤7：模拟交易
    print(f"\n[步骤7] 模拟交易过程（初始资金: {initial_capital} 元）")
    try:
        trade_result = simulate_trading(trading_df, predictions, date_col, initial_capital)
        print(f"初始资金: {trade_result['initial_capital']:.2f} 元")
        print(f"最终价值: {trade_result['final_value']:.2f} 元")
        print(f"利润: {trade_result['profit']:.2f} 元")
        print(f"收益率: {trade_result['profit_rate']:.2f}%")
        print(f"交易次数: {trade_result['total_trades']} 次")
    except Exception as e:
        print(f"模拟交易失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤8：保存结果
    print(f"\n[步骤8] 保存结果")
    try:
        base_output_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 创建输出文件夹：<股票码>_<投入金额>_<日期年月起止>
        stock_code = train_info['stock_code']
        capital_str = f"{int(initial_capital)}"
        
        # 获取日期范围（年月格式，如 202509-202512）
        start_date_obj = trading_df[date_col].min()
        end_date_obj = trading_df[date_col].max()
        if hasattr(start_date_obj, 'strftime'):
            date_range = f"{start_date_obj.strftime('%Y%m')}-{end_date_obj.strftime('%Y%m')}"
        else:
            # 转换为datetime对象
            start_date_obj = pd.to_datetime(start_date_obj)
            end_date_obj = pd.to_datetime(end_date_obj)
            date_range = f"{start_date_obj.strftime('%Y%m')}-{end_date_obj.strftime('%Y%m')}"
        
        # 创建输出文件夹
        output_dir = os.path.join(base_output_dir, f"{stock_code}_{capital_str}_{date_range}")
        os.makedirs(output_dir, exist_ok=True)
        print(f"输出目录: {output_dir}")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = os.path.join(output_dir, f"profit_result_{stock_code}_{timestamp}.json")
        
        # 获取收盘价列名
        close_col = None
        for col in trading_df.columns:
            if '收盘' in col or 'close' in col.lower():
                close_col = col
                break
        
        # 创建每日数据（包含收盘价和实际涨跌）
        daily_data = []
        prev_close = None
        
        for idx, row in trading_df.iterrows():
            date = row[date_col]
            if hasattr(date, 'strftime'):
                date_str = date.strftime('%Y-%m-%d')
            else:
                date_str = str(date)
            
            close_price = float(row[close_col]) if close_col else None
            
            # 计算实际涨跌（与前一天比较）
            if prev_close is not None and close_price is not None:
                actual_change = 1 if close_price > prev_close else 0
                price_change_pct = ((close_price - prev_close) / prev_close) * 100
            else:
                actual_change = None
                price_change_pct = None
            
            daily_data.append({
                'date': date_str,
                'close_price': close_price,
                'actual_change': actual_change,  # 1=涨, 0=跌
                'price_change_pct': price_change_pct
            })
            
            # 更新前一天的收盘价
            prev_close = close_price
        
        # 为predictions添加实际涨跌信息
        predictions_with_actual = []
        for pred in predictions:
            # 找到对应的daily_data
            matching_daily = next((d for d in daily_data if d['date'] == pred['date']), None)
            if matching_daily:
                pred_copy = pred.copy()
                pred_copy['actual_change'] = matching_daily['actual_change']
                pred_copy['close_price'] = matching_daily['close_price']
                pred_copy['price_change_pct'] = matching_daily['price_change_pct']
                predictions_with_actual.append(pred_copy)
            else:
                predictions_with_actual.append(pred)
        
        result_data = {
            'stock_code': train_info['stock_code'],
            'log_folder': log_folder_path,
            'num_trading_days': num_trading_days,
            'start_date': start_date,
            'trading_period': {
                'start': str(trading_df[date_col].min()),
                'end': str(trading_df[date_col].max())
            },
            'initial_capital': initial_capital,
            'final_value': trade_result['final_value'],
            'profit': trade_result['profit'],
            'profit_rate': trade_result['profit_rate'],
            'total_trades': trade_result['total_trades'],
            'predictions': predictions_with_actual,
            'trade_records': trade_result['trade_records'],
            'daily_data': daily_data  # 添加每日数据
        }
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result_data, f, indent=2, ensure_ascii=False, default=str)
        
        # 同时保存CSV格式的交易记录
        if trade_result['trade_records']:
            trade_df = pd.DataFrame(trade_result['trade_records'])
            csv_file = os.path.join(output_dir, f"trade_records_{stock_code}_{timestamp}.csv")
            trade_df.to_csv(csv_file, index=False, encoding='utf-8-sig')
            print(f"交易记录已保存到: {csv_file}")
        
        # 确保文件确实存在
        if not os.path.exists(output_file):
            raise FileNotFoundError(f"结果文件保存失败: {output_file}")
        
    except Exception as e:
        print(f"保存结果失败: {e}")
        import traceback
        traceback.print_exc()
        # 即使保存失败，也返回结果，但 output_file 为 None
        output_file = None
        output_dir = None
    
    print("\n" + "="*60)
    print("利润计算完成!")
    print("="*60)
    
    result_dict = {
        'stock_code': train_info['stock_code'],
        'initial_capital': initial_capital,
        'final_value': trade_result['final_value'],
        'profit': trade_result['profit'],
        'profit_rate': trade_result['profit_rate'],
        'total_trades': trade_result['total_trades'],
        'output_file': output_file if 'output_file' in locals() and output_file else None,
        'output_dir': output_dir if 'output_dir' in locals() and output_dir else None
    }
    
    return result_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='基于模型预测的股票交易利润计算')
    parser.add_argument('log_folder', type=str,
                       help='训练日志目录路径，例如: logs/20251225_175236_train_518880_20251225')
    parser.add_argument('--days', type=int, default=30,
                       help='交易日数量，默认30天')
    parser.add_argument('--start-date', type=str, default=None,
                       help='起始日期（格式: YYYY-MM-DD），如果不指定则从最后一天往前推')
    parser.add_argument('--capital', type=float, default=10000,
                       help='初始资金，默认10000元')
    parser.add_argument('--no-visualize', action='store_true',
                       help='不自动生成可视化图表（默认会自动生成）')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    if os.path.isabs(args.log_folder):
        log_folder_path = args.log_folder
    else:
        # 相对路径：先尝试相对于当前工作目录
        current_dir = os.getcwd()
        log_folder_path = os.path.abspath(os.path.join(current_dir, args.log_folder))
        
        # 如果不存在，尝试相对于项目根目录
        if not os.path.exists(log_folder_path):
            project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            log_folder_path = os.path.abspath(os.path.join(project_root, args.log_folder))
    
    # 确保路径存在
    if not os.path.exists(log_folder_path):
        print(f"错误: 训练日志目录不存在: {log_folder_path}")
        print(f"当前工作目录: {os.getcwd()}")
        print(f"输入的路径: {args.log_folder}")
        sys.exit(1)
    
    result = calculate_profit(
        log_folder_path=log_folder_path,
        num_trading_days=args.days,
        start_date=args.start_date,
        initial_capital=args.capital
    )
    
    if result and result.get('output_file'):
        print("\n计算完成！")
        
        # 自动生成可视化图表
        if not args.no_visualize:
            print("\n" + "="*60)
            print("开始生成可视化图表...")
            print("="*60)
            
            try:
                # 检查输出文件是否存在
                if not os.path.exists(result['output_file']):
                    raise FileNotFoundError(f"结果文件不存在: {result['output_file']}")
                
                # 导入可视化模块
                current_dir = os.path.dirname(os.path.abspath(__file__))
                visualize_module_path = os.path.join(current_dir, 'visualize_profit.py')
                
                if not os.path.exists(visualize_module_path):
                    raise FileNotFoundError(f"可视化模块不存在: {visualize_module_path}")
                
                # 动态导入可视化模块
                import importlib.util
                spec = importlib.util.spec_from_file_location("visualize_profit", visualize_module_path)
                visualize_module = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(visualize_module)
                
                # 调用可视化函数
                output_dir = result.get('output_dir')
                if output_dir:
                    visualize_module.visualize_all(result['output_file'], output_dir)
                else:
                    visualize_module.visualize_all(result['output_file'])
                
                print("\n" + "="*60)
                print("所有任务完成！")
                print("="*60)
            except Exception as e:
                print(f"\n生成可视化图表失败: {e}")
                import traceback
                traceback.print_exc()
                print("\n可以稍后手动运行可视化脚本:")
                print(f"python {os.path.join(os.path.dirname(os.path.abspath(__file__)), 'visualize_profit.py')} {result['output_file']}")
        else:
            print("\n跳过可视化（使用 --no-visualize 参数）")
            print(f"可以稍后运行: python visualize_profit.py {result['output_file']}")
    elif result:
        print("\n计算完成！但未生成输出文件")
    else:
        print("\n计算失败！")
        sys.exit(1)

