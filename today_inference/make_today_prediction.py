"""
今天股票涨跌预测决策脚本
基于训练日志目录，完成一次推理预测
"""
import os
import sys
import json
import argparse
from datetime import datetime

# 添加code目录到路径
current_dir = os.path.dirname(os.path.abspath(__file__))
code_dir = os.path.join(current_dir, 'code')
sys.path.append(code_dir)

from get_stock_data import get_stock_data
from process_and_create_sequence import process_and_get_today_sequence
from predict_today import predict_today


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
        # 如果model_path不在train_info中，尝试从log_folder推断
        model_path = os.path.join(log_folder_path, 'best_model.pth')
    
    # 确保模型文件存在
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    
    # 获取股票代码
    stock_code = train_info.get('stock_code')
    if stock_code is None:
        raise ValueError("训练信息中缺少股票代码")
    
    # 获取序列长度（从配置中获取，如果没有则使用默认值60）
    # 注意：序列长度通常在数据预处理时确定，这里使用默认值60
    # 如果需要，可以从原始配置文件中读取
    sequence_length = 60  # 默认值
    
    return {
        'stock_code': stock_code,
        'model_path': model_path,
        'sequence_length': sequence_length,
        'train_info': train_info
    }


def get_current_date():
    """
    获取系统当前日期
    
    Returns:
        str: 当前日期字符串，格式 YYYYMMDD
    """
    today = datetime.now()
    date_str = today.strftime('%Y%m%d')
    return date_str


def main(log_folder_path):
    """
    主函数：依次调用所有功能模块完成今日预测
    
    Args:
        log_folder_path: 训练日志目录路径
    """
    print("="*60)
    print("今天股票涨跌预测决策系统")
    print("="*60)
    
    # 步骤1：加载训练信息
    print("\n[步骤1] 加载训练信息")
    try:
        train_info = load_train_info(log_folder_path)
        print(f"训练日志目录: {log_folder_path}")
        print(f"股票代码: {train_info['stock_code']}")
        print(f"模型路径: {train_info['model_path']}")
        print(f"训练准确率: {train_info['train_info'].get('best_val_accuracy', 'N/A'):.2f}%")
    except Exception as e:
        print(f"加载训练信息失败: {e}")
        return None
    
    # 步骤2：获取系统日期
    print("\n[步骤2] 获取系统日期")
    current_date = get_current_date()
    print(f"当前系统日期: {current_date}")
    
    # 步骤3：获取全部时间的股票数据
    print(f"\n[步骤3] 获取股票 {train_info['stock_code']} 的全部历史数据")
    try:
        stock_df = get_stock_data(train_info['stock_code'])
        print(f"成功获取 {len(stock_df)} 条历史数据")
    except Exception as e:
        print(f"获取数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤4：处理数据并生成今天的序列
    print(f"\n[步骤4] 处理数据并生成今天的预测序列（序列长度: {train_info['sequence_length']}）")
    try:
        close_seq, volume_seq, predict_date = process_and_get_today_sequence(
            stock_df, 
            sequence_length=train_info['sequence_length']
        )
        print(f"序列准备完成，预测目标日期: {predict_date}")
    except Exception as e:
        print(f"处理数据失败: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # 步骤5：使用模型推理今天的股票涨跌
    print(f"\n[步骤5] 使用LSTM模型进行推理预测")
    try:
        result = predict_today(close_seq, volume_seq, train_info['model_path'])
        
        # 输出最终结果
        print("\n" + "="*60)
        print("预测结果汇总")
        print("="*60)
        print(f"股票代码: {train_info['stock_code']}")
        print(f"预测日期: {predict_date}")
        print(f"预测结果: {'涨 (1)' if result['prediction'] == 1 else '跌 (0)'}")
        print(f"涨的概率: {result['probability_up']:.4f} ({result['probability_up']*100:.2f}%)")
        print(f"跌的概率: {result['probability_down']:.4f} ({result['probability_down']*100:.2f}%)")
        print(f"模型来源: {log_folder_path}")
        print("="*60)
        
        # 返回结果
        return {
            'stock_code': train_info['stock_code'],
            'date': predict_date,
            'prediction': result['prediction'],
            'prediction_text': '涨' if result['prediction'] == 1 else '跌',
            'probability_up': result['probability_up'],
            'probability_down': result['probability_down'],
            'model_path': train_info['model_path'],
            'log_folder': log_folder_path
        }
        
    except Exception as e:
        print(f"推理预测失败: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='今天股票涨跌预测决策系统')
    parser.add_argument('log_folder', type=str, 
                       help='训练日志目录路径，例如: logs/20251225_175236_train_518880_20251225')
    
    args = parser.parse_args()
    
    # 转换为绝对路径
    if not os.path.isabs(args.log_folder):
        # 尝试相对于项目根目录
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_folder_path = os.path.join(project_root, args.log_folder)
    else:
        log_folder_path = args.log_folder
    
    # 确保路径存在
    if not os.path.exists(log_folder_path):
        print(f"错误: 训练日志目录不存在: {log_folder_path}")
        sys.exit(1)
    
    result = main(log_folder_path)
    if result:
        print("\n预测完成！")
    else:
        print("\n预测失败！")
        sys.exit(1)

