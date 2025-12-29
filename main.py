"""
股票LSTM预测模型主程序
"""
import os
import sys
import yaml
from datetime import datetime
from pathlib import Path

# 警告：code目录与Python标准库的code模块冲突
# 临时解决方案：在导入torch之前，先确保标准库的code模块可用
# 最佳解决方案：重命名code目录为其他名称（如src、modules等）

# 添加code目录到路径
code_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'code')
if code_dir not in sys.path:
    sys.path.insert(0, code_dir)

# 直接导入项目模块（因为code目录已在sys.path中）
from data_fetcher import fetch_stock_data
from data_processor import process_stock_data
from train_lstm import train_lstm_model
from inference import inference_model


def load_config(config_path=None):
    """加载配置文件"""
    if config_path is None:
        config_path = os.path.join(os.path.dirname(__file__), 'config', 'config.yaml')
    
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"配置文件不存在: {config_path}")
    
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    return config


def main():
    """主函数"""
    print("=" * 60)
    print("股票LSTM预测模型")
    print("=" * 60)

    # 加载配置
    config = load_config()
    print(f"\n配置已加载")
    
    # 获取股票代码和截止日期
    stock_code = config['stock']['stock_code']
    end_date = config['stock']['end_date']
    
    if end_date is None:
        end_date = datetime.now().strftime("%Y%m%d")
    
    print(f"\n股票代码: {stock_code}")
    print(f"截止日期: {end_date}")
    print(f"模式: {config['mode']}")
    
    # 确定路径
    base_dir = os.path.dirname(__file__)
    raw_data_dir = os.path.join(base_dir, config['data']['raw_data_dir'])
    processed_data_dir = os.path.join(base_dir, config['data']['processed_data_dir'])
    log_dir = os.path.join(base_dir, config['logging']['log_dir'])
    
    # 创建必要的目录
    os.makedirs(raw_data_dir, exist_ok=True)
    os.makedirs(processed_data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    
    mode = config['mode']
    
    if mode == 'train':
        print("\n" + "=" * 60)
        print("训练模式")
        print("=" * 60)
        
        # 步骤1: 获取数据（如果需要）
        raw_data_filename = f"{stock_code}_{end_date}.csv"
        raw_data_path = os.path.join(raw_data_dir, raw_data_filename)
        
        if not os.path.exists(raw_data_path) and config['data']['auto_fetch_data']:
            print(f"\n步骤1: 获取股票数据...")
            try:
                fetch_stock_data(stock_code, end_date, raw_data_dir)
                print("✓ 数据获取完成")
            except Exception as e:
                print(f"✗ 数据获取失败: {e}")
                return
        elif os.path.exists(raw_data_path):
            print(f"\n步骤1: 使用已有数据文件: {raw_data_path}")
        else:
            print(f"\n✗ 数据文件不存在且未启用自动获取: {raw_data_path}")
            return
        
        # 步骤2: 处理数据
        print(f"\n步骤2: 处理数据...")
        try:
            process_result = process_stock_data(
                raw_data_path=raw_data_path,
                stock_code=stock_code,
                end_date=end_date,
                sequence_length=config['data']['sequence_length'],
                train_ratio=config['data']['train_ratio'],
                val_ratio=config['data']['val_ratio'],
                output_dir=processed_data_dir
            )
            print(f"✓ 数据处理完成")
            print(f"  输出目录: {process_result['output_folder']}")
            print(f"  训练集: {process_result['train_size']} 个序列")
            print(f"  验证集: {process_result['val_size']} 个序列")
        except Exception as e:
            print(f"✗ 数据处理失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        # 步骤3: 训练模型
        print(f"\n步骤3: 训练模型...")
        try:
            training_config = {
                'batch_size': config['training']['batch_size'],
                'num_epochs': config['training']['num_epochs'],
                'learning_rate': config['training']['learning_rate'],
                'hidden_size': config['training']['hidden_size'],
                'early_stopping_patience': config['training']['early_stopping_patience'],
                'random_seed': config['training']['random_seed'],
                'use_gpu': config['training']['use_gpu']
            }
            
            model_path = train_lstm_model(
                data_folder=process_result['output_folder'],
                stock_code=stock_code,
                end_date=end_date,
                config=training_config,
                log_dir=log_dir
            )
            print(f"✓ 模型训练完成")
            print(f"  模型路径: {model_path}")
        except Exception as e:
            print(f"✗ 模型训练失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n" + "=" * 60)
        print("训练完成!")
        print("=" * 60)
        print(f"模型已保存到: {model_path}")
        print(f"训练日志保存在: {log_dir}")
        
    elif mode == 'inference':
        print("\n" + "=" * 60)
        print("推理模式")
        print("=" * 60)
        
        # 检查配置
        checkpoint_path = config['inference']['checkpoint_path']
        data_path = config['inference']['data_path']
        
        if checkpoint_path is None:
            print("✗ 错误: 推理模式需要指定 checkpoint_path")
            return
        
        if data_path is None:
            print("✗ 错误: 推理模式需要指定 data_path")
            return
        
        # 转换为绝对路径
        if not os.path.isabs(checkpoint_path):
            checkpoint_path = os.path.join(base_dir, checkpoint_path)
        
        if not os.path.isabs(data_path):
            data_path = os.path.join(base_dir, data_path)
        
        # 检查文件是否存在
        if not os.path.exists(checkpoint_path):
            print(f"✗ 错误: 模型文件不存在: {checkpoint_path}")
            return
        
        if not os.path.exists(data_path):
            print(f"✗ 错误: 数据文件不存在: {data_path}")
            return
        
        print(f"模型路径: {checkpoint_path}")
        print(f"数据路径: {data_path}")
        
        # 运行推理
        print(f"\n运行推理...")
        try:
            inference_config = {
                'batch_size': config['inference']['batch_size'],
                'use_gpu': config['inference']['use_gpu']
            }
            
            result = inference_model(
                model_path=checkpoint_path,
                data_path=data_path,
                stock_code=stock_code,
                end_date=end_date,
                config=inference_config,
                log_dir=log_dir
            )
            
            print(f"✓ 推理完成")
            print(f"  结果文件: {result['results_path']}")
            print(f"  日志目录: {result['log_folder']}")
            if result['accuracy'] is not None:
                print(f"  准确率: {result['accuracy']*100:.2f}%")
        except Exception as e:
            print(f"✗ 推理失败: {e}")
            import traceback
            traceback.print_exc()
            return
        
        print("\n" + "=" * 60)
        print("推理完成!")
        print("=" * 60)
        print(f"推理结果保存在: {result['log_folder']}")
        print(f"结果CSV文件: {result['results_path']}")
        
    else:
        print(f"✗ 错误: 未知的模式 '{mode}'，应该是 'train' 或 'inference'")


if __name__ == "__main__":
    main()

