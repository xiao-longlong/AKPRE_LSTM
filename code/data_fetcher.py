"""
股票数据获取模块
模拟 get_glod_etf.py，支持任意股票的数据读取
"""
import akshare as ak
import pandas as pd
import os
from datetime import datetime


def fetch_stock_data(stock_code, end_date=None, save_dir=None):
    """
    获取股票数据
    
    Args:
        stock_code: 股票代码，如 "518880" (ETF) 或 "000001" (股票)
        end_date: 截止日期，格式 "YYYYMMDD"，如果为None则使用当前日期
        save_dir: 保存目录，如果为None则不保存
    
    Returns:
        DataFrame: 股票数据
    """
    try:
        # 判断是ETF、股票还是期货
        # ETF通常使用 fund_etf_hist_em
        # 股票使用 stock_zh_a_hist
        # 期货使用 futures_zh_daily_sina 或 futures_main_sina
        
        df = None
        
        # 尝试ETF方式
        try:
            df = ak.fund_etf_hist_em(symbol=stock_code)
            if not df.empty:
                print(f"成功获取ETF数据: {stock_code}")
            else:
                df = None  # 如果为空，继续尝试其他方式
        except:
            df = None  # 如果异常，继续尝试其他方式
        
        # 如果ETF方式失败或返回空数据，尝试股票方式
        if df is None or df.empty:
            try:
                # 需要添加市场前缀，如 "000001" -> "000001.SZ" 或 "600000" -> "600000.SH"
                if stock_code.startswith('0') or stock_code.startswith('3'):
                    ts_code = f"{stock_code}.SZ"
                elif stock_code.startswith('6'):
                    ts_code = f"{stock_code}.SH"
                else:
                    ts_code = stock_code
                
                # 获取股票历史数据
                df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="")
                if not df.empty:
                    print(f"成功获取股票数据: {stock_code}")
                else:
                    df = None  # 如果为空，继续尝试其他方式
            except Exception as e:
                df = None  # 如果异常，继续尝试其他方式
        
        # 如果股票方式也失败，尝试期货方式
        if df is None or df.empty:
            try:
                # 尝试获取期货主连数据
                # NI0 等期货代码需要使用 futures_main_sina 或 futures_zh_daily_sina
                df = ak.futures_zh_daily_sina(symbol=stock_code)
                if not df.empty:
                    print(f"成功获取期货数据: {stock_code}")
                else:
                    df = None
            except Exception as e:
                try:
                    # 尝试另一种期货接口
                    df = ak.futures_main_sina(symbol=stock_code)
                    if not df.empty:
                        print(f"成功获取期货主连数据: {stock_code}")
                    else:
                        df = None
                except:
                    df = None
        
        # 如果所有方式都失败，抛出异常
        if df is None or df.empty:
            raise ValueError(f"获取到的数据为空: {stock_code}，请检查代码是否正确（ETF/股票/期货）")
        
        # 确保日期列存在
        if '日期' not in df.columns and 'date' in df.columns:
            df.rename(columns={'date': '日期'}, inplace=True)
        
        # 按日期排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values(by='日期').reset_index(drop=True)
        
        # 如果指定了保存目录，保存数据
        if save_dir is not None:
            os.makedirs(save_dir, exist_ok=True)
            
            # 确定截止日期
            if end_date is None:
                end_date = datetime.now().strftime("%Y%m%d")
            
            # 文件名格式: <股票码_截止日期>.csv
            filename = f"{stock_code}_{end_date}.csv"
            filepath = os.path.join(save_dir, filename)
            
            df.to_csv(filepath, index=False, encoding='utf-8-sig')
            print(f"数据已保存到: {filepath}")
            print(f"共 {len(df)} 条记录")
        
        return df
    
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        raise


if __name__ == "__main__":
    # 测试
    df = fetch_stock_data("518880", save_dir="../data/raw")
    print(df.head())

