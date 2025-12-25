"""
获取股票数据的函数
支持任意股票（ETF或普通股票）
"""
import akshare as ak
import pandas as pd
from datetime import datetime


def get_stock_data(stock_code):
    """
    获取股票的全部历史数据（最新数据）
    
    Args:
        stock_code: 股票代码，如 "518880" (ETF) 或 "000001" (股票)
    
    Returns:
        pandas.DataFrame: 包含股票数据的DataFrame
            - 日期
            - 收盘
            - 成交量
            - 其他字段
    """
    print(f"正在获取股票数据: {stock_code}...")
    
    try:
        # 尝试ETF方式
        try:
            df = ak.fund_etf_hist_em(symbol=stock_code)
            print(f"成功获取ETF数据: {stock_code}")
        except:
            # 如果不是ETF，尝试股票方式
            try:
                df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="")
                print(f"成功获取股票数据: {stock_code}")
            except Exception as e:
                print(f"尝试其他方式获取数据: {e}")
                # 如果还是失败，尝试使用通用接口
                df = ak.stock_zh_a_hist(symbol=stock_code, period="daily", adjust="qfq")
        
        if df.empty:
            raise ValueError(f"获取到的数据为空: {stock_code}")
        
        # 确保日期列存在
        if '日期' not in df.columns and 'date' in df.columns:
            df.rename(columns={'date': '日期'}, inplace=True)
        
        # 按日期排序
        if '日期' in df.columns:
            df['日期'] = pd.to_datetime(df['日期'])
            df = df.sort_values(by='日期').reset_index(drop=True)
        
        print(f"成功获取 {len(df)} 条数据")
        print(f"数据日期范围: {df['日期'].min()} 到 {df['日期'].max()}")
        
        return df
    
    except Exception as e:
        print(f"获取股票数据失败: {e}")
        raise


if __name__ == "__main__":
    # 测试函数
    df = get_stock_data("518880")
    print("\n数据预览:")
    print(df.head())
    print("\n数据列名:")
    print(df.columns.tolist())

