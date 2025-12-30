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

