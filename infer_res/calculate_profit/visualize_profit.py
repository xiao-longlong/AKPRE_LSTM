"""
Profit calculation result visualization script
Draws three charts:
1. Gold price change curve by date
2. Comparison of total assets: trading strategy vs buy-and-hold strategy
3. Prediction accuracy convergence curve over time
"""
import os
import sys
import json
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime

# 设置字体（使用英文，不需要中文字体）
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['axes.unicode_minus'] = False


def load_result_file(json_file):
    """
    加载结果JSON文件
    
    Args:
        json_file: JSON文件路径
    
    Returns:
        dict: 结果数据
    """
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def plot_price_changes(data, output_file=None):
    """
    Plot the first chart: Gold price change curve by date
    
    Args:
        data: Result data dictionary
        output_file: Output file path, if None then don't save
    """
    daily_data = data.get('daily_data', [])
    if not daily_data:
        print("Warning: No daily data available, cannot plot price changes")
        return None
    
    # Extract data
    dates = []
    close_prices = []
    price_changes = []
    
    for item in daily_data:
        if item.get('close_price') is not None:
            dates.append(pd.to_datetime(item['date']))
            close_prices.append(item['close_price'])
            # Handle None values for price_change_pct (first day has no previous day)
            pct_change = item.get('price_change_pct')
            price_changes.append(pct_change if pct_change is not None else 0)
    
    if not dates:
        print("Warning: No valid price data")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Closing price curve
    ax1.plot(dates, close_prices, linewidth=2, color='#2E86AB', label='Closing Price')
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Closing Price (CNY)', fontsize=12)
    ax1.set_title('Gold Price Trend', fontsize=14, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot 2: Daily change percentage
    # Filter out None values for plotting
    valid_indices = [i for i, x in enumerate(price_changes) if x is not None]
    valid_dates = [dates[i] for i in valid_indices]
    valid_changes = [price_changes[i] for i in valid_indices]
    colors = ['red' if x < 0 else 'green' for x in valid_changes]
    ax2.bar(valid_dates, valid_changes, color=colors, alpha=0.6, width=0.8)
    ax2.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax2.set_xlabel('Date', fontsize=12)
    ax2.set_ylabel('Daily Change (%)', fontsize=12)
    ax2.set_title('Daily Price Change Percentage', fontsize=14, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    # Format x-axis dates
    ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Price changes chart saved to: {output_file}")
    
    return fig


def plot_asset_comparison(data, output_file=None):
    """
    Plot the second chart: Comparison of total assets (trading strategy vs buy-and-hold)
    
    Args:
        data: Result data dictionary
        output_file: Output file path, if None then don't save
    """
    trade_records = data.get('trade_records', [])
    daily_data = data.get('daily_data', [])
    initial_capital = data.get('initial_capital', 10000)
    
    if not trade_records or not daily_data:
        print("Warning: No trade records or daily data available, cannot plot asset comparison")
        return None
    
    # Extract total assets from trade records
    trade_dates = []
    trade_values = []
    
    for record in trade_records:
        trade_dates.append(pd.to_datetime(record['date']))
        trade_values.append(record['total_value'])
    
    # Calculate buy-and-hold strategy assets
    if len(daily_data) >= 2:
        first_close = daily_data[0].get('close_price')
        if first_close:
            hold_dates = []
            hold_values = []
            
            for item in daily_data:
                if item.get('close_price') is not None:
                    date = pd.to_datetime(item['date'])
                    close_price = item['close_price']
                    # Calculate value if bought on first day and held until now
                    shares = initial_capital / first_close
                    value = shares * close_price
                    hold_dates.append(date)
                    hold_values.append(value)
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot trading strategy total assets
    ax.plot(trade_dates, trade_values, linewidth=2.5, color='#2E86AB', 
            label='Trading Strategy Total Assets', marker='o', markersize=4)
    
    # Plot buy-and-hold strategy total assets
    if 'hold_dates' in locals() and hold_dates:
        ax.plot(hold_dates, hold_values, linewidth=2.5, color='#A23B72', 
                label='Buy-and-Hold Strategy Total Assets', linestyle='--', marker='s', markersize=4)
    
    # Add initial capital reference line
    ax.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1, 
               label=f'Initial Capital ({initial_capital:.0f} CNY)', alpha=0.7)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Total Assets (CNY)', fontsize=12)
    ax.set_title('Trading Strategy vs Buy-and-Hold Strategy Comparison', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    # Format x-axis dates
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(trade_dates)//10)))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Add final profit annotation
    final_trade_value = trade_values[-1] if trade_values else initial_capital
    final_hold_value = hold_values[-1] if 'hold_values' in locals() and hold_values else initial_capital
    
    trade_profit = final_trade_value - initial_capital
    hold_profit = final_hold_value - initial_capital
    
    text_str = f'Trading Strategy Final: {final_trade_value:.2f} CNY (Profit: {trade_profit:.2f} CNY, {trade_profit/initial_capital*100:.2f}%)\n'
    text_str += f'Buy-and-Hold Final: {final_hold_value:.2f} CNY (Profit: {hold_profit:.2f} CNY, {hold_profit/initial_capital*100:.2f}%)'
    
    ax.text(0.02, 0.98, text_str, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Asset comparison chart saved to: {output_file}")
    
    return fig


def plot_accuracy_convergence(data, output_file=None):
    """
    Plot the third chart: Prediction accuracy convergence curve over time
    
    Args:
        data: Result data dictionary
        output_file: Output file path, if None then don't save
    """
    predictions = data.get('predictions', [])
    
    if not predictions:
        print("Warning: No prediction data available, cannot plot accuracy convergence")
        return None
    
    # Calculate daily accuracy (cumulative accuracy)
    dates = []
    cumulative_correct = []
    cumulative_total = []
    accuracy_rates = []
    
    correct_count = 0
    total_count = 0
    
    for pred in predictions:
        date = pd.to_datetime(pred['date'])
        prediction = pred.get('prediction')
        actual_change = pred.get('actual_change')
        
        # Only calculate when actual change data is available
        if actual_change is not None:
            total_count += 1
            if prediction == actual_change:
                correct_count += 1
            
            dates.append(date)
            cumulative_correct.append(correct_count)
            cumulative_total.append(total_count)
            accuracy_rates.append(correct_count / total_count if total_count > 0 else 0)
    
    if not dates:
        print("Warning: No valid prediction-actual comparison data, cannot calculate accuracy")
        return None
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
    
    # Subplot 1: Cumulative accuracy curve
    ax1.plot(dates, accuracy_rates, linewidth=2.5, color='#06A77D', marker='o', markersize=4)
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1, label='Random Guess (50%)', alpha=0.7)
    ax1.set_xlabel('Date', fontsize=12)
    ax1.set_ylabel('Cumulative Accuracy', fontsize=12)
    ax1.set_title('Prediction Accuracy Convergence Over Time', fontsize=14, fontweight='bold')
    ax1.set_ylim([0, 1])
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=10)
    
    # Add final accuracy annotation
    final_accuracy = accuracy_rates[-1] if accuracy_rates else 0
    ax1.text(0.98, 0.02, f'Final Accuracy: {final_accuracy:.2%}', 
             transform=ax1.transAxes, fontsize=11,
             horizontalalignment='right', verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    # Format x-axis dates
    ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(dates)//10)))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Subplot 2: Daily prediction correct/incorrect statistics
    daily_correct = []
    daily_total = []
    daily_dates = []
    
    for i, pred in enumerate(predictions):
        actual_change = pred.get('actual_change')
        if actual_change is not None:
            date = pd.to_datetime(pred['date'])
            prediction = pred.get('prediction')
            is_correct = 1 if prediction == actual_change else 0
            
            daily_dates.append(date)
            daily_correct.append(is_correct)
            daily_total.append(1)
    
    # Plot rolling accuracy (window size = 10)
    if len(daily_correct) >= 10:
        window_size = min(10, len(daily_correct) // 5)
        rolling_accuracy = []
        rolling_dates = []
        
        for i in range(window_size - 1, len(daily_correct)):
            window_correct = sum(daily_correct[i - window_size + 1:i + 1])
            window_total = sum(daily_total[i - window_size + 1:i + 1])
            rolling_accuracy.append(window_correct / window_total if window_total > 0 else 0)
            rolling_dates.append(daily_dates[i])
        
        ax2.plot(rolling_dates, rolling_accuracy, linewidth=2, color='#F18F01', 
                label=f'Rolling Accuracy (Window={window_size})', marker='s', markersize=3)
        ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
        ax2.set_xlabel('Date', fontsize=12)
        ax2.set_ylabel('Rolling Accuracy', fontsize=12)
        ax2.set_title(f'Rolling Accuracy (Window={window_size} days)', fontsize=14, fontweight='bold')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # Format x-axis dates
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        ax2.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(rolling_dates)//10)))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    
    if output_file:
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Accuracy convergence chart saved to: {output_file}")
    
    return fig


def visualize_all(json_file, output_dir=None):
    """
    Generate all visualization charts
    
    Args:
        json_file: Result JSON file path
        output_dir: Output directory, if None then use the directory of JSON file
    """
    print("="*60)
    print("Profit Calculation Result Visualization")
    print("="*60)
    
    # Load data
    print(f"\nLoading result file: {json_file}")
    data = load_result_file(json_file)
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(json_file))
    
    # Generate output file names
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    
    # Plot three charts
    print("\n[1/3] Plotting price changes chart...")
    fig1 = plot_price_changes(data, os.path.join(output_dir, f"{base_name}_price_changes.png"))
    if fig1:
        plt.close(fig1)
    
    print("\n[2/3] Plotting asset comparison chart...")
    fig2 = plot_asset_comparison(data, os.path.join(output_dir, f"{base_name}_asset_comparison.png"))
    if fig2:
        plt.close(fig2)
    
    print("\n[3/3] Plotting accuracy convergence chart...")
    fig3 = plot_accuracy_convergence(data, os.path.join(output_dir, f"{base_name}_accuracy_convergence.png"))
    if fig3:
        plt.close(fig3)
    
    # Generate combined chart (all three plots in one file)
    print("\nGenerating combined chart...")
    fig = plt.figure(figsize=(16, 12))
    
    # Plot 1: Price changes
    if data.get('daily_data'):
        ax1 = plt.subplot(3, 1, 1)
        daily_data = data['daily_data']
        dates = [pd.to_datetime(item['date']) for item in daily_data if item.get('close_price') is not None]
        prices = [item['close_price'] for item in daily_data if item.get('close_price') is not None]
        if dates and prices:
            ax1.plot(dates, prices, linewidth=2, color='#2E86AB', label='Closing Price')
            ax1.set_ylabel('Closing Price (CNY)', fontsize=10)
            ax1.set_title('Gold Price Trend', fontsize=12, fontweight='bold')
            ax1.grid(True, alpha=0.3)
            ax1.legend(fontsize=9)
            ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 2: Asset comparison
    if data.get('trade_records') and data.get('daily_data'):
        ax2 = plt.subplot(3, 1, 2)
        trade_records = data['trade_records']
        trade_dates = [pd.to_datetime(r['date']) for r in trade_records]
        trade_values = [r['total_value'] for r in trade_records]
        ax2.plot(trade_dates, trade_values, linewidth=2, color='#2E86AB', 
                label='Trading Strategy Total Assets', marker='o', markersize=3)
        
        # Buy-and-hold strategy
        daily_data = data['daily_data']
        initial_capital = data.get('initial_capital', 10000)
        if len(daily_data) >= 2:
            first_close = daily_data[0].get('close_price')
            if first_close:
                hold_dates = []
                hold_values = []
                for item in daily_data:
                    if item.get('close_price') is not None:
                        hold_dates.append(pd.to_datetime(item['date']))
                        shares = initial_capital / first_close
                        hold_values.append(shares * item['close_price'])
                if hold_dates:
                    ax2.plot(hold_dates, hold_values, linewidth=2, color='#A23B72', 
                            label='Buy-and-Hold Strategy Total Assets', linestyle='--', marker='s', markersize=3)
        
        ax2.axhline(y=initial_capital, color='gray', linestyle=':', linewidth=1, alpha=0.7)
        ax2.set_ylabel('Total Assets (CNY)', fontsize=10)
        ax2.set_title('Trading Strategy vs Buy-and-Hold Strategy Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=9)
        ax2.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Plot 3: Accuracy convergence
    if data.get('predictions'):
        ax3 = plt.subplot(3, 1, 3)
        predictions = data['predictions']
        dates = []
        accuracy_rates = []
        correct_count = 0
        total_count = 0
        
        for pred in predictions:
            actual_change = pred.get('actual_change')
            if actual_change is not None:
                total_count += 1
                if pred.get('prediction') == actual_change:
                    correct_count += 1
                dates.append(pd.to_datetime(pred['date']))
                accuracy_rates.append(correct_count / total_count if total_count > 0 else 0)
        
        if dates:
            ax3.plot(dates, accuracy_rates, linewidth=2, color='#06A77D', marker='o', markersize=3)
            ax3.axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.7)
            ax3.set_xlabel('Date', fontsize=10)
            ax3.set_ylabel('Cumulative Accuracy', fontsize=10)
            ax3.set_title('Prediction Accuracy Convergence Over Time', fontsize=12, fontweight='bold')
            ax3.set_ylim([0, 1])
            ax3.grid(True, alpha=0.3)
            ax3.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
            plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    plt.tight_layout()
    combined_file = os.path.join(output_dir, f"{base_name}_combined.png")
    plt.savefig(combined_file, dpi=300, bbox_inches='tight')
    print(f"Combined chart saved to: {combined_file}")
    plt.close()
    
    print("\n" + "="*60)
    print("Visualization completed!")
    print("="*60)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Profit calculation result visualization')
    parser.add_argument('json_file', type=str,
                       help='Result JSON file path, e.g.: profit_result_518880_20251229_201528.json')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory, if not specified then use the directory of JSON file')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    if not os.path.isabs(args.json_file):
        json_file = os.path.abspath(args.json_file)
    else:
        json_file = args.json_file
    
    if not os.path.exists(json_file):
        print(f"Error: Result file does not exist: {json_file}")
        sys.exit(1)
    
    visualize_all(json_file, args.output_dir)

