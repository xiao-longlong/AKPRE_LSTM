# 股票交易利润计算工具

基于LSTM模型预测结果，模拟交易策略并计算利润。

## 功能说明

- 根据模型预测结果进行买入/卖出决策
- 如果预测第二天涨（prediction=1），买入或持有
- 如果预测第二天跌（prediction=0），卖出
- 全仓操作（买入时使用全部现金，卖出时卖出全部持仓）
- 支持自定义交易日数量、起始日期和初始资金

## 使用方法

```bash
python calculate_profit.py <训练日志目录> [选项]
```

**注意：运行一次即可完成利润计算和可视化图表生成！**

### 参数说明

- `log_folder`: 训练日志目录路径（必需）
  - 例如: `logs/20251225_175236_train_518880_20251225`
  - 或绝对路径: `/home/wxl/wxlcode/stock/AKPRE_LSTM/logs/20251225_175236_train_518880_20251225`

### 选项

- `--days`: 交易日数量，默认30天
- `--start-date`: 起始日期（格式: YYYY-MM-DD），如果不指定则从最后一天往前推
- `--capital`: 初始资金，默认10000元
- `--no-visualize`: 不自动生成可视化图表（默认会自动生成）

## 使用示例

```bash
# 使用默认参数（30个交易日，从最后一天往前推，初始资金10000元）
# 会自动生成所有结果文件和可视化图表
python calculate_profit.py /home/wxl/wxlcode/stock/AKPRE_LSTM/logs/20251225_175236_train_518880_20251225

# 指定交易日数量为60天
python calculate_profit.py /home/wxl/wxlcode/stock/AKPRE_LSTM/logs/20251225_175236_train_518880_20251225 --days 60

# 指定起始日期和初始资金
python calculate_profit.py /home/wxl/wxlcode/stock/AKPRE_LSTM/logs/20251225_175236_train_518880_20251225 --start-date 2024-11-01 --capital 20000

# 使用相对路径
python calculate_profit.py ../../logs/20251225_175236_train_518880_20251225 --days 30

# 不自动生成可视化图表（只计算利润）
python calculate_profit.py /home/wxl/wxlcode/stock/AKPRE_LSTM/logs/20251225_175236_train_518880_20251225 --no-visualize
```

## 输出结果

脚本会自动生成以下文件（保存在 `<股票码>_<投入金额>_<日期年月起止>` 文件夹下）：

1. **JSON结果文件** (`profit_result_<股票代码>_<时间戳>.json`)
   - 包含完整的计算结果、预测记录和交易记录
   - 包含每日的收盘价和实际涨跌情况（用于计算准确率）

2. **CSV交易记录文件** (`trade_records_<股票代码>_<时间戳>.csv`)
   - 包含每日的交易操作记录，方便分析

3. **可视化图表**（自动生成，除非使用 `--no-visualize` 参数）
   - **金价涨跌曲线** (`*_price_changes.png`)
     - 收盘价走势曲线
     - 每日涨跌百分比柱状图
   - **资产对比曲线** (`*_asset_comparison.png`)
     - 交易策略总资产 vs 一直持有策略总资产
     - 显示两种策略的收益对比
   - **准确率收敛曲线** (`*_accuracy_convergence.png`)
     - 预测准确率随日期增加的累积准确率曲线
     - 滚动准确率曲线（窗口大小=10天）
   - **综合图表** (`*_combined.png`)
     - 包含以上三个图表的综合视图

## 独立运行可视化（可选）

如果需要单独运行可视化脚本：

```bash
python visualize_profit.py <JSON结果文件> [选项]
```

### 可视化选项

- `json_file`: 结果JSON文件路径（必需）
- `--output-dir`: 输出目录，如果不指定则使用JSON文件所在目录

## 结果说明

- **初始资金**: 开始时的资金
- **最终价值**: 结束时的总资产（现金+持仓价值）
- **利润**: 最终价值 - 初始资金
- **收益率**: (利润 / 初始资金) × 100%
- **交易次数**: 买入和卖出的总次数

## 注意事项

1. 交易日数量指的是实际的交易日，不包括周末和节假日
2. 如果指定了起始日期，会从该日期开始到数据的最新日期
3. 如果没有指定起始日期，会从最后一天往前推指定数量的交易日
4. 最后一天如果还有持仓，会以收盘价自动卖出

