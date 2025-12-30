#!/bin/bash

# 串行训练股票模型脚本
# 用法: ./train_stocks.sh [stock_list_file]
# 默认使用 stock_list.txt

# 获取脚本所在目录
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# 配置文件路径
CONFIG_FILE="config/config.yaml"
STOCK_LIST_FILE="${1:-stock_list5.txt}"

# 日志文件
LOG_FILE="training.log"
SUCCESS_FILE="training_success.txt"
FAILED_FILE="training_failed.txt"

# 初始化日志
echo "=========================================" > "$LOG_FILE"
echo "训练开始时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
echo "股票列表文件: $STOCK_LIST_FILE" >> "$LOG_FILE"
echo "=========================================" >> "$LOG_FILE"
echo "" >> "$LOG_FILE"

# 清空成功和失败记录
> "$SUCCESS_FILE"
> "$FAILED_FILE"

# 函数：更新配置文件中的股票代码
update_stock_code() {
    local stock_code=$1
    local config_file=$2
    
    # 使用Python更新YAML文件
    python3 << EOF
import yaml
import sys

try:
    with open('$config_file', 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    
    # 更新股票代码
    config['stock']['stock_code'] = '$stock_code'
    
    # 确保模式是train
    config['mode'] = 'train'
    
    # 写回文件
    with open('$config_file', 'w', encoding='utf-8') as f:
        yaml.dump(config, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    
    print(f"✓ 配置文件已更新: stock_code = $stock_code")
except Exception as e:
    print(f"✗ 更新配置文件失败: {e}", file=sys.stderr)
    sys.exit(1)
EOF
}

# 函数：训练单个股票
train_stock() {
    local stock_code=$1
    
    echo "" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
    echo "开始训练股票: $stock_code" >> "$LOG_FILE"
    echo "时间: $(date '+%Y-%m-%d %H:%M:%S')" >> "$LOG_FILE"
    echo "----------------------------------------" >> "$LOG_FILE"
    
    # 更新配置文件
    if ! update_stock_code "$stock_code" "$CONFIG_FILE" >> "$LOG_FILE" 2>&1; then
        echo "✗ 更新配置文件失败: $stock_code" | tee -a "$LOG_FILE"
        return 1
    fi
    
    # 运行训练
    echo "运行训练: python main.py" >> "$LOG_FILE"
    local start_time=$(date +%s)
    
    # 运行训练，捕获错误
    if python -u main.py >> "$LOG_FILE" 2>&1; then
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "✓ 训练成功: $stock_code (耗时: ${duration}秒)" | tee -a "$LOG_FILE"
        echo "$stock_code" >> "$SUCCESS_FILE"
        return 0
    else
        local end_time=$(date +%s)
        local duration=$((end_time - start_time))
        echo "✗ 训练失败: $stock_code (耗时: ${duration}秒)" | tee -a "$LOG_FILE"
        echo "$stock_code" >> "$FAILED_FILE"
        return 1
    fi
}

# 检查股票列表文件
if [ ! -f "$STOCK_LIST_FILE" ]; then
    echo "错误: 股票列表文件不存在: $STOCK_LIST_FILE"
    exit 1
fi

# 统计信息
TOTAL_STOCKS=$(grep -v '^#' "$STOCK_LIST_FILE" | grep -v '^$' | wc -l | tr -d ' ')
CURRENT=0
SUCCESS_COUNT=0
FAILED_COUNT=0

echo "=========================================" | tee -a "$LOG_FILE"
echo "开始串行训练" | tee -a "$LOG_FILE"
echo "总股票数: $TOTAL_STOCKS" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"

# 循环训练每个股票（串行执行）
while IFS= read -r stock_code || [ -n "$stock_code" ]; do
    # 跳过空行和注释行
    stock_code=$(echo "$stock_code" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')
    if [ -z "$stock_code" ] || [[ "$stock_code" =~ ^# ]]; then
        continue
    fi
    
    # 验证股票代码格式（6位数字）或期货代码格式（2-3位字母+数字，如AU0, AG0）
    if ! [[ "$stock_code" =~ ^[0-9]{6}$ ]] && ! [[ "$stock_code" =~ ^[A-Z]{2,3}[0-9]$ ]]; then
        echo "⚠ 跳过无效代码格式: $stock_code (应为6位数字股票代码或期货代码如AU0)" | tee -a "$LOG_FILE"
        continue
    fi
    
    CURRENT=$((CURRENT + 1))
    
    echo "[$CURRENT/$TOTAL_STOCKS] 处理股票: $stock_code" | tee -a "$LOG_FILE"
    echo "[$CURRENT/$TOTAL_STOCKS] 处理股票: $stock_code"
    
    # 训练股票（即使失败也继续下一个）
    if train_stock "$stock_code"; then
        SUCCESS_COUNT=$((SUCCESS_COUNT + 1))
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
    
    # 显示进度
    echo "进度: $CURRENT/$TOTAL_STOCKS | 成功: $SUCCESS_COUNT | 失败: $FAILED_COUNT" | tee -a "$LOG_FILE"
    echo "进度: $CURRENT/$TOTAL_STOCKS | 成功: $SUCCESS_COUNT | 失败: $FAILED_COUNT"
    
done < "$STOCK_LIST_FILE"

# 输出最终统计
echo "" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "训练完成!" | tee -a "$LOG_FILE"
echo "完成时间: $(date '+%Y-%m-%d %H:%M:%S')" | tee -a "$LOG_FILE"
echo "总股票数: $TOTAL_STOCKS" | tee -a "$LOG_FILE"
echo "成功: $SUCCESS_COUNT" | tee -a "$LOG_FILE"
echo "失败: $FAILED_COUNT" | tee -a "$LOG_FILE"
echo "=========================================" | tee -a "$LOG_FILE"
echo "" | tee -a "$LOG_FILE"
echo "详细日志: $LOG_FILE" | tee -a "$LOG_FILE"
echo "成功列表: $SUCCESS_FILE" | tee -a "$LOG_FILE"
echo "失败列表: $FAILED_FILE" | tee -a "$LOG_FILE"

