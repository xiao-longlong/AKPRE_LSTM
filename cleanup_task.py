"""
批量删除训练/推理任务的全部相关数据

使用方法:
    1. 列出所有日志文件夹:
       python cleanup_task.py --list
    
    2. 删除单个任务（预览模式）:
       python cleanup_task.py logs/20251225_173526_train_518880_20251225 --dry-run
    
    3. 删除单个任务（实际删除）:
       python cleanup_task.py logs/20251225_173526_train_518880_20251225
    
    4. 删除多个任务:
       python cleanup_task.py logs/folder1 logs/folder2 --no-confirm
    
    5. 删除所有任务（危险操作）:
       python cleanup_task.py --all

删除的内容包括:
    - 日志文件夹（包含模型、训练曲线、日志等）
    - 原始数据文件（data/raw/<股票码_日期>.csv）
    - 处理后数据文件夹（data/processed/<股票码_日期>/）
"""
import os
import sys
import json
import shutil
import argparse
from pathlib import Path
from typing import List, Tuple, Optional


def parse_log_folder_name(log_folder_path: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """
    从日志文件夹名称解析股票代码、日期和任务类型
    
    Args:
        log_folder_path: 日志文件夹路径，如 "logs/20251225_173526_train_518880_20251225"
    
    Returns:
        (stock_code, end_date, task_type) 或 (None, None, None) 如果无法解析
    """
    folder_name = os.path.basename(log_folder_path.rstrip('/'))
    
    # 格式: <日期时间>_<任务类型>_<股票码>_<截止日期>
    # 例如: 20251225_173526_train_518880_20251225
    # 或: 20251225_173526_inference_518880_20251225
    
    parts = folder_name.split('_')
    if len(parts) >= 5:
        # 提取任务类型（train或inference）
        task_type = parts[2] if parts[2] in ['train', 'inference'] else None
        # 提取股票代码（倒数第二部分）
        stock_code = parts[-2] if len(parts) >= 5 else None
        # 提取截止日期（最后一部分）
        end_date = parts[-1] if len(parts) >= 5 else None
        
        return stock_code, end_date, task_type
    
    return None, None, None


def get_related_files(log_folder_path: str, base_dir: str = None) -> dict:
    """
    获取与日志文件夹相关的所有文件路径
    
    Args:
        log_folder_path: 日志文件夹路径
        base_dir: 项目根目录，如果为None则自动检测
    
    Returns:
        dict: 包含所有相关文件路径的字典
    """
    if base_dir is None:
        # 自动检测项目根目录
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = script_dir
    
    # 解析日志文件夹名称
    stock_code, end_date, task_type = parse_log_folder_name(log_folder_path)
    
    # 如果无法从名称解析，尝试从train_info.json读取
    if stock_code is None or end_date is None:
        train_info_path = os.path.join(log_folder_path, 'train_info.json')
        if os.path.exists(train_info_path):
            try:
                with open(train_info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                    stock_code = info.get('stock_code')
                    end_date = info.get('end_date')
            except Exception as e:
                print(f"警告: 无法读取train_info.json: {e}")
    
    # 如果还是无法获取，尝试从inference_info.json读取
    if stock_code is None or end_date is None:
        inference_info_path = os.path.join(log_folder_path, 'inference_info.json')
        if os.path.exists(inference_info_path):
            try:
                with open(inference_info_path, 'r', encoding='utf-8') as f:
                    info = json.load(f)
                    stock_code = info.get('stock_code')
                    end_date = info.get('end_date')
            except Exception as e:
                print(f"警告: 无法读取inference_info.json: {e}")
    
    if stock_code is None or end_date is None:
        print(f"错误: 无法从日志文件夹 '{log_folder_path}' 解析股票代码和日期")
        return {}
    
    related_files = {
        'log_folder': log_folder_path,
        'raw_data': os.path.join(base_dir, 'data', 'raw', f"{stock_code}_{end_date}.csv"),
        'processed_data_folder': os.path.join(base_dir, 'data', 'processed', f"{stock_code}_{end_date}"),
        'stock_code': stock_code,
        'end_date': end_date,
        'task_type': task_type
    }
    
    return related_files


def find_all_log_folders(logs_dir: str = None) -> List[str]:
    """
    查找所有日志文件夹
    
    Args:
        logs_dir: 日志目录路径
    
    Returns:
        日志文件夹路径列表
    """
    if logs_dir is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        logs_dir = os.path.join(script_dir, 'logs')
    
    if not os.path.exists(logs_dir):
        return []
    
    log_folders = []
    for item in os.listdir(logs_dir):
        item_path = os.path.join(logs_dir, item)
        if os.path.isdir(item_path):
            log_folders.append(item_path)
    
    return sorted(log_folders)


def delete_files(files_dict: dict, dry_run: bool = False) -> dict:
    """
    删除相关文件
    
    Args:
        files_dict: 相关文件字典
        dry_run: 如果为True，只显示将要删除的文件，不实际删除
    
    Returns:
        删除结果字典
    """
    results = {
        'deleted': [],
        'not_found': [],
        'errors': []
    }
    
    # 删除日志文件夹
    log_folder = files_dict.get('log_folder')
    if log_folder and os.path.exists(log_folder):
        try:
            if not dry_run:
                shutil.rmtree(log_folder)
            results['deleted'].append(('日志文件夹', log_folder))
        except Exception as e:
            results['errors'].append((log_folder, str(e)))
    elif log_folder:
        results['not_found'].append(('日志文件夹', log_folder))
    
    # 删除原始数据文件
    raw_data = files_dict.get('raw_data')
    if raw_data and os.path.exists(raw_data):
        try:
            if not dry_run:
                os.remove(raw_data)
            results['deleted'].append(('原始数据', raw_data))
        except Exception as e:
            results['errors'].append((raw_data, str(e)))
    elif raw_data:
        results['not_found'].append(('原始数据', raw_data))
    
    # 删除处理后数据文件夹
    processed_folder = files_dict.get('processed_data_folder')
    if processed_folder and os.path.exists(processed_folder):
        try:
            if not dry_run:
                shutil.rmtree(processed_folder)
            results['deleted'].append(('处理后数据', processed_folder))
        except Exception as e:
            results['errors'].append((processed_folder, str(e)))
    elif processed_folder:
        results['not_found'].append(('处理后数据', processed_folder))
    
    return results


def print_summary(files_dict: dict, results: dict, dry_run: bool = False):
    """打印删除摘要"""
    print("\n" + "=" * 60)
    if dry_run:
        print("【预览模式】将要删除的文件:")
    else:
        print("删除结果摘要:")
    print("=" * 60)
    
    print(f"\n任务信息:")
    print(f"  股票代码: {files_dict.get('stock_code', 'N/A')}")
    print(f"  截止日期: {files_dict.get('end_date', 'N/A')}")
    print(f"  任务类型: {files_dict.get('task_type', 'N/A')}")
    
    if results['deleted']:
        print(f"\n✓ 已删除 ({len(results['deleted'])} 项):")
        for item_type, path in results['deleted']:
            print(f"  - {item_type}: {path}")
    
    if results['not_found']:
        print(f"\n⚠ 未找到 ({len(results['not_found'])} 项):")
        for item_type, path in results['not_found']:
            print(f"  - {item_type}: {path}")
    
    if results['errors']:
        print(f"\n✗ 删除失败 ({len(results['errors'])} 项):")
        for path, error in results['errors']:
            print(f"  - {path}: {error}")


def cleanup_task(log_folder_path: str, base_dir: str = None, 
                 dry_run: bool = False, confirm: bool = True) -> bool:
    """
    清理单个任务的所有相关数据
    
    Args:
        log_folder_path: 日志文件夹路径
        base_dir: 项目根目录
        dry_run: 如果为True，只预览不删除
        confirm: 如果为True，删除前需要确认
    
    Returns:
        是否成功删除
    """
    # 获取相关文件
    files_dict = get_related_files(log_folder_path, base_dir)
    
    if not files_dict:
        print(f"错误: 无法解析日志文件夹 '{log_folder_path}'")
        return False
    
    # 显示将要删除的文件
    print("\n" + "=" * 60)
    print("清理任务数据")
    print("=" * 60)
    print(f"\n日志文件夹: {log_folder_path}")
    print(f"\n相关文件:")
    print(f"  1. 日志文件夹: {files_dict['log_folder']}")
    print(f"  2. 原始数据: {files_dict['raw_data']}")
    print(f"  3. 处理后数据: {files_dict['processed_data_folder']}")
    
    # 检查文件是否存在
    existing_files = []
    if os.path.exists(files_dict['log_folder']):
        existing_files.append(('日志文件夹', files_dict['log_folder']))
    if os.path.exists(files_dict['raw_data']):
        existing_files.append(('原始数据', files_dict['raw_data']))
    if os.path.exists(files_dict['processed_data_folder']):
        existing_files.append(('处理后数据', files_dict['processed_data_folder']))
    
    if not existing_files:
        print("\n⚠ 没有找到任何相关文件，可能已经被删除。")
        return True
    
    print(f"\n找到 {len(existing_files)} 个相关文件/文件夹")
    
    # 确认删除
    if confirm and not dry_run:
        response = input("\n确认删除以上所有文件? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("取消删除")
            return False
    
    # 执行删除
    results = delete_files(files_dict, dry_run=dry_run)
    print_summary(files_dict, results, dry_run=dry_run)
    
    return len(results['errors']) == 0


def cleanup_multiple_tasks(log_folder_paths: List[str], base_dir: str = None,
                           dry_run: bool = False, confirm: bool = True, 
                           already_confirmed: bool = False):
    """
    批量清理多个任务
    
    Args:
        log_folder_paths: 日志文件夹路径列表
        base_dir: 项目根目录
        dry_run: 预览模式
        confirm: 是否需要确认（如果already_confirmed=True则忽略此参数）
        already_confirmed: 是否已经确认过（用于--all模式，只需确认一次）
    """
    print("\n" + "=" * 60)
    print(f"批量清理 {len(log_folder_paths)} 个任务")
    print("=" * 60)
    
    # 如果已经确认过（比如使用--all），就不需要每个任务都确认
    # 否则，如果是批量删除多个任务，先统一确认一次
    if confirm and not dry_run and not already_confirmed:
        print(f"\n将要删除 {len(log_folder_paths)} 个任务的所有相关数据")
        response = input("确认删除以上所有任务? (yes/no): ").strip().lower()
        if response not in ['yes', 'y']:
            print("取消删除")
            return
        already_confirmed = True
    
    success_count = 0
    fail_count = 0
    
    for i, log_folder_path in enumerate(log_folder_paths, 1):
        print(f"\n[{i}/{len(log_folder_paths)}] 处理: {log_folder_path}")
        try:
            # 如果已经确认过，就不需要每个任务都确认
            if cleanup_task(log_folder_path, base_dir, dry_run=dry_run, 
                          confirm=False if already_confirmed else confirm):
                success_count += 1
            else:
                fail_count += 1
        except Exception as e:
            print(f"✗ 处理失败: {e}")
            fail_count += 1
    
    print("\n" + "=" * 60)
    print("批量清理完成")
    print("=" * 60)
    print(f"成功: {success_count}, 失败: {fail_count}")


def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='批量删除训练/推理任务的全部相关数据')
    parser.add_argument('log_folders', nargs='*', 
                       help='日志文件夹路径（可以是多个），如果不提供则列出所有日志文件夹')
    parser.add_argument('--base-dir', type=str, default=None,
                       help='项目根目录（默认自动检测）')
    parser.add_argument('--dry-run', action='store_true',
                       help='预览模式，只显示将要删除的文件，不实际删除')
    parser.add_argument('--no-confirm', action='store_true',
                       help='不询问确认，直接删除')
    parser.add_argument('--list', action='store_true',
                       help='列出所有可用的日志文件夹')
    parser.add_argument('--all', action='store_true',
                       help='删除所有日志文件夹（危险操作）')
    
    args = parser.parse_args()
    
    base_dir = args.base_dir
    if base_dir is None:
        base_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 列出所有日志文件夹
    if args.list or (not args.log_folders and not args.all):
        log_folders = find_all_log_folders(os.path.join(base_dir, 'logs'))
        if log_folders:
            print("\n可用的日志文件夹:")
            print("=" * 60)
            for i, folder in enumerate(log_folders, 1):
                folder_name = os.path.basename(folder)
                stock_code, end_date, task_type = parse_log_folder_name(folder)
                print(f"{i}. {folder_name}")
                if stock_code and end_date:
                    print(f"   股票: {stock_code}, 日期: {end_date}, 类型: {task_type}")
            print("=" * 60)
            print(f"\n使用方法: python cleanup_task.py <日志文件夹路径>")
            print(f"示例: python cleanup_task.py logs/20251225_173526_train_518880_20251225")
        else:
            print("没有找到任何日志文件夹")
        return
    
    # 获取要删除的日志文件夹列表
    already_confirmed = False
    if args.all:
        log_folders = find_all_log_folders(os.path.join(base_dir, 'logs'))
        if not log_folders:
            print("没有找到任何日志文件夹")
            return
        if not args.no_confirm:
            response = input(f"\n确认删除所有 {len(log_folders)} 个日志文件夹及其相关数据? (yes/no): ").strip().lower()
            if response not in ['yes', 'y']:
                print("取消删除")
                return
            already_confirmed = True  # 标记已经确认过
    else:
        log_folders = args.log_folders
        # 转换为绝对路径
        log_folders = [os.path.abspath(f) if not os.path.isabs(f) else f for f in log_folders]
    
    # 执行清理
    if len(log_folders) == 1:
        cleanup_task(log_folders[0], base_dir, dry_run=args.dry_run, confirm=not args.no_confirm)
    else:
        cleanup_multiple_tasks(log_folders, base_dir, dry_run=args.dry_run, 
                              confirm=not args.no_confirm, already_confirmed=already_confirmed)


if __name__ == "__main__":
    main()

