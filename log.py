import re
import json
from typing import List, Dict, Tuple
import matplotlib.pyplot as plt
import os

class MultiLogScanner:
    def __init__(self):
        self.correct_pattern = r'INFO:__main__:Correct:\s*([\d.]+)\s*\(trial\s*(\d+)\)' 
        self.costs_pattern = r'INFO:__main__:Costs:.*?\'total\':\s*([\d.]+).*?\(trial\s*(\d+)\)'
    
    def scan_log_file(self, file_path: str) -> Tuple[str, List[Dict]]:
        """扫描单个日志文件，返回文件名和结果"""
        results = []
        file_name = os.path.basename(file_path).replace('.log', '')
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                results = self.parse_log_content(content)
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            
        return file_name, results

    def scan_multiple_logs(self, file_paths: List[str]) -> Dict[str, List[Dict]]:
        """扫描多个日志文件"""
        all_results = {}
        
        for file_path in file_paths:
            file_name, results = self.scan_log_file(file_path)
            if results:  # 只添加有数据的文件
                all_results[file_name] = results
                print(f"Loaded {len(results)} trials from {file_name}")
            else:
                print(f"No data found in {file_name}")
                
        return all_results

    def parse_log_content(self, content: str) -> List[Dict]:
        """解析日志内容"""
        results = []
        correct_matches = re.findall(self.correct_pattern, content)
        correct_data = {int(trial): float(correct) for correct, trial in correct_matches}
        costs_matches = re.findall(self.costs_pattern, content)
        costs_data = {int(trial): float(total) for total, trial in costs_matches}
        all_trials = set(correct_data.keys()) | set(costs_data.keys())
        
        for trial in sorted(all_trials):
            result = {
                'trial': trial,
                'correct': correct_data.get(trial),
                'costs_total': costs_data.get(trial)
            }
            results.append(result)
        return results
    
    def print_comparison_results(self, all_results: Dict[str, List[Dict]]):
        """打印对比结果"""
        print("\n" + "="*80)
        print("Multi-Log Comparison Results")
        print("="*80)
        
        for file_name, results in all_results.items():
            print(f"\n{file_name}:")
            print(f"{'Trial':<8} {'Correct':<12} {'Costs Total':<12}")
            print("-" * 35)
            
            for result in results:
                trial = result['trial']
                correct = result['correct'] if result['correct'] is not None else 'N/A'
                costs = result['costs_total'] if result['costs_total'] is not None else 'N/A'
                print(f"{trial:<8} {correct:<12} {costs:<12}")

    def plot_comparison(self, all_results: Dict[str, List[Dict]], save_path: str = 'multi_log_comparison.png'):
        """绘制多日志对比图（两个子图）"""
        if not all_results:
            print("No data to plot")
            return
            
        # 设置颜色和标记样式
        colors = ['#2E86AB', '#A23B72', '#F18F01', '#C73E1D', '#6A994E', '#7209B7']
        markers = ['o', 's', '^', 'D', 'v', 'p']
        
        # 创建两个子图
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 12))
        fig.suptitle('Multi-Log Analysis: Performance Comparison', fontsize=16, fontweight='bold')
        
        # 子图1：准确率对比
        ax1.set_title('Accuracy Comparison Across Different Logs', fontsize=14, pad=15)
        ax1.set_xlabel('Trial', fontsize=12)
        ax1.set_ylabel('Accuracy (Correct)', fontsize=12)
        ax1.grid(True, alpha=0.3)
        
        # 子图2：成本对比
        ax2.set_title('Cost Comparison Across Different Logs', fontsize=14, pad=15)
        ax2.set_xlabel('Trial', fontsize=12)
        ax2.set_ylabel('Total Cost', fontsize=12)
        ax2.grid(True, alpha=0.3)
        
        # 为每个日志文件绘制数据
        for idx, (file_name, results) in enumerate(all_results.items()):
            color = colors[idx % len(colors)]
            marker = markers[idx % len(markers)]
            
            # 准备数据
            trials = []
            correct_values = []
            costs_values = []
            
            for result in results:
                trials.append(result['trial'])
                correct_values.append(result['correct'])
                costs_values.append(result['costs_total'])
            
            # 绘制准确率
            correct_clean = [(t, v) for t, v in zip(trials, correct_values) if v is not None]
            if correct_clean:
                trials_correct, values_correct = zip(*correct_clean)
                ax1.plot(trials_correct, values_correct, marker=marker, linestyle='-', 
                        linewidth=2.5, markersize=7, label=file_name, color=color, alpha=0.8)
                
                # 添加数值标签（仅在数据点较少时）
                if len(trials_correct) <= 10:
                    for x, y in zip(trials_correct, values_correct):
                        ax1.annotate(f'{y:.3f}', (x, y), textcoords="offset points", 
                                   xytext=(0, 10), ha='center', fontsize=8, color=color)
            
            # 绘制成本
            costs_clean = [(t, v) for t, v in zip(trials, costs_values) if v is not None]
            if costs_clean:
                trials_costs, values_costs = zip(*costs_clean)
                ax2.plot(trials_costs, values_costs, marker=marker, linestyle='-', 
                        linewidth=2.5, markersize=7, label=file_name, color=color, alpha=0.8)
                
                # 添加数值标签（仅在数据点较少时）
                if len(trials_costs) <= 10:
                    for x, y in zip(trials_costs, values_costs):
                        ax2.annotate(f'{y:.4f}', (x, y), textcoords="offset points", 
                                   xytext=(0, 10), ha='center', fontsize=8, color=color)
        
        # 添加图例
        ax1.legend(loc='best', fontsize=10, framealpha=0.9)
        ax2.legend(loc='best', fontsize=10, framealpha=0.9)
        

        plt.tight_layout()
        

        plt.savefig(save_path, dpi=300, bbox_inches='tight', 
                   facecolor='white', edgecolor='none')
        print(f"Comparison chart saved to: {save_path}")
        




scanner = MultiLogScanner()
log_files = [
    "/ceph/hpc/home/eutianyih/CLAN/swift_reflect/cachesaver/logs/correctness/gpt-4.1-nano/hotpotqa/reflexion_react.log",
    "/ceph/hpc/home/eutianyih/CLAN/swift_reflect/cachesaver/logs/correctness/gpt-4.1-nano/hotpotqa/reflect_prev_k.log",
    "/ceph/hpc/home/eutianyih/CLAN/swift_reflect/cachesaver/logs/correctness/gpt-4.1-nano/hotpotqa/reflect_summary.log"
]
all_results = scanner.scan_multiple_logs(log_files)

if all_results:
    scanner.print_comparison_results(all_results)
    scanner.plot_comparison(all_results, 'multi_log_comparison.png')
    
else:
    print("No valid log files found!")


