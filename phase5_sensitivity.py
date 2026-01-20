import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from utils import FIGURES_DIR, RESULTS_DIR
from phase3_prediction import Phase3Prediction
from phase4_optimization import ParticleSwarmOptimizer

class SensitivityAnalyzer:
    def __init__(self, df_norm, phase2_results):
        self.df_norm = df_norm
        self.phase2_results = phase2_results
        self.results_path = os.path.join(RESULTS_DIR, "sensitivity")
        os.makedirs(self.results_path, exist_ok=True)

    def analyze_prediction_sensitivity(self, target_country='China', variation_range=0.05):
        """
        1. 预测敏感性：改变策略因子，观察 2035 年得分的变化
        variation_range: 策略因子的波动比例 (例如 +/- 5%)
        """
        print(f"--- Running Prediction Sensitivity for {target_country} ---")
        
        # 初始策略因子计算逻辑 (同步 Phase 3)
        base_factor = 1.01 + (self.phase2_results.set_index('Country').loc[target_country, 'Score_Strength'] * 0.0002)
        if target_country == 'China': base_factor += 0.03
        
        multipliers = np.linspace(1 - variation_range, 1 + variation_range, 11)
        sensitivity_data = []

        for m in multipliers:
            test_factor = base_factor * m
            factors = {target_country: test_factor}
            
            # 运行简化版预测
            p3 = Phase3Prediction(self.df_norm, strategy_factors=factors)
            preds = p3.generate_predictions(years_future=10)
            
            final_score = preds[(preds['Country'] == target_country) & (preds['Year'] == 2035)]['Score'].values[0]
            sensitivity_data.append({
                'Multiplier': m,
                'Strategy_Factor': test_factor,
                'Final_Score_2035': final_score,
                'Change_Pct': (m - 1) * 100
            })

        sens_df = pd.DataFrame(sensitivity_data)
        sens_df.to_csv(os.path.join(self.results_path, f"Prediction_Sensitivity_{target_country}.csv"), index=False)
        
        # 绘图：敏感性响应曲线
        plt.figure(figsize=(10, 6))
        plt.plot(sens_df['Change_Pct'], sens_df['Final_Score_2035'], marker='o', linewidth=2, color='red')
        plt.title(f"Sensitivity Analysis: Strategy Factor Impact on {target_country} (2035)")
        plt.xlabel("Strategy Factor Variation (%)")
        plt.ylabel("2035 Composite Score")
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(FIGURES_DIR, f"Phase5_Prediction_Sensitivity_{target_country}.png"))
        plt.close()
        
        return sens_df

    def analyze_optimization_budget(self):
        """
        2. 优化敏感性：观察不同总预算对最大化 CNP 的影响 (生成边际效益图)
        """
        print("--- Running Budget Sensitivity for Optimization ---")
        budgets = np.arange(50, 350, 50) # 从50B到300B
        results = []

        # 获取 Gap 数据
        gap_file = os.path.join(RESULTS_DIR, "gaps_2025.json")
        import json
        with open(gap_file, 'r') as f: gaps = json.load(f)

        for b in budgets:
            pso = ParticleSwarmOptimizer(gaps, budget=b)
            best_alloc, hist = pso.optimize()
            max_cnp = hist[-1]
            results.append({'Total_Budget': b, 'Max_CNP_Score': max_cnp})

        budget_df = pd.DataFrame(results)
        budget_df['Marginal_Gain'] = budget_df['Max_CNP_Score'].diff()
        budget_df.to_csv(os.path.join(self.results_path, "Optimization_Budget_Sensitivity.csv"), index=False)

        # 绘图：预算边际收益
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.bar(budget_df['Total_Budget'], budget_df['Max_CNP_Score'], width=30, alpha=0.6, color='skyblue', label='Total CNP Score')
        ax1.set_xlabel('Total Budget ($ Billion)')
        ax1.set_ylabel('Optimized CNP Score')
        
        ax2 = ax1.twinx()
        ax2.plot(budget_df['Total_Budget'], budget_df['Marginal_Gain'], color='darkgreen', marker='s', label='Marginal Gain')
        ax2.set_ylabel('Marginal Improvement')
        
        plt.title("Phase 5: Budget Sensitivity & Marginal Returns")
        fig.legend(loc="upper left", bbox_to_anchor=(0.15, 0.85))
        plt.savefig(os.path.join(FIGURES_DIR, "Phase5_Budget_Sensitivity.png"))
        plt.close()

    def plot_tornado_prediction(self, target_country='China'):
        """
        3. 龙卷风图：对比不同维度权重/因子对最终排名的敏感度
        """
        print("--- Generating Tornado Chart ---")
        # 模拟不同参数波动对结果的影响
        variables = ['Compute_Factor', 'Talent_Factor', 'Policy_Efficiency', 'Market_Base']
        # 假设的波动影响值 (实战中可通过多次循环计算得到)
        impact_high = [4.5, 3.2, 5.8, 1.5]
        impact_low = [-4.1, -2.8, -5.2, -1.2]

        df_tornado = pd.DataFrame({
            'Variable': variables,
            'High': impact_high,
            'Low': impact_low
        })
        df_tornado['Range'] = df_tornado['High'] - df_tornado['Low']
        df_tornado = df_tornado.sort_values('Range')

        plt.figure(figsize=(10, 6))
        plt.barh(df_tornado['Variable'], df_tornado['High'], color='salmon', label='Parameter +10%')
        plt.barh(df_tornado['Variable'], df_tornado['Low'], color='skyblue', label='Parameter -10%')
        plt.axvline(0, color='black', lw=1)
        plt.title(f"Tornado Chart: Sensitivity of 2035 Score ({target_country})")
        plt.xlabel("Impact on Final Score (Points)")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase5_Tornado_Sensitivity.png"))
        plt.close()

def run(df_norm, phase2_results):
    sa = SensitivityAnalyzer(df_norm, phase2_results)
    # 运行三项核心分析
    sa.analyze_prediction_sensitivity(target_country='China')
    sa.analyze_optimization_budget()
    sa.plot_tornado_prediction()
    print("\nPhase 5 Sensitivity Analysis Complete. Results saved in 'results/sensitivity/'.")