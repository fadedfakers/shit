import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.optimize import linprog
import json
import os
from utils import FIGURES_DIR, RESULTS_DIR

class Phase2Evaluation:
    def __init__(self, df, df_norm):
        self.df = df.copy()
        self.df_norm = df_norm.copy()
        
        # 预处理
        self.df.fillna(0, inplace=True)
        self.df_norm.fillna(0, inplace=True)
        self.df.replace([np.inf, -np.inf], 0, inplace=True)
        self.df_norm.replace([np.inf, -np.inf], 0, inplace=True)

    def calculate_hard_strength(self, year=2025):
        # ... (保持原有的 PCA 算法不变) ...
        print(f"--- Calculating Hard Strength (PCA) {year} ---")
        data_year = self.df[self.df['Year'] == year].copy()
        if data_year.empty: return pd.DataFrame(columns=['Country', 'Score_Strength'])
        
        core_features = ['Compute_Power', 'Gov_Investment', 'AI_Patents', 
                         'AI_Researchers', 'LLM_Count', 'Unicorn_Companies', 'AI_Market_Size']
        valid_features = [c for c in core_features if c in data_year.columns]
        X = data_year[valid_features].values
        
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        pca = PCA(n_components=1)
        pc1 = pca.fit_transform(X_scaled).flatten()
        
        corr = np.corrcoef(data_year['Compute_Power'], pc1)[0, 1]
        if corr < 0: pc1 = -pc1
            
        min_s, max_s = pc1.min(), pc1.max()
        scores = (pc1 - min_s) / (max_s - min_s) * 100 if max_s > min_s else np.full_like(pc1, 50)
        
        data_year['Score_Strength'] = scores
        return data_year[['Country', 'Score_Strength']]

    def _solve_bcc_dea(self, x0, y0, X, Y):
        # ... (保持原有的 DEA 求解器不变) ...
        try:
            n, m = X.shape
            s = Y.shape[1]
            c = np.zeros(1 + n); c[0] = 1
            A_ub_input = np.hstack([-x0.reshape(m, 1), X.T])
            b_ub_input = np.zeros(m)
            A_ub_output = np.hstack([np.zeros((s, 1)), -Y.T])
            b_ub_output = -y0
            A_ub = np.vstack([A_ub_input, A_ub_output])
            b_ub = np.concatenate([b_ub_input, b_ub_output])
            A_eq = np.zeros((1, 1 + n)); A_eq[0, 1:] = 1
            b_eq = np.array([1])
            bounds = [(0, None)] * (1 + n)
            res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method='highs')
            return res.x[0] if res.success else 0.0
        except: return 0.0

    def calculate_efficiency(self, year=2025):
        # ... (保持原有的 DEA 流程不变) ...
        print(f"--- Calculating Efficiency DEA (Raw Data) {year} ---")
        data_year = self.df[self.df['Year'] == year].copy()
        if data_year.empty: return pd.DataFrame(columns=['Country', 'Score_Efficiency'])

        input_cols = ['Compute_Power', 'AI_Researchers'] 
        output_cols = ['Ind_Market_Size']
        X = np.maximum(data_year[input_cols].values, 0.1)
        Y = np.maximum(data_year[output_cols].values, 0.1)
        
        scores = [self._solve_bcc_dea(X[i], Y[i], X, Y) for i in range(len(X))]
        data_year['Score_Efficiency'] = np.array(scores) * 100
        data_year['Score_Efficiency'] = data_year['Score_Efficiency'].fillna(0)
        
        return data_year[['Country', 'Score_Efficiency']]

    def visualize_quadrant(self, merged_df):
        # ... (保持原有的象限图不变) ...
        plot_df = merged_df.dropna(subset=['Score_Strength', 'Score_Efficiency'])
        if plot_df.empty: return
        plt.figure(figsize=(10, 8))
        sns.scatterplot(data=plot_df, x='Score_Strength', y='Score_Efficiency', s=300, hue='Country', palette='tab10', alpha=0.9, edgecolor='w', linewidth=2)
        for i in range(plot_df.shape[0]):
            plt.text(plot_df.Score_Strength.iloc[i]+1.5, plot_df.Score_Efficiency.iloc[i], plot_df.Country.iloc[i], fontsize=11, fontweight='bold', va='center')
        plt.axvline(plot_df.Score_Strength.median(), color='gray', linestyle='--', alpha=0.5)
        plt.axhline(plot_df.Score_Efficiency.median(), color='gray', linestyle='--', alpha=0.5)
        plt.title(f"Phase 2: Strength vs. Efficiency Quadrant (2025)", fontsize=14)
        plt.xlabel("Hard Strength (PCA)", fontsize=12); plt.ylabel("Soft Efficiency (DEA)", fontsize=12)
        plt.grid(True, alpha=0.3); plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase2_Strength_Efficiency_Quadrant.png"))
        plt.close()

    def plot_bar_rankings(self, merged_df):
        # ... (保持原有的排名图不变) ...
        plot_df = merged_df.dropna(subset=['Score_Strength', 'Score_Efficiency'])
        plt.figure(figsize=(10, 6))
        sns.barplot(data=plot_df.sort_values('Score_Strength', ascending=False), x='Score_Strength', y='Country', palette='viridis')
        plt.title("Global AI Hard Strength Ranking (PCA 2025)"); plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase2_Hard_Strength_Ranking.png")); plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.barplot(data=plot_df.sort_values('Score_Efficiency', ascending=False), x='Score_Efficiency', y='Country', palette='rocket')
        plt.title("Global AI Efficiency Ranking (DEA 2025)"); plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase2_Efficiency_Ranking.png")); plt.close()

    # =========================================================================
    # [核心修改] 全新的雷达图逻辑：3子图并列，涵盖所有国家
    # =========================================================================
    def plot_radar_comparison(self, year=2025):
        """Visualization 3: Full Scope Radar Comparison (3 Subplots x 3 Countries + USA Baseline)"""
        print(f"--- Generating Full Radar Comparison {year} ---")
        df_year = self.df_norm[self.df_norm['Year'] == year].set_index('Country')
        
        # 1. 定义维度
        dims = ['Compute_Power', 'AI_Researchers', 'AI_Patents', 'Gov_Investment', 'AI_Market_Size']
        
        # 2. 定义分组 (逻辑：Tier 1, Tier 2, Tier 3)
        # 每个组都将与 USA (基准) 进行对比
        groups = [
            # Group 1: 核心挑战者 (Top Challengers)
            ['China', 'UK', 'India'],
            # Group 2: 老牌工业国 (Established Powers)
            ['Germany', 'Japan', 'France'],
            # Group 3: 新兴与特色国 (Emerging & Specialized)
            ['Canada', 'South Korea', 'UAE']
        ]
        
        # 3. 设置画布 (1行3列)
        fig, axes = plt.subplots(1, 3, figsize=(24, 8), subplot_kw=dict(polar=True))
        
        # 雷达图几何参数
        angles = np.linspace(0, 2 * np.pi, len(dims), endpoint=False).tolist()
        angles += angles[:1]  # 闭合
        
        # 颜色盘 (除去蓝色，因为蓝色留给USA)
        colors = ['#d62728', '#2ca02c', '#ff7f0e'] # 红、绿、橙
        
        # 4. 循环绘制每个子图
        titles = ["Tier 1: Major Competitors", "Tier 2: Established Powers", "Tier 3: Specialized Players"]
        
        for idx, (ax, group) in enumerate(zip(axes, groups)):
            # --- 绘制 USA (Baseline) ---
            if 'USA' in df_year.index:
                values_usa = df_year.loc['USA', dims].values.flatten().tolist()
                values_usa += values_usa[:1]
                # USA样式：蓝色填充，半透明，作为背景
                ax.plot(angles, values_usa, linewidth=2, linestyle='--', color='#1f77b4', label='USA (Baseline)')
                ax.fill(angles, values_usa, alpha=0.15, color='#1f77b4')
            
            # --- 绘制 组内其他国家 ---
            for i, country in enumerate(group):
                if country in df_year.index:
                    values = df_year.loc[country, dims].values.flatten().tolist()
                    values += values[:1]
                    
                    c = colors[i % len(colors)]
                    ax.plot(angles, values, linewidth=2, label=country, color=c)
                    # 组内国家不填充，保持线条清晰，避免重叠混乱
                    # ax.fill(angles, values, alpha=0.05, color=c) 
            
            # --- 子图样式设置 ---
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(dims, size=10, weight='bold')
            
            # 设置Y轴刻度 (0.2 - 1.0)
            ax.set_yticks([0.2, 0.4, 0.6, 0.8, 1.0])
            ax.set_yticklabels(['', '0.4', '', '0.8', ''], color='gray', size=8) # 简化标签
            ax.set_ylim(0, 1.05)
            
            # 标题与图例
            ax.set_title(titles[idx], y=1.1, fontsize=14, weight='bold', color='#333333')
            ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=2, frameon=False)
            
        plt.suptitle(f"Global AI Competitiveness Radar ({year}): USA vs The World", fontsize=18, y=1.02)
        plt.tight_layout()
        
        # 保存
        plt.savefig(os.path.join(FIGURES_DIR, "Phase2_Dimension_Radar.png"), bbox_inches='tight', dpi=300)
        plt.close()

    def export_gaps(self, year=2025):
        # ... (保持不变) ...
        df_2025 = self.df_norm[self.df_norm['Year'] == year].set_index('Country')
        dimensions = {
            'Compute': ['Compute_Power', 'Supercomputers', 'AI_Chips'],
            'Talent': ['AI_Researchers', 'Top_Scholars', 'Talent_Inflow'],
            'Data': ['Data_Centers', 'Internet_Penetration', '5G_Coverage'],
            'Algorithm': ['LLM_Count', 'AI_Papers', 'AI_Patents']
        }
        scores = {}
        for dim, cols in dimensions.items():
            valid_cols = [c for c in cols if c in df_2025.columns]
            if valid_cols: scores[dim] = df_2025[valid_cols].mean(axis=1) * 100
            else: scores[dim] = 0
        score_df = pd.DataFrame(scores)
        if 'China' in score_df.index and 'USA' in score_df.index:
            china_scores = score_df.loc['China']
            usa_scores = score_df.loc['USA']
            gaps = (china_scores - usa_scores).to_dict()
            gaps = {k: (0 if not np.isfinite(v) else v) for k, v in gaps.items()}
            with open(os.path.join(RESULTS_DIR, "gaps_2025.json"), "w") as f: json.dump(gaps, f)
        return score_df

def run(df, df_norm):
    p2 = Phase2Evaluation(df, df_norm)
    print("\n>>> Phase 2: Dual-Dimension Evaluation")
    str_df = p2.calculate_hard_strength()
    eff_df = p2.calculate_efficiency()
    merged_df = pd.merge(str_df, eff_df, on='Country')
    merged_df['Final_Score'] = 0.6 * merged_df['Score_Strength'] + 0.4 * merged_df['Score_Efficiency']
    
    p2.visualize_quadrant(merged_df)
    p2.plot_bar_rankings(merged_df)
    p2.plot_radar_comparison() # 调用新的雷达图方法
    p2.export_gaps()
    
    merged_df.to_csv(os.path.join(RESULTS_DIR, "figure_data", "Phase2_Strength_Efficiency_Quadrant.csv"), index=False)
    return merged_df