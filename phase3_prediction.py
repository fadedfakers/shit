import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy.optimize import curve_fit
from utils import FIGURES_DIR, RESULTS_DIR
import os
import platform

# 设置中文字体支持
def setup_chinese_font():
    """配置 matplotlib 中文字体"""
    system = platform.system()
    
    # 根据操作系统选择字体
    if system == 'Windows':
        # Windows 系统常用中文字体
        font_list = ['Microsoft YaHei', 'SimHei', 'KaiTi', 'FangSong', 'SimSun']
    elif system == 'Darwin':  # macOS
        font_list = ['Arial Unicode MS', 'PingFang SC', 'STHeiti', 'Heiti TC']
    else:  # Linux
        font_list = ['WenQuanYi Micro Hei', 'WenQuanYi Zen Hei', 'Noto Sans CJK SC', 'Droid Sans Fallback']
    
    # 获取系统可用字体
    available_fonts = [f.name for f in matplotlib.font_manager.fontManager.ttflist]
    
    # 尝试设置字体
    font_set = False
    for font in font_list:
        if font in available_fonts:
            plt.rcParams['font.sans-serif'] = [font] + [f for f in plt.rcParams['font.sans-serif'] if f != font]
            print(f"✓ 成功设置中文字体: {font}")
            font_set = True
            break
    
    if not font_set:
        # 如果所有字体都失败，尝试使用系统默认中文字体
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
        print("⚠ 警告: 未找到合适的中文字体，可能无法正常显示中文")
        print(f"   可用字体列表（前10个）: {available_fonts[:10]}")
    
    # 解决负号显示问题
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置图表样式
    plt.rcParams['figure.dpi'] = 300
    plt.rcParams['savefig.dpi'] = 300
    plt.rcParams['savefig.bbox'] = 'tight'
    
    # 设置默认字体大小
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.titlesize'] = 12
    plt.rcParams['axes.labelsize'] = 11
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

# 初始化字体设置
setup_chinese_font()

class Phase3Prediction:
    def __init__(self, df_norm, strategy_factors=None):
        self.df = df_norm.copy()
        self.strategy_factors = strategy_factors if strategy_factors else {}
        # 定义颜色盘，保证全流程一致
        self.colors = {
            'USA': '#1f77b4', 'China': '#d62728', 'India': '#ff7f0e',
            'UK': '#2ca02c', 'Germany': '#9467bd', 'Japan': '#8c564b',
            'France': '#e377c2', 'Canada': '#7f7f7f', 'South Korea': '#bcbd22', 'UAE': '#17becf'
        }

    def _verhulst_model(self, t, K, P0, r):
        """Verhulst S-Curve: K=Capacity, P0=Start, r=Rate"""
        return K / (1 + (K / P0 - 1) * np.exp(-r * t))

    def fit_and_predict(self, y_history, steps=10, country_name='Unknown'):
        """
        生成预测数据，包含 'Base' (基础), 'Optimistic' (乐观), 'Conservative' (保守) 三种场景
        返回: (final_base, final_opt, final_cons, fit_params, t_hist, y_fit_hist)
        """
        n_history = len(y_history)
        current_val = y_history[-1]
        t_hist = np.arange(n_history)
        
        # --- 1. 拟合 S 曲线 ---
        # 约束: 上限 K 至少是当前的 1.1 倍，最大 5 倍
        bounds = ([max(y_history)*1.1, 0, 0.01], [max(y_history)*5.0, np.inf, 1.5])
        p0_guess = [max(y_history)*1.5, y_history[0], 0.2]
        
        use_s_curve = True
        popt = None
        
        try:
            popt, _ = curve_fit(self._verhulst_model, t_hist, y_history, p0=p0_guess, bounds=bounds, maxfev=10000)
            t_future = np.arange(n_history, n_history + steps)
            preds_base = self._verhulst_model(t_future, *popt)
            # 计算历史拟合值（用于可视化）
            y_fit_hist = self._verhulst_model(t_hist, *popt)
        except:
            # 降级方案：线性外推
            use_s_curve = False
            z = np.polyfit(t_hist, y_history, 1)
            p = np.poly1d(z)
            preds_base = p(np.arange(n_history, n_history + steps))
            preds_base = np.maximum(preds_base, current_val)
            # 线性拟合的历史值
            y_fit_hist = p(t_hist)

        # --- 2. 应用战略因子 (Strategy Impact) ---
        factor = self.strategy_factors.get(country_name, 1.0)
        
        # 基础预测 (Standard)
        final_base = preds_base * np.array([factor ** (i+1) for i in range(steps)])
        
        # 乐观预测 (Optimistic): 假设战略执行极其完美 (Factor + 0.02)
        opt_factor = factor + 0.02
        final_opt = preds_base * np.array([opt_factor ** (i+1) for i in range(steps)])
        
        # 保守预测 (Conservative): 假设战略受阻 (Factor - 0.01)
        cons_factor = max(1.0, factor - 0.01)
        final_cons = preds_base * np.array([cons_factor ** (i+1) for i in range(steps)])
        
        # 保存拟合参数信息
        fit_params = {
            'use_s_curve': use_s_curve,
            'K': popt[0] if popt is not None else None,
            'P0': popt[1] if popt is not None else None,
            'r': popt[2] if popt is not None else None,
            'factor': factor
        }
        
        return final_base, final_opt, final_cons, fit_params, t_hist, y_fit_hist

    def generate_predictions(self, years_future=10):
        print(f"--- Running Prediction Model (2026-{2025+years_future}) ---")
        countries = self.df['Country'].unique()
        future_years = list(range(2026, 2026 + years_future))
        
        results = []
        # 保存拟合信息用于可视化
        self.fit_info = {}
        
        # 计算历史综合分 (作为拟合输入)
        cols = ['AI_Patents', 'AI_Market_Size', 'Unicorn_Companies', 'Gov_Investment']
        valid = [c for c in cols if c in self.df.columns]
        self.df['Composite'] = self.df[valid].mean(axis=1) * 100
        
        for country in countries:
            country_data = self.df[self.df['Country'] == country].sort_values('Year')
            y_hist = country_data['Composite'].rolling(2, min_periods=1).mean().values
            hist_years = country_data['Year'].values
            
            # 获取三种场景的预测和拟合信息
            pred_base, pred_opt, pred_cons, fit_params, t_hist, y_fit_hist = self.fit_and_predict(
                y_hist, steps=years_future, country_name=country)
            
            # 保存拟合信息
            self.fit_info[country] = {
                'hist_years': hist_years,
                'y_hist': y_hist,
                'y_fit_hist': y_fit_hist,
                'future_years': future_years,
                'pred_base': pred_base,
                'pred_opt': pred_opt,
                'pred_cons': pred_cons,
                'fit_params': fit_params
            }
            
            for i, y in enumerate(future_years):
                results.append({
                    'Country': country,
                    'Year': y,
                    'Score': pred_base[i],
                    'Score_Opt': pred_opt[i],
                    'Score_Cons': pred_cons[i]
                })
        
        preds_df = pd.DataFrame(results)
        
        # 保存预测结果到CSV
        preds_df.to_csv(os.path.join(RESULTS_DIR, "predictions_2035.csv"), index=False)
        print(f"✓ 预测结果已保存: predictions_2035.csv")
        
        return preds_df

    # ================= 绘图区域 (6张图) =================

    def plot_bump_chart(self, df):
        """[图1] 排名变化图"""
        # 确保使用中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        pivot = df.pivot(index='Year', columns='Country', values='Score')
        ranks = pivot.rank(axis=1, ascending=False)
        
        plt.figure(figsize=(14, 8))
        for col in ranks.columns:
            is_top = col in ['USA', 'China', 'India']
            lw = 4 if is_top else 1.5
            plt.plot(ranks.index, ranks[col], label=col, lw=lw, marker='o', 
                     color=self.colors.get(col, 'gray'), alpha=0.9 if is_top else 0.5)
            
            # 标注文字
            plt.text(2035.2, ranks[col].iloc[-1], col, va='center', 
                     fontweight='bold' if is_top else 'normal', color=self.colors.get(col, 'black'))

        plt.gca().invert_yaxis()
        plt.yticks(range(1, 11))
        plt.title("Fig 1. AI Competitiveness Ranking Forecast (2026-2035)", fontsize=14, pad=20)
        plt.ylabel("Rank (1=Top)")
        plt.grid(ls='--', alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase3_1_Ranking_Bump.png"), dpi=300)
        plt.close()
        
        # 导出排名数据
        ranks_export = ranks.reset_index()
        ranks_export = ranks_export.melt(id_vars='Year', var_name='Country', value_name='Rank')
        ranks_export = ranks_export.sort_values(['Year', 'Rank'])
        ranks_export.to_csv(os.path.join(RESULTS_DIR, "Phase3_Ranking_Bump_Chart.csv"), index=False)
        print(f"✓ 排名数据已保存: Phase3_Ranking_Bump_Chart.csv")

    def plot_trajectory(self, df):
        """[图2] 综合得分轨迹"""
        # 确保使用中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.figure(figsize=(12, 6))
        sns.lineplot(data=df, x='Year', y='Score', hue='Country', palette=self.colors, lw=2.5)
        plt.title("Fig 2. Composite Score Trajectory (Base Scenario)", fontsize=14)
        plt.grid(True, alpha=0.3)
        plt.legend(bbox_to_anchor=(1.02, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase3_2_Score_Trajectory.png"), dpi=300)
        plt.close()
        
        # 导出轨迹数据（已在generate_predictions中保存为predictions_2035.csv）
        print(f"✓ 轨迹数据已保存: predictions_2035.csv")

    def plot_heatmap(self, df):
        """[图3] 增长率热力图"""
        # 确保使用中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        pivot = df.pivot(index='Country', columns='Year', values='Score')
        growth = pivot.pct_change(axis=1) * 100
        
        plt.figure(figsize=(12, 7))
        sns.heatmap(growth.iloc[:, 1:], cmap='RdYlGn', annot=True, fmt=".1f", center=0)
        plt.title("Fig 3. Predicted Annual Growth Rate (%)", fontsize=14)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase3_3_Growth_Heatmap.png"), dpi=300)
        plt.close()
        
        # 导出增长率数据
        growth_export = growth.reset_index()
        growth_export = growth_export.melt(id_vars='Country', var_name='Year', value_name='Growth_Rate_Pct')
        growth_export = growth_export.sort_values(['Country', 'Year'])
        growth_export.to_csv(os.path.join(RESULTS_DIR, "Phase3_Growth_Heatmap.csv"), index=False)
        print(f"✓ 增长率数据已保存: Phase3_Growth_Heatmap.csv")

    def plot_2035_bar(self, df):
        """[图4] 2035 最终得分"""
        # 确保使用中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        data_2035 = df[df['Year'] == 2035].sort_values('Score', ascending=False)
        plt.figure(figsize=(10, 6))
        sns.barplot(data=data_2035, x='Score', y='Country', palette='viridis')
        plt.title("Fig 4. Forecasted AI Competitiveness in 2035", fontsize=14)
        plt.xlabel("Projected Score")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase3_4_2035_Snapshot.png"), dpi=300)
        plt.close()
        
        # 导出2035年快照数据
        data_2035_export = data_2035[['Country', 'Score']].copy()
        data_2035_export['Rank'] = range(1, len(data_2035_export) + 1)
        data_2035_export = data_2035_export[['Rank', 'Country', 'Score']]
        data_2035_export.to_csv(os.path.join(RESULTS_DIR, "Phase3_2035_Forecast_Bar.csv"), index=False)
        print(f"✓ 2035年快照数据已保存: Phase3_2035_Forecast_Bar.csv")

    def plot_uncertainty_bounds(self, df):
        """[图5 NEW] 不确定性分析 (区间带状图) - 仅展示前3名"""
        # 确保使用中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        top_countries = ['China', 'USA', 'India']
        plt.figure(figsize=(12, 6))
        
        for country in top_countries:
            c_data = df[df['Country'] == country]
            color = self.colors.get(country, 'gray')
            
            # 画主线
            plt.plot(c_data['Year'], c_data['Score'], label=f"{country} (Base)", color=color, lw=3)
            
            # 画区间 (Optimistic - Conservative)
            plt.fill_between(c_data['Year'], c_data['Score_Cons'], c_data['Score_Opt'], 
                             color=color, alpha=0.2, label=f"{country} Uncertainty Range")
            
        plt.title("Fig 5. Prediction Uncertainty Analysis (Top 3 Countries)\n(Shaded Area: Range between Conservative and Optimistic Policies)", fontsize=14)
        plt.ylabel("Composite Score")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase3_5_Uncertainty_Bounds.png"), dpi=300)
        plt.close()
        
        # 导出不确定性分析数据
        uncertainty_data = []
        for country in top_countries:
            c_data = df[df['Country'] == country]
            for _, row in c_data.iterrows():
                uncertainty_data.append({
                    'Country': country,
                    'Year': row['Year'],
                    'Score_Base': row['Score'],
                    'Score_Optimistic': row['Score_Opt'],
                    'Score_Conservative': row['Score_Cons'],
                    'Uncertainty_Range': row['Score_Opt'] - row['Score_Cons']
                })
        uncertainty_df = pd.DataFrame(uncertainty_data)
        uncertainty_df.to_csv(os.path.join(RESULTS_DIR, "Phase3_Uncertainty_Bounds.csv"), index=False)
        print(f"✓ 不确定性分析数据已保存: Phase3_Uncertainty_Bounds.csv")

    def plot_gap_analysis(self, df):
        """[图6 NEW] 领跑者差距分析 (Gap to Leader)"""
        # 确保使用中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        pivot = df.pivot(index='Year', columns='Country', values='Score')
        # 假设 USA 是基准，计算其他国家与 USA 的差值 (如果是负数，表示落后)
        # 如果某个年份其他国家超过 USA，数值变为正
        
        challengers = ['China', 'India', 'UK', 'Germany']
        plt.figure(figsize=(12, 6))
        
        # 绘制 0 线 (USA Baseline)
        plt.axhline(0, color='black', lw=1, linestyle='--', label="USA Baseline")
        
        for country in challengers:
            gap = pivot[country] - pivot['USA']
            color = self.colors.get(country, 'gray')
            
            plt.plot(pivot.index, gap, label=country, color=color, lw=2.5)
            
            # 填充颜色，红色表示落后，绿色表示超越（如果有）
            plt.fill_between(pivot.index, gap, 0, where=(gap < 0), color=color, alpha=0.1)
            plt.fill_between(pivot.index, gap, 0, where=(gap >= 0), color=color, alpha=0.3)

        plt.title("Fig 6. Gap Analysis: Distance to USA (2026-2035)", fontsize=14)
        plt.ylabel("Score Difference (Relative to USA)")
        plt.legend(loc='lower right')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase3_6_Gap_Analysis.png"), dpi=300)
        plt.close()
        
        # 导出差距分析数据
        gap_data = []
        for country in challengers:
            gap = pivot[country] - pivot['USA']
            for year in pivot.index:
                gap_data.append({
                    'Year': year,
                    'Country': country,
                    'Gap_to_USA': gap[year]
                })
        gap_df = pd.DataFrame(gap_data)
        gap_df.to_csv(os.path.join(RESULTS_DIR, "Phase3_Gap_Analysis.csv"), index=False)
        print(f"✓ 差距分析数据已保存: Phase3_Gap_Analysis.csv")

    def plot_s_curve(self, countries=None):
        """[图7 NEW] S 型曲线拟合可视化 - 展示历史数据、拟合曲线和预测"""
        # 确保使用中文字体设置
        plt.rcParams['axes.unicode_minus'] = False
        
        if not hasattr(self, 'fit_info') or not self.fit_info:
            print("警告: 没有拟合信息，请先运行 generate_predictions()")
            return
        
        if countries is None:
            # 默认显示所有国家
            countries = sorted(list(self.fit_info.keys()))
        
        # 过滤出有数据的国家
        available_countries = [c for c in countries if c in self.fit_info]
        if not available_countries:
            print(f"警告: 指定的国家 {countries} 没有拟合数据")
            return
        
        # 计算网格布局：每行最多4个国家
        n_countries = len(available_countries)
        n_cols = min(4, n_countries)  # 每行最多4列
        n_rows = (n_countries + n_cols - 1) // n_cols  # 向上取整
        
        # 创建网格布局
        if n_countries == 1:
            fig, ax = plt.subplots(1, 1, figsize=(6, 5))
            axes_flat = [ax]
        else:
            fig, axes = plt.subplots(n_rows, n_cols, figsize=(6*n_cols, 5*n_rows))
            # 将 axes 展平为一维数组便于索引
            if n_rows == 1:
                axes_flat = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
            elif n_cols == 1:
                axes_flat = axes.flatten() if hasattr(axes, 'flatten') else list(axes)
            else:
                axes_flat = axes.flatten()
        
        # 先收集所有数据范围，用于统一坐标轴
        all_years = []
        all_y_values = []
        
        for country in available_countries:
            info = self.fit_info[country]
            # 收集所有年份
            all_years.extend(info['hist_years'].tolist())
            all_years.extend(info['future_years'])
            # 收集所有y值（历史、拟合、预测）
            all_y_values.extend(info['y_hist'].tolist())
            all_y_values.extend(info['y_fit_hist'].tolist())
            all_y_values.extend(info['pred_base'].tolist())
            all_y_values.extend(info['pred_opt'].tolist())
            all_y_values.extend(info['pred_cons'].tolist())
        
        # 计算全局坐标轴范围（添加5%的边距）
        x_min = min(all_years) - 0.5
        x_max = max(all_years) + 0.5
        y_min = max(0, min(all_y_values) * 0.95)  # 确保y_min >= 0
        y_max = max(all_y_values) * 1.05
        
        # 绘制所有子图
        for idx, country in enumerate(available_countries):
            ax = axes_flat[idx]
            info = self.fit_info[country]
            color = self.colors.get(country, 'gray')
            params = info['fit_params']
            
            # 1. 绘制历史数据点
            ax.scatter(info['hist_years'], info['y_hist'], 
                      color='black', s=60, zorder=5, label='历史数据', marker='o', 
                      edgecolors='white', linewidths=1.2)
            
            # 2. 绘制拟合的 S 曲线（历史部分）
            if params['use_s_curve']:
                # 生成更密集的时间点用于绘制平滑曲线
                t_smooth_hist = np.linspace(0, len(info['y_hist'])-1, 100)
                y_smooth_hist = self._verhulst_model(t_smooth_hist, params['K'], params['P0'], params['r'])
                # 将时间索引转换为实际年份
                year_start = info['hist_years'][0]
                year_end = info['hist_years'][-1]
                years_smooth_hist = np.linspace(year_start, year_end, 100)
                ax.plot(years_smooth_hist, y_smooth_hist, 
                       color='blue', linestyle='--', linewidth=1.8, alpha=0.7, 
                       label='S曲线拟合', zorder=3)
            else:
                # 线性拟合的情况
                ax.plot(info['hist_years'], info['y_fit_hist'], 
                       color='blue', linestyle='--', linewidth=1.8, alpha=0.7, 
                       label='线性拟合', zorder=3)
            
            # 3. 绘制预测部分（基础场景）
            ax.plot(info['future_years'], info['pred_base'], 
                   color=color, linestyle='-', linewidth=2.5, marker='s', markersize=5, 
                   label='基础预测', zorder=4)
            
            # 4. 绘制不确定性区间（乐观-保守）
            ax.fill_between(info['future_years'], info['pred_cons'], info['pred_opt'],
                           color=color, alpha=0.2, label='不确定性区间', zorder=2)
            
            # 5. 添加分界线（历史与预测之间）
            ax.axvline(x=info['hist_years'][-1], color='gray', linestyle=':', linewidth=1.2, 
                      alpha=0.5, zorder=1, label='预测起点')
            
            # 6. 添加参数信息文本（字体大小根据子图数量调整）
            font_size = 8 if n_countries > 4 else 9
            if params['use_s_curve']:
                param_text = f"K={params['K']:.1f}\nP₀={params['P0']:.1f}\nr={params['r']:.3f}\n因子={params['factor']:.3f}"
            else:
                param_text = f"线性外推\n因子={params['factor']:.3f}"
            
            ax.text(0.02, 0.98, param_text, transform=ax.transAxes,
                   fontsize=font_size, verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
            
            # 设置标题和标签（确保使用中文字体）
            ax.set_title(f"{country} - S曲线预测模型", fontsize=11, fontweight='bold', pad=8)
            ax.set_xlabel('年份', fontsize=10)
            ax.set_ylabel('综合分数', fontsize=10)
            ax.legend(loc='best', fontsize=8, framealpha=0.9, ncol=1)
            ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)
            ax.tick_params(labelsize=8)
            
            # 统一设置坐标轴范围
            ax.set_xlim(x_min, x_max)
            ax.set_ylim(y_min, y_max)
        
        # 隐藏多余的子图
        for idx in range(n_countries, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        # 设置总标题（确保使用中文字体）
        plt.suptitle("图7. S型曲线拟合与预测可视化（所有国家）", 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout(rect=[0, 0, 1, 0.98])  # 为总标题留出空间
        plt.savefig(os.path.join(FIGURES_DIR, "Phase3_7_S_Curve_Fitting.png"), dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✓ S曲线可视化图已生成: Phase3_7_S_Curve_Fitting.png (包含 {n_countries} 个国家)")
        
        # 导出S曲线拟合数据
        self._export_s_curve_data(available_countries)
    
    def _export_s_curve_data(self, countries):
        """导出S曲线拟合的详细数据"""
        # 1. 导出拟合参数
        params_data = []
        for country in countries:
            if country in self.fit_info:
                params = self.fit_info[country]['fit_params']
                params_data.append({
                    'Country': country,
                    'Model_Type': 'S-Curve' if params['use_s_curve'] else 'Linear',
                    'K_Capacity': params['K'] if params['K'] is not None else None,
                    'P0_Initial': params['P0'] if params['P0'] is not None else None,
                    'r_GrowthRate': params['r'] if params['r'] is not None else None,
                    'Strategy_Factor': params['factor']
                })
        params_df = pd.DataFrame(params_data)
        params_df.to_csv(os.path.join(RESULTS_DIR, "Phase3_S_Curve_Parameters.csv"), index=False)
        print(f"✓ S曲线拟合参数已保存: Phase3_S_Curve_Parameters.csv")
        
        # 2. 导出历史+拟合+预测的完整数据
        full_data = []
        for country in countries:
            if country in self.fit_info:
                info = self.fit_info[country]
                # 历史数据
                for i, year in enumerate(info['hist_years']):
                    full_data.append({
                        'Country': country,
                        'Year': year,
                        'Type': 'Historical',
                        'Value_Actual': info['y_hist'][i],
                        'Value_Fitted': info['y_fit_hist'][i],
                        'Value_Base': None,
                        'Value_Optimistic': None,
                        'Value_Conservative': None
                    })
                # 预测数据
                for i, year in enumerate(info['future_years']):
                    full_data.append({
                        'Country': country,
                        'Year': year,
                        'Type': 'Forecast',
                        'Value_Actual': None,
                        'Value_Fitted': None,
                        'Value_Base': info['pred_base'][i],
                        'Value_Optimistic': info['pred_opt'][i],
                        'Value_Conservative': info['pred_cons'][i]
                    })
        full_df = pd.DataFrame(full_data)
        full_df.to_csv(os.path.join(RESULTS_DIR, "Phase3_S_Curve_Fitting_Data.csv"), index=False)
        print(f"✓ S曲线完整拟合数据已保存: Phase3_S_Curve_Fitting_Data.csv")

def run(df_norm, phase2_results_df=None):
    # 1. 动态计算策略因子
    strategy_factors = {}
    if phase2_results_df is not None:
        print(">>> Calculating Strategy Factors from Phase 2 Data...")
        # 逻辑：Final_Score 越低，catch-up 空间越大；Strength 越高，基础越好
        for _, row in phase2_results_df.iterrows():
            c = row['Country']
            # 简单示例逻辑：基础增长 1.0 + (硬实力/2000) + (追赶红利)
            growth_potential = 1.01 + (row['Score_Strength'] * 0.0002) 
            if c == 'China': growth_potential += 0.03 # 政策加成
            if c == 'India': growth_potential += 0.035
            strategy_factors[c] = min(growth_potential, 1.08) # 封顶
    else:
        print(">>> Warning: No Phase 2 data. Using defaults.")
        strategy_factors = {'USA':1.02, 'China':1.05, 'India':1.06}

    # 2. 预测
    p3 = Phase3Prediction(df_norm, strategy_factors)
    preds = p3.generate_predictions()
    
    # 3. 绘图 (7张，新增S曲线可视化)
    if not preds.empty:
        p3.plot_bump_chart(preds)
        p3.plot_trajectory(preds)
        p3.plot_heatmap(preds)
        p3.plot_2035_bar(preds)
        p3.plot_uncertainty_bounds(preds)
        p3.plot_gap_analysis(preds)
        p3.plot_s_curve(None)  # None表示显示所有国家
        print("Phase 3 Visualizations (7 charts) Generated Successfully.")
        
    return preds