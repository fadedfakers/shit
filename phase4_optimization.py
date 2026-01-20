import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import os
from mpl_toolkits.mplot3d import Axes3D
from utils import FIGURES_DIR, RESULTS_DIR

class ParticleSwarmOptimizer:
    def __init__(self, gaps, budget=140):
        # Default gaps if missing (fallback)
        default_gaps = {'Compute': -20, 'Talent': -10, 'Data': -15, 'Algorithm': -5}
        
        # Merge provided gaps with defaults to ensure all keys exist
        self.gaps_dict = {k: gaps.get(k, default_gaps.get(k, -10)) for k in ['Compute', 'Talent', 'Data', 'Algorithm']}
        
        self.gaps = np.array([self.gaps_dict['Compute'], self.gaps_dict['Talent'], 
                              self.gaps_dict['Data'], self.gaps_dict['Algorithm']])
        
        self.budget = budget
        self.dim_names = ['Compute', 'Talent', 'Data', 'Algorithm']
        
    def response_curve(self, investment, gap):
        """
        S-Curve Investment Response Function.
        - K (Potential): Proportional to the gap size (larger gap = more room to grow).
        - r (Rate): Fixed efficiency rate.
        """
        K = abs(gap) * 1.5 + 15  # Potential improvement cap
        # Logistic function shifted to start responding after some threshold
        return K / (1 + np.exp(-0.1 * (investment - 20)))

    def objective(self, allocation):
        """
        Objective: Maximize Comprehensive National Power (CNP).
        Logic: Barrel Theory (Min-Max).
        Z = 0.7 * Min(Dimensions) + 0.3 * Mean(Dimensions)
        """
        current_scores = 100 + self.gaps # Baseline USA=100
        improvements = self.response_curve(allocation, self.gaps)
        new_scores = current_scores + improvements
        
        # We want to maximize Z, so we return -Z for the minimizer
        z = 0.7 * np.min(new_scores) + 0.3 * np.mean(new_scores)
        return -z

    def optimize(self):
        print("--- Running PSO Optimization ---")
        num_particles = 50
        max_iter = 100
        dim = 4
        
        # Initialize Particles
        X = np.random.rand(num_particles, dim)
        # Normalize to budget
        X = X / X.sum(axis=1, keepdims=True) * self.budget
        V = np.random.randn(num_particles, dim) * 0.1
        
        pbest = X.copy()
        pbest_val = np.array([self.objective(x) for x in X])
        
        gbest = pbest[np.argmin(pbest_val)]
        gbest_val = np.min(pbest_val)
        
        w = 0.7  # Inertia
        c1 = 1.4 # Cognitive
        c2 = 1.4 # Social
        
        history = []
        
        for i in range(max_iter):
            # Update Velocity
            r1, r2 = np.random.rand(2)
            V = w*V + c1*r1*(pbest - X) + c2*r2*(gbest - X)
            X = X + V
            
            # Boundary Handling & Constraint Enforcement
            X = np.maximum(0, X) # Non-negative
            
            # Hard Constraint: Sum = Budget
            totals = X.sum(axis=1, keepdims=True)
            totals[totals==0] = 1 # Avoid div/0
            X = X / totals * self.budget
            
            # Evaluate
            vals = np.array([self.objective(x) for x in X])
            
            # Update Personal Best
            better_mask = vals < pbest_val
            pbest[better_mask] = X[better_mask]
            pbest_val[better_mask] = vals[better_mask]
            
            # Update Global Best
            min_val = np.min(vals)
            if min_val < gbest_val:
                gbest_val = min_val
                gbest = X[np.argmin(vals)]
                
            history.append(-gbest_val) # Track maximization score
            
        return gbest, history

    def visualize_radar(self, allocation):
        """Visualization 1: Before/After Radar Chart"""
        current_scores = 100 + self.gaps
        improvements = self.response_curve(allocation, self.gaps)
        future_scores = current_scores + improvements
        
        labels = self.dim_names
        num_vars = len(labels)
        angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
        angles += [angles[0]] # Close loop
        
        stats_us = [100]*4 + [100]
        stats_cn_now = np.concatenate((current_scores, [current_scores[0]]))
        stats_cn_fut = np.concatenate((future_scores, [future_scores[0]]))
        
        fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
        
        # USA Baseline
        ax.plot(angles, stats_us, '--', color='gray', label='USA Baseline (100)')
        
        # China Before
        ax.plot(angles, stats_cn_now, 'o-', color='red', label='China (2025)')
        ax.fill(angles, stats_cn_now, alpha=0.1, color='red')
        
        # China Optimized
        ax.plot(angles, stats_cn_fut, 'o-', color='green', linewidth=2, label='China (Optimized 2035)')
        ax.fill(angles, stats_cn_fut, alpha=0.2, color='green')
        
        ax.set_thetagrids(np.degrees(angles[:-1]), labels)
        plt.title(f"Phase 4: Optimization Results (Budget: ${self.budget}B)", y=1.05)
        plt.legend(loc='lower right', bbox_to_anchor=(1.2, 0.1))
        
        plt.savefig(os.path.join(FIGURES_DIR, "Phase4_Optimization_Radar.png"), bbox_inches='tight')
        plt.close()

    def visualize_allocation_pie(self, allocation):
        """Visualization 2: Optimal Budget Allocation"""
        plt.figure(figsize=(8, 8))
        # Donut Chart
        plt.pie(allocation, labels=self.dim_names, autopct='%1.1f%%', 
                startangle=140, pctdistance=0.85, colors=plt.cm.Paired.colors)
        
        centre_circle = plt.Circle((0,0), 0.70, fc='white')
        fig = plt.gcf()
        fig.gca().add_artist(centre_circle)
        
        plt.title(f"Phase 4: Optimal Budget Allocation (Total ${self.budget}B)")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase4_Budget_Allocation.png"))
        plt.close()

    def visualize_waterfall(self, allocation):
        """Visualization 3: Score Improvement Breakdown"""
        improvements = self.response_curve(allocation, self.gaps)
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(self.dim_names, improvements, color='forestgreen', alpha=0.7)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                     f'+{height:.1f}', ha='center', va='bottom')
            
        plt.title("Phase 4: Projected Score Gain by Dimension")
        plt.ylabel("Score Increase (Points)")
        plt.ylim(0, max(improvements)*1.2)
        plt.grid(axis='y', linestyle='--', alpha=0.5)
        plt.savefig(os.path.join(FIGURES_DIR, "Phase4_Gap_Analysis.png"))
        plt.close()

    def visualize_response_surface(self):
        """Visualization 4: 3D Response Surface (Compute vs Talent)"""
        # Simulate surface for the first two dimensions
        x = np.linspace(0, 100, 30) # Compute Invest
        y = np.linspace(0, 100, 30) # Talent Invest
        X, Y = np.meshgrid(x, y)
        
        # Simplified response logic for visualization
        # Z represents the Combined Score Gain
        Z = (self.response_curve(X, -20) + self.response_curve(Y, -10)) / 2
        
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        surf = ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.9)
        
        ax.set_xlabel('Invest: Compute ($B)')
        ax.set_ylabel('Invest: Talent ($B)')
        ax.set_zlabel('Score Gain')
        ax.set_title('Phase 4: Investment Efficiency Response Surface')
        fig.colorbar(surf, shrink=0.5, aspect=5)
        
        plt.savefig(os.path.join(FIGURES_DIR, "Phase4_Response_Surface.png"))
        plt.close()

def run():
    # 1. Load Data
    gap_file = os.path.join(RESULTS_DIR, "gaps_2025.json")
    if not os.path.exists(gap_file):
        print("Warning: 'gaps_2025.json' not found. Using default hypothetical gaps.")
        gaps = {'Compute': -20, 'Talent': -10, 'Data': -15, 'Algorithm': -5}
    else:
        with open(gap_file, 'r') as f:
            gaps = json.load(f)
    
    # 2. Optimize
    pso = ParticleSwarmOptimizer(gaps)
    best_alloc, hist = pso.optimize()
    
    print("\n=== Optimization Results ===")
    print(f"Optimal Allocation (Total ${pso.budget}B):")
    for dim, val in zip(pso.dim_names, best_alloc):
        print(f"  {dim}: ${val:.2f}B")
    
    # 3. Visualize
    pso.visualize_radar(best_alloc)
    pso.visualize_allocation_pie(best_alloc)
    pso.visualize_waterfall(best_alloc)
    pso.visualize_response_surface()
    print("Phase 4 Optimization and Plots Generated.")