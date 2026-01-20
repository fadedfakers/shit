import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.tsa.stattools import grangercausalitytests
from utils import DataLoader, FIGURES_DIR, RESULTS_DIR
import os

class Phase1Mechanism:
    def __init__(self, df):
        self.df = df
        
    def plot_correlation_matrix(self):
        """1. Visualization: Correlation Heatmap of Key Indicators"""
        print("Generating Correlation Matrix...")
        # Select key indicators to avoid clutter
        cols = ['Gov_Investment', 'AI_Researchers', 'Compute_Power', 'AI_Patents', 
                'AI_Market_Size', 'GDP', 'Internet_Penetration', 'LLM_Count']
        
        # Ensure columns exist (handle translation/renaming)
        valid_cols = [c for c in cols if c in self.df.columns]
        corr = self.df[valid_cols].corr()
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', vmin=-1, vmax=1)
        plt.title("Phase 1: Correlation Matrix of Key Indicators")
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase1_Correlation_Matrix.png"))
        plt.close()

    def plot_lagged_scatter(self):
        """2. Visualization: Lagged Effect (Investment t vs Patents t+2)"""
        print("Generating Lagged Scatter Plot...")
        df_plot = self.df.copy()
        
        # Calculate Lag-2 for Patents (grouped by Country to ensure data consistency)
        df_plot['Patents_Lag2'] = df_plot.groupby('Country')['AI_Patents'].shift(-2)
        
        # Remove NaN created by shifting
        df_plot = df_plot.dropna(subset=['Patents_Lag2', 'Gov_Investment'])
        
        plt.figure(figsize=(10, 6))
        
        # Scatter plot with different colors for countries
        sns.scatterplot(data=df_plot, x='Gov_Investment', y='Patents_Lag2', 
                        hue='Country', style='Country', s=100, alpha=0.8, palette='tab10')
        
        # Add a global trend line to show the general mechanism
        sns.regplot(data=df_plot, x='Gov_Investment', y='Patents_Lag2', 
                    scatter=False, color='gray', line_kws={'linestyle':'--', 'label':'Global Trend'})
        
        plt.title("Phase 1: Time-Lagged Mechanism (Investment T -> Patents T+2)")
        plt.xlabel("Government Investment (Year T)")
        plt.ylabel("AI Patents (Year T+2)")
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase1_Lagged_Effect.png"))
        plt.close()

    def calculate_granger_stats(self):
        """3. Statistics: Run Granger Causality Tests"""
        print("Running Granger Causality Tests...")
        target_countries = ['China', 'USA']
        results = []
        
        for country in target_countries:
            if country not in self.df['Country'].unique():
                continue
                
            sub = self.df[self.df['Country'] == country]
            
            # Test 1: Investment -> Patents (Lag 1 & 2)
            try:
                gc = grangercausalitytests(sub[['AI_Patents', 'Gov_Investment']], maxlag=2, verbose=False)
                # Get P-value for F-test at Lag 2
                p_val = gc[2][0]['ssr_ftest'][1]
                results.append({'Country': country, 'Path': 'Invest->Patents', 'Lag': 2, 'P-Value': p_val})
            except Exception as e:
                print(f"Skipping Granger for {country} (Data Insufficient): {e}")

            # Test 2: Patents -> Market Size (Lag 1 & 2)
            try:
                gc = grangercausalitytests(sub[['AI_Market_Size', 'AI_Patents']], maxlag=2, verbose=False)
                p_val = gc[2][0]['ssr_ftest'][1]
                results.append({'Country': country, 'Path': 'Patents->Market', 'Lag': 2, 'P-Value': p_val})
            except:
                pass
                
        res_df = pd.DataFrame(results)
        res_df.to_csv(os.path.join(RESULTS_DIR, "Phase1_Granger_Stats.csv"), index=False)
        print("Granger Stats Saved.")
        return res_df

    def plot_network_graph(self):
        """4. Visualization: Mechanism Network Graph"""
        print("Generating Network Graph...")
        G = nx.DiGraph()
        
        # Define Layers (Input -> Innovation -> Output)
        layers = {
            0: ['Gov_Investment', 'AI_Researchers'],
            1: ['Compute_Power', 'AI_Patents'],
            2: ['AI_Market_Size', 'LLM_Count']
        }
        
        pos = {}
        for layer_idx, nodes in layers.items():
            for i, node in enumerate(nodes):
                G.add_node(node, layer=layer_idx)
                # Position logic: x=layer, y=centered index
                pos[node] = (layer_idx * 3, i - 0.5)

        # Conceptual edges
        edges_config = [
            ('Gov_Investment', 'Compute_Power'),
            ('Gov_Investment', 'AI_Patents'),
            ('AI_Researchers', 'AI_Patents'),
            ('Compute_Power', 'LLM_Count'),
            ('AI_Patents', 'AI_Market_Size'),
            ('LLM_Count', 'AI_Market_Size')
        ]
        
        # Calculate REAL correlation weights for these edges
        corr_matrix = self.df.corr(numeric_only=True)
        
        for u, v in edges_config:
            if u in corr_matrix.index and v in corr_matrix.columns:
                weight = abs(corr_matrix.loc[u, v]) # Use actual data correlation
            else:
                weight = 0.5 # Default fallback
            
            G.add_edge(u, v, weight=weight)

        plt.figure(figsize=(12, 7))
        
        # Draw edges with variable width
        edges = G.edges()
        weights = [G[u][v]['weight'] * 6 for u, v in edges] # Scale for visibility
        
        nx.draw_networkx_nodes(G, pos, node_size=3500, node_color='skyblue', edgecolors='#333333', linewidths=1.5)
        nx.draw_networkx_edges(G, pos, width=weights, edge_color='#666666', arrowstyle='-|>', arrowsize=25, connectionstyle="arc3,rad=0.1")
        
        # Draw labels with a nice font
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', font_family='sans-serif')
        
        # Add Edge Labels (Correlation Values)
        edge_labels = { (u,v): f"{G[u][v]['weight']:.2f}" for u,v in G.edges() }
        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=8)
        
        plt.title("Phase 1: Causal Mechanism Network (Edge Width = Correlation Strength)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_DIR, "Phase1_Causal_Network.png"))
        plt.close()

def run(df_norm):
    p1 = Phase1Mechanism(df_norm)
    
    # 1. Heatmap
    p1.plot_correlation_matrix()
    
    # 2. Scatter with Lag
    p1.plot_lagged_scatter()
    
    # 3. Statistical Test
    p1.calculate_granger_stats()
    
    # 4. Path Diagram
    p1.plot_network_graph()