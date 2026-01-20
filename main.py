import utils
import phase1_mechanism
import phase2_evaluation
import phase3_prediction
import phase4_optimization  # 确保这里已取消注释
import phase5_sensitivity   # <--- 添加这一行
import warnings
import pandas as pd

def main():
    warnings.filterwarnings('ignore')
    print("==========================================")
    print("   ICM 2026 Problem B - Enhanced Viz      ")
    print("==========================================")
    
    # 0. Data
    print("\n[Phase 0] Loading Data...")
    loader = utils.DataLoader()
    df = loader.load_data()
    df_norm = loader.get_normalized_data(df)
    
    # 1. Mechanism
    print("\n[Phase 1] Generating Mechanism Visualizations...")
    phase1_mechanism.run(df_norm)
    
    # 2. Evaluation
    print("\n[Phase 2] Generating Evaluation Visualizations...")
    phase2_results_df = phase2_evaluation.run(df, df_norm)
    
    # 3. Prediction
    print("\n[Phase 3] Generating Prediction Visualizations (7 Charts)...")
    phase3_prediction.run(df_norm, phase2_results_df)
    
    # 4. Optimization
    print("\n[Phase 4] Generating Optimization Visualizations...")
    phase4_optimization.run() # 确保运行了优化，因为 Phase 5 需要读取它生成的 gaps_2025.json
    
    # 5. Sensitivity Analysis
    print("\n[Phase 5] Running Sensitivity Analysis...")
    phase5_sensitivity.run(df_norm, phase2_results_df) # <--- 调用运行
    
    print("\nDone! Check 'results/figures/' for all images.")

if __name__ == "__main__":
    main()