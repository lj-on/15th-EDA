"""
군집 0과 군집 2 간 Features 비교분석

두 군집 간 각 feature의 유의미한 차이를 통계적으로 검정하고 비교합니다.
- t-test (정규분포 가정) 또는 Mann-Whitney U test (비모수)
- 효과 크기 (Cohen's d)
- 시각화
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")


def find_project_root():
    _cwd = Path.cwd()
    if (_cwd / "theta_feature_vis.ipynb").exists():
        return _cwd
    if (_cwd / "Q2_code" / "theta_feature_vis.ipynb").exists():
        return _cwd / "Q2_code"
    return _cwd


def get_feature_columns(df):
    """비교에 사용할 feature 컬럼"""
    candidates = [
        "accuracy",
        "avg_response_time_ms",
        "n_problems",
        "abandon_rate",
        "time_on_task_total_ms",
        "consecutive_days",
        "active_days",
        "erase_rate",
        "undo_rate",
        "answer_change_rate",
        "explanation_adoption_rate",
        "explanation_after_wrong_rate",
        "source_entropy",
        "adaptive_offer",
        "media_play_rate",
        "accuracy_per_time",
        "accuracy_per_problem",
        "theta",
        "theta_z",
    ]
    return [c for c in candidates if c in df.columns]


def load_clustered_data():
    """클러스터링된 학생 데이터 로드"""
    Q2_CODE = find_project_root()
    OUT = Q2_CODE / "outputs"
    path = OUT / "student_features_clustered.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Clustered data not found: {path}. Run student_clustering.py first."
        )
    df = pd.read_csv(path)
    if "cluster_4" not in df.columns:
        raise ValueError("cluster_4 column not found.")
    df = df[df["cluster_4"].notna()].copy()
    df["cluster_4"] = df["cluster_4"].astype(int)
    return df


def test_normality(data, alpha=0.05):
    """Shapiro-Wilk test로 정규성 검정"""
    if len(data) < 3:
        return False
    if len(data) > 5000:  # 너무 크면 샘플링
        data = np.random.choice(data, 5000, replace=False)
    try:
        stat, p = stats.shapiro(data)
        return p >= alpha
    except:
        return False


def cohens_d(group1, group2):
    """Cohen's d 효과 크기 계산"""
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    
    # Pooled standard deviation
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    
    if pooled_std == 0:
        return np.nan
    
    d = (np.mean(group1) - np.mean(group2)) / pooled_std
    return d


def compare_clusters(df, cluster1=0, cluster2=2, alpha=0.05):
    """두 군집 간 feature 비교"""
    feature_cols = get_feature_columns(df)
    
    cluster1_data = df[df["cluster_4"] == cluster1]
    cluster2_data = df[df["cluster_4"] == cluster2]
    
    if len(cluster1_data) == 0 or len(cluster2_data) == 0:
        raise ValueError(f"Cluster {cluster1} or {cluster2} not found in data.")
    
    print(f"\nComparing Cluster {cluster1} (n={len(cluster1_data):,}) vs Cluster {cluster2} (n={len(cluster2_data):,})")
    print("="*80)
    
    results = []
    
    for col in feature_cols:
        c1_vals = cluster1_data[col].dropna().values
        c2_vals = cluster2_data[col].dropna().values
        
        if len(c1_vals) < 3 or len(c2_vals) < 3:
            continue
        
        # 기본 통계
        c1_mean = np.mean(c1_vals)
        c2_mean = np.mean(c2_vals)
        c1_std = np.std(c1_vals, ddof=1)
        c2_std = np.std(c2_vals, ddof=1)
        c1_median = np.median(c1_vals)
        c2_median = np.median(c2_vals)
        mean_diff = c1_mean - c2_mean
        
        # 정규성 검정
        c1_normal = test_normality(c1_vals, alpha)
        c2_normal = test_normality(c2_vals, alpha)
        
        # 통계 검정
        if c1_normal and c2_normal:
            # t-test
            stat, p_value = stats.ttest_ind(c1_vals, c2_vals, equal_var=False)
            test_name = "t-test (Welch)"
        else:
            # Mann-Whitney U test
            stat, p_value = stats.mannwhitneyu(c1_vals, c2_vals, alternative='two-sided')
            test_name = "Mann-Whitney U"
        
        # 효과 크기
        effect_size = cohens_d(c1_vals, c2_vals)
        
        # 효과 크기 해석
        if abs(effect_size) < 0.2:
            effect_interpretation = "negligible"
        elif abs(effect_size) < 0.5:
            effect_interpretation = "small"
        elif abs(effect_size) < 0.8:
            effect_interpretation = "medium"
        else:
            effect_interpretation = "large"
        
        results.append({
            "feature": col,
            "cluster_0_mean": round(c1_mean, 4),
            "cluster_0_std": round(c1_std, 4),
            "cluster_0_median": round(c1_median, 4),
            "cluster_2_mean": round(c2_mean, 4),
            "cluster_2_std": round(c2_std, 4),
            "cluster_2_median": round(c2_median, 4),
            "mean_difference": round(mean_diff, 4),
            "test_statistic": round(stat, 4),
            "p_value": round(p_value, 6),
            "significant": p_value < alpha,
            "test_used": test_name,
            "cohens_d": round(effect_size, 4),
            "effect_size": effect_interpretation,
            "cluster_0_n": len(c1_vals),
            "cluster_2_n": len(c2_vals),
        })
    
    results_df = pd.DataFrame(results)
    return results_df


def visualize_comparison(df, results_df, cluster1=0, cluster2=2, output_dir=None):
    """비교 결과 시각화"""
    if output_dir is None:
        Q2_CODE = find_project_root()
        output_dir = Q2_CODE / "outputs" / "plots"
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    feature_cols = get_feature_columns(df)
    cluster1_data = df[df["cluster_4"] == cluster1]
    cluster2_data = df[df["cluster_4"] == cluster2]
    
    # 유의한 feature만 선택 (최대 12개)
    sig_features = results_df[results_df["significant"]].sort_values("p_value")["feature"].head(12).tolist()
    
    if len(sig_features) == 0:
        print("No significant features to visualize.")
        return
    
    # 박스플롯
    n_features = len(sig_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5 * n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    
    for idx, feat in enumerate(sig_features):
        ax = axes[idx]
        
        data_to_plot = [
            cluster1_data[feat].dropna().values,
            cluster2_data[feat].dropna().values
        ]
        
        bp = ax.boxplot(data_to_plot, labels=[f"Cluster {cluster1}", f"Cluster {cluster2}"], patch_artist=True)
        
        # 색상
        colors = ['#3498db', '#e74c3c']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_title(f"{feat}\n(p={results_df[results_df['feature']==feat]['p_value'].values[0]:.4f})", fontsize=10)
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
    
    # 빈 subplot 제거
    for idx in range(n_features, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.suptitle(f"Significant Feature Differences: Cluster {cluster1} vs Cluster {cluster2}", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_dir / f"cluster_{cluster1}_vs_{cluster2}_boxplot.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 효과 크기 막대그래프
    sig_results = results_df[results_df["significant"]].copy()
    if len(sig_results) > 0:
        fig, ax = plt.subplots(figsize=(10, max(6, len(sig_results) * 0.4)))
        y_pos = np.arange(len(sig_results))
        
        colors_bar = ['#e74c3c' if d < 0 else '#3498db' for d in sig_results['cohens_d']]
        ax.barh(y_pos, sig_results['cohens_d'], color=colors_bar, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sig_results['feature'])
        ax.set_xlabel("Cohen's d (Effect Size)")
        ax.set_title(f"Effect Sizes: Cluster {cluster1} vs Cluster {cluster2}\n(Red: Cluster {cluster1} < Cluster {cluster2}, Blue: Cluster {cluster1} > Cluster {cluster2})")
        ax.axvline(x=0, color='black', linestyle='--', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        plt.tight_layout()
        plt.savefig(output_dir / f"cluster_{cluster1}_vs_{cluster2}_effect_size.png", dpi=150, bbox_inches="tight")
        plt.close()
    
    print(f"\nVisualizations saved to: {output_dir}")


def main():
    Q2_CODE = find_project_root()
    OUT = Q2_CODE / "outputs"
    
    print("="*80)
    print("Cluster 0 vs Cluster 2: Feature Comparison Analysis")
    print("="*80)
    
    # 데이터 로드
    df = load_clustered_data()
    print(f"\nLoaded {len(df):,} students")
    
    # 비교 분석
    results_df = compare_clusters(df, cluster1=0, cluster2=2, alpha=0.05)
    
    # 결과 출력
    print("\n" + "="*80)
    print("Comparison Results:")
    print("="*80)
    
    sig_results = results_df[results_df["significant"]]
    print(f"\nSignificant differences (p < 0.05): {len(sig_results)}/{len(results_df)} features")
    
    if len(sig_results) > 0:
        print("\nTop significant features:")
        display_cols = ["feature", "cluster_0_mean", "cluster_2_mean", "mean_difference", 
                       "p_value", "cohens_d", "effect_size"]
        print(sig_results[display_cols].to_string(index=False))
    
    # 전체 결과 저장
    results_path = OUT / f"cluster_0_vs_2_comparison.csv"
    results_df.to_csv(results_path, index=False, encoding="utf-8-sig")
    print(f"\nFull results saved to: {results_path}")
    
    # 유의한 결과만 저장
    if len(sig_results) > 0:
        sig_path = OUT / f"cluster_0_vs_2_significant.csv"
        sig_results.to_csv(sig_path, index=False, encoding="utf-8-sig")
        print(f"Significant results saved to: {sig_path}")
    
    # 시각화
    visualize_comparison(df, results_df, cluster1=0, cluster2=2, output_dir=OUT / "plots")
    
    print("\n" + "="*80)
    print("Analysis completed!")
    print("="*80)
    
    return results_df


if __name__ == "__main__":
    results_df = main()
