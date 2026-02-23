"""
군집별 문항 오답률 상관관계 분석 (사분면 기반)

클러스터링 결과를 바탕으로:
1. 문항을 사분면(Quadrant)으로 클러스터링 (correct_rate × confusion 기준)
2. 각 군집별 사분면별 오답률 계산
3. 군집 간 사분면 오답률 상관관계 분석 및 유의성 검정
"""

from pathlib import Path
import pandas as pd
import numpy as np
from scipy.stats import spearmanr, pearsonr, kruskal
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

# 사분면 순서 및 라벨
QUADRANT_ORDER = [
    "Q1_easy_highConf",   # 쉬운데 헷갈림
    "Q2_hard_highConf",   # 어렵고 헷갈림
    "Q3_hard_lowConf",    # 어려우나 안정적
    "Q4_easy_lowConf",    # 쉬움 + 안정적
]

QUADRANT_LABELS_KR = {
    "Q1_easy_highConf": "쉬운데 헷갈림",
    "Q2_hard_highConf": "어렵고 헷갈림",
    "Q3_hard_lowConf": "어려우나 안정적",
    "Q4_easy_lowConf": "쉬움+안정적",
}

QUADRANT_LABELS_EN = {
    "Q1_easy_highConf": "Q1 easy+confused",
    "Q2_hard_highConf": "Q2 hard+confused",
    "Q3_hard_lowConf": "Q3 hard+stable",
    "Q4_easy_lowConf": "Q4 easy+stable",
}

COLOR_MAP = {
    "Q1_easy_highConf": "#d62728",
    "Q2_hard_highConf": "#ff7f0e",
    "Q3_hard_lowConf": "#2ca02c",
    "Q4_easy_lowConf": "#1f77b4",
}

ALPHA = 0.05


def find_project_root():
    """프로젝트 루트 경로 찾기"""
    _cwd = Path.cwd()
    if (_cwd / "theta_feature_vis.ipynb").exists():
        return _cwd
    elif (_cwd / "Q2_code" / "theta_feature_vis.ipynb").exists():
        return _cwd / "Q2_code"
    else:
        return _cwd


def load_clustered_data():
    """클러스터링된 학생 데이터 로드"""
    Q2_CODE = find_project_root()
    OUT = Q2_CODE / "outputs"
    
    # 클러스터링 결과 파일 찾기
    clustered_path = OUT / "student_features_clustered.csv"
    if not clustered_path.exists():
        # 클러스터링이 안 되어 있으면 기본 파일에서 클러스터링 수행
        print("Clustered data not found. Running clustering first...")
        from student_clustering import main
        main()
        clustered_path = OUT / "student_features_clustered.csv"
    
    print(f"Loading clustered data from: {clustered_path}")
    df = pd.read_csv(clustered_path)
    
    # cluster_4가 없는 경우 에러
    if "cluster_4" not in df.columns:
        raise ValueError("cluster_4 column not found. Please run student_clustering.py first.")
    
    # cluster_4가 NaN인 행 제외
    df = df[df["cluster_4"].notna()].copy()
    df["cluster_4"] = df["cluster_4"].astype(int)
    df["user_id"] = df["user_id"].astype(str)
    
    print(f"Loaded {len(df):,} clustered students")
    print(f"Cluster distribution:")
    print(df["cluster_4"].value_counts().sort_index())
    
    return df


def load_kt1_data(user_ids=None, dataset_root=None):
    """KT1 데이터 로드 (문항별 정오답 정보)"""
    if dataset_root is None:
        Q2_CODE = find_project_root()
        # 워크스페이스 루트에서 KT1 찾기
        dataset_root = Q2_CODE.parent
    
    kt1_dir = dataset_root / "KT1"
    if not kt1_dir.exists():
        raise FileNotFoundError(f"KT1 directory not found: {kt1_dir}")
    
    print(f"\nLoading KT1 data from: {kt1_dir}")
    
    if user_ids is None:
        clustered_df = load_clustered_data()
        user_ids = set(clustered_df["user_id"].astype(str))
    else:
        user_ids = set(str(uid) for uid in user_ids)
    
    # KT1 파일들을 읽어서 문항별 정오답 정보 수집
    kt1_records = []
    kt1_files = list(kt1_dir.glob("u*.csv"))
    
    print(f"Processing {len(kt1_files):,} KT1 files (filtering to {len(user_ids):,} users)...")
    processed = 0
    
    for kt1_file in kt1_files:
        user_id = kt1_file.stem
        if user_id not in user_ids:
            continue
        
        try:
            df_user = pd.read_csv(kt1_file)
            if "question_id" not in df_user.columns or "correct" not in df_user.columns:
                continue
            
            # 벡터화된 처리
            df_user = df_user[df_user["question_id"].notna() & df_user["correct"].notna()].copy()
            if len(df_user) > 0:
                df_user["user_id"] = user_id
                kt1_records.append(df_user[["user_id", "question_id", "correct"]])
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed:,} users...")
        
        except Exception as e:
            print(f"  Warning: Failed to process {user_id}: {e}")
            continue
    
    if len(kt1_records) == 0:
        raise ValueError("No KT1 data loaded")
    
    kt1_df = pd.concat(kt1_records, ignore_index=True)
    kt1_df["question_id"] = kt1_df["question_id"].astype(str)
    kt1_df["correct"] = kt1_df["correct"].astype(float)
    print(f"\nLoaded {len(kt1_df):,} question attempts from {processed:,} users")
    
    return kt1_df


def load_kt4_data(user_ids=None, dataset_root=None):
    """KT4 데이터 로드 (문항별 confusion 지표 계산용)"""
    if dataset_root is None:
        Q2_CODE = find_project_root()
        dataset_root = Q2_CODE.parent
    
    kt4_dir = dataset_root / "KT4"
    if not kt4_dir.exists():
        print(f"Warning: KT4 directory not found: {kt4_dir}")
        return None
    
    print(f"\nLoading KT4 data from: {kt4_dir}")
    
    if user_ids is None:
        clustered_df = load_clustered_data()
        user_ids = set(clustered_df["user_id"].astype(str))
    else:
        user_ids = set(str(uid) for uid in user_ids)
    
    # KT4 파일들을 읽어서 문항별 confusion 액션 수집
    kt4_records = []
    kt4_files = list(kt4_dir.glob("u*.csv"))
    
    print(f"Processing {len(kt4_files):,} KT4 files (filtering to {len(user_ids):,} users)...")
    processed = 0
    
    for kt4_file in kt4_files:
        user_id = kt4_file.stem
        if user_id not in user_ids:
            continue
        
        try:
            df_user = pd.read_csv(kt4_file)
            if "action_type" not in df_user.columns or "item_id" not in df_user.columns:
                continue
            
            # 벡터화된 처리: q로 시작하는 item_id만 필터링
            df_user = df_user[df_user["item_id"].astype(str).str.startswith("q")].copy()
            if len(df_user) == 0:
                continue
            
            df_user["question_id"] = df_user["item_id"].astype(str)
            df_user["user_id"] = user_id
            
            # confusion 액션 카운트 (벡터화)
            df_user["is_erase"] = df_user["action_type"].isin(["erase_choice", "eliminate_choice"]).astype(int)
            df_user["is_undo"] = (df_user["action_type"] == "undo_erase_choice").astype(int)
            df_user["is_action"] = 1
            
            # 그룹화하여 집계
            confusion_summary = df_user.groupby(["user_id", "question_id"]).agg({
                "is_erase": "sum",
                "is_undo": "sum",
                "is_action": "sum"
            }).reset_index()
            confusion_summary.columns = ["user_id", "question_id", "erase_count", "undo_count", "total_actions"]
            
            kt4_records.append(confusion_summary)
            
            processed += 1
            if processed % 1000 == 0:
                print(f"  Processed {processed:,} users...")
        
        except Exception as e:
            print(f"  Warning: Failed to process {user_id}: {e}")
            continue
    
    if len(kt4_records) == 0:
        print("  No KT4 data found")
        return None
    
    kt4_df = pd.concat(kt4_records, ignore_index=True)
    print(f"\nLoaded {len(kt4_df):,} question-user confusion records from {processed:,} users")
    
    return kt4_df


def calculate_question_statistics(kt1_df, kt4_df=None):
    """문항별 통계 계산 (correct_rate, confusion 등)"""
    print("\nCalculating question statistics...")
    
    # 문항별 정답률 (correct_rate)
    question_stats = kt1_df.groupby("question_id").agg({
        "correct": ["mean", "count"]
    }).reset_index()
    question_stats.columns = ["question_id", "correct_rate", "n_attempts"]
    
    # confusion 계산: KT4 데이터가 있으면 사용, 없으면 fallback
    if kt4_df is not None and len(kt4_df) > 0:
        print("  Using KT4 data for confusion calculation...")
        
        # 문항별 confusion 액션 집계
        question_confusion = kt4_df.groupby("question_id").agg({
            "erase_count": "sum",
            "undo_count": "sum",
            "total_actions": "sum",
            "user_id": "nunique"  # 문항을 시도한 유저 수
        }).reset_index()
        question_confusion.columns = [
            "question_id", "total_erase", "total_undo", "total_actions", "n_users"
        ]
        
        # 문항별 평균 confusion rate 계산
        # confusion = (erase + undo) / total_actions (높을수록 헷갈림)
        question_confusion["confusion_rate"] = (
            (question_confusion["total_erase"] + question_confusion["total_undo"]) / 
            question_confusion["total_actions"].replace(0, np.nan)
        )
        
        # 문항별 평균 confusion (유저별 평균)
        user_question_confusion = kt4_df.groupby(["question_id", "user_id"]).agg({
            "erase_count": "sum",
            "undo_count": "sum",
            "total_actions": "sum"
        }).reset_index()
        user_question_confusion["user_confusion_rate"] = (
            (user_question_confusion["erase_count"] + user_question_confusion["undo_count"]) /
            user_question_confusion["total_actions"].replace(0, np.nan)
        )
        
        question_confusion_mean = user_question_confusion.groupby("question_id")["user_confusion_rate"].mean().reset_index()
        question_confusion_mean.columns = ["question_id", "q_confusion_mean"]
        
        # 병합
        question_stats = question_stats.merge(
            question_confusion_mean[["question_id", "q_confusion_mean"]],
            on="question_id",
            how="left"
        )
        
        # KT4 데이터가 없는 문항은 fallback 방법 사용
        missing_confusion = question_stats["q_confusion_mean"].isna()
        if missing_confusion.sum() > 0:
            print(f"  {missing_confusion.sum():,} questions without KT4 data, using fallback...")
            # Fallback: 정답률이 중간값(0.5)에 가까울수록 헷갈림
            question_stats.loc[missing_confusion, "confusion_proxy"] = np.abs(
                question_stats.loc[missing_confusion, "correct_rate"] - 0.5
            )
            question_stats.loc[missing_confusion, "q_confusion_mean"] = (
                1.0 - question_stats.loc[missing_confusion, "confusion_proxy"]
            )
        
        print(f"  Calculated confusion for {question_stats['q_confusion_mean'].notna().sum():,} questions")
    else:
        print("  KT4 data not available, using fallback confusion calculation...")
        # Fallback: 정답률이 중간값(0.5)에 가까울수록 헷갈림
        question_stats["confusion_proxy"] = np.abs(question_stats["correct_rate"] - 0.5)
        question_stats["q_confusion_mean"] = 1.0 - question_stats["confusion_proxy"]
    
    # NaN 처리
    question_stats["q_confusion_mean"] = question_stats["q_confusion_mean"].fillna(0.5)
    
    # 최소 시도 횟수 필터링
    min_attempts = 10
    question_stats = question_stats[question_stats["n_attempts"] >= min_attempts].copy()
    
    print(f"  Valid questions (>= {min_attempts} attempts): {len(question_stats):,}")
    
    return question_stats


def assign_quadrant_ctt(row, x_cut, y_cut):
    """CTT: correct_rate(높을수록 쉬움) × q_confusion_mean. median cut 기준."""
    if row["correct_rate"] >= x_cut and row["q_confusion_mean"] >= y_cut:
        return "Q1_easy_highConf"
    if row["correct_rate"] < x_cut and row["q_confusion_mean"] >= y_cut:
        return "Q2_hard_highConf"
    if row["correct_rate"] < x_cut and row["q_confusion_mean"] < y_cut:
        return "Q3_hard_lowConf"
    return "Q4_easy_lowConf"


def build_quadrant_mapping(question_stats):
    """문항을 사분면으로 분류"""
    print("\nBuilding quadrant mapping...")
    
    x_cut = question_stats["correct_rate"].median()
    y_cut = question_stats["q_confusion_mean"].median()
    
    print(f"  Median cut points: correct_rate={x_cut:.3f}, q_confusion_mean={y_cut:.3f}")
    
    question_stats["quadrant"] = question_stats.apply(
        lambda r: assign_quadrant_ctt(r, x_cut, y_cut), axis=1
    )
    
    quadrant_counts = question_stats["quadrant"].value_counts()
    print("\n  Quadrant distribution:")
    for q in QUADRANT_ORDER:
        count = quadrant_counts.get(q, 0)
        print(f"    {QUADRANT_LABELS_KR.get(q, q)}: {count:,} questions")
    
    return question_stats[["question_id", "quadrant", "correct_rate", "q_confusion_mean"]]


def calculate_user_quadrant_errors(kt1_df, quadrant_mapping, clustered_df):
    """유저별 사분면별 오답률 계산"""
    print("\nCalculating user-level quadrant errors...")
    
    # KT1 데이터에 quadrant 정보 추가
    kt1_with_quadrant = kt1_df.merge(
        quadrant_mapping[["question_id", "quadrant"]],
        on="question_id",
        how="inner"
    )
    
    # 클러스터 정보 추가
    kt1_with_quadrant = kt1_with_quadrant.merge(
        clustered_df[["user_id", "cluster_4", "theta", "accuracy"]],
        on="user_id",
        how="inner"
    )
    
    # 오답률 계산 (1 - correct)
    kt1_with_quadrant["error_rate"] = 1.0 - kt1_with_quadrant["correct"]
    
    # 유저별 사분면별 오답률
    user_quadrant_stats = kt1_with_quadrant.groupby(["user_id", "quadrant"], as_index=False).agg({
        "error_rate": "mean",
        "correct": "count"
    })
    user_quadrant_stats.columns = ["user_id", "quadrant", "error_rate", "n_attempts"]
    
    # 피벗 테이블로 변환
    user_quadrant_wide = user_quadrant_stats.pivot(
        index="user_id",
        columns="quadrant",
        values="error_rate"
    ).reindex(columns=QUADRANT_ORDER)
    
    user_quadrant_wide.columns = [f"{q}_error_rate" for q in QUADRANT_ORDER]
    
    # 시도 횟수도 피벗
    n_attempts_wide = user_quadrant_stats.pivot(
        index="user_id",
        columns="quadrant",
        values="n_attempts"
    ).reindex(columns=QUADRANT_ORDER)
    n_attempts_wide.columns = [f"{q}_n" for q in QUADRANT_ORDER]
    
    user_quadrant_wide = user_quadrant_wide.join(n_attempts_wide)
    
    # 클러스터, theta, accuracy 정보 추가
    user_meta = clustered_df[["user_id", "cluster_4", "theta", "accuracy"]].drop_duplicates()
    user_quadrant_wide = user_quadrant_wide.reset_index().merge(
        user_meta,
        on="user_id",
        how="left"
    )
    
    # dominant wrong quadrant 계산
    err_cols = [f"{q}_error_rate" for q in QUADRANT_ORDER]
    user_quadrant_wide["dominant_wrong_quadrant"] = user_quadrant_wide[err_cols].idxmax(axis=1)
    user_quadrant_wide["dominant_wrong_quadrant"] = user_quadrant_wide["dominant_wrong_quadrant"].str.replace("_error_rate", "", regex=False)
    
    print(f"  Processed {len(user_quadrant_wide):,} users")
    
    return user_quadrant_wide


def calculate_cluster_quadrant_summary(user_quadrant_wide):
    """군집별 사분면 평균 오답률 및 유의성 검정"""
    print("\nCalculating cluster-level quadrant summary...")
    
    err_cols = [f"{q}_error_rate" for q in QUADRANT_ORDER]
    
    # 군집별 평균, 표준오차, 개수
    cluster_summary = user_quadrant_wide.groupby("cluster_4", as_index=True)[err_cols].agg(["mean", "sem", "count"])
    
    # 컬럼명 평탄화
    flat_cols = []
    for q in QUADRANT_ORDER:
        flat_cols.extend([f"{q}_mean", f"{q}_sem", f"{q}_n"])
    cluster_summary.columns = flat_cols
    
    cluster_summary = cluster_summary.reset_index()
    
    # Kruskal-Wallis 검정으로 군집 간 유의성 검정
    kruskal_p_values = []
    for q in QUADRANT_ORDER:
        col = f"{q}_error_rate"
        groups = [
            user_quadrant_wide.loc[user_quadrant_wide["cluster_4"] == c, col].dropna().values
            for c in sorted(user_quadrant_wide["cluster_4"].unique())
        ]
        groups = [g for g in groups if len(g) >= 3]
        
        if len(groups) >= 2:
            try:
                stat, p = kruskal(*groups)
                kruskal_p_values.append(p)
            except:
                kruskal_p_values.append(np.nan)
        else:
            kruskal_p_values.append(np.nan)
    
    for i, q in enumerate(QUADRANT_ORDER):
        cluster_summary[f"{q}_kruskal_p"] = kruskal_p_values[i]
    
    print("\n  Cluster summary:")
    mean_cols = [f"{q}_mean" for q in QUADRANT_ORDER]
    print(cluster_summary[["cluster_4"] + mean_cols].round(3))
    
    return cluster_summary


def analyze_correlation(user_quadrant_wide):
    """상관관계 분석 (theta, accuracy, Q1~Q4 error_rate 간)"""
    print("\nAnalyzing correlations...")
    
    num_cols = ["theta", "accuracy"] + [f"{q}_error_rate" for q in QUADRANT_ORDER]
    df = user_quadrant_wide[num_cols].dropna()
    
    if len(df) < 10:
        print("  Insufficient data for correlation analysis")
        return pd.DataFrame(), pd.DataFrame()
    
    # 모든 쌍에 대한 상관계수 계산
    corrs = []
    for i, a in enumerate(num_cols):
        for j, b in enumerate(num_cols):
            if i >= j:
                continue
            pair = df[[a, b]].dropna()
            if len(pair) < 10:
                corrs.append({
                    "var1": a,
                    "var2": b,
                    "correlation": np.nan,
                    "p_value": np.nan,
                    "significant": False
                })
                continue
            
            try:
                r, p = pearsonr(pair[a], pair[b])
                corrs.append({
                    "var1": a,
                    "var2": b,
                    "correlation": float(r),
                    "p_value": float(p),
                    "significant": float(p) < ALPHA
                })
            except:
                corrs.append({
                    "var1": a,
                    "var2": b,
                    "correlation": np.nan,
                    "p_value": np.nan,
                    "significant": False
                })
    
    all_corr = pd.DataFrame(corrs).sort_values("p_value").reset_index(drop=True)
    sig_corr = all_corr[all_corr["significant"]].reset_index(drop=True)
    
    print(f"  Total correlation pairs: {len(all_corr)}")
    print(f"  Significant pairs (p < {ALPHA}): {len(sig_corr)}")
    
    if len(sig_corr) > 0:
        print("\n  Top significant correlations:")
        print(sig_corr.head(10)[["var1", "var2", "correlation", "p_value"]].to_string(index=False))
    
    return all_corr, sig_corr


def visualize_results(user_quadrant_wide, cluster_summary, all_corr, sig_corr):
    """결과 시각화"""
    Q2_CODE = find_project_root()
    OUT = Q2_CODE / "outputs"
    PLOTS = OUT / "plots"
    PLOTS.mkdir(parents=True, exist_ok=True)
    
    plt.rcParams["font.family"] = "DejaVu Sans"
    
    # 1. 군집 × 사분면 평균 오답률 히트맵
    fig, ax = plt.subplots(figsize=(8, 4))
    mean_cols = [f"{q}_mean" for q in QUADRANT_ORDER]
    p_cols = [f"{q}_kruskal_p" for q in QUADRANT_ORDER]
    
    mat = cluster_summary[mean_cols].values.T
    labels = [QUADRANT_LABELS_EN.get(q, q) for q in QUADRANT_ORDER]
    
    sns.heatmap(
        mat,
        xticklabels=[f"Cluster {int(c)}" for c in cluster_summary["cluster_4"]],
        yticklabels=labels,
        annot=True,
        fmt=".3f",
        cmap="YlOrRd",
        ax=ax,
        cbar_kws={"label": "Mean error rate"},
    )
    
    # 유의한 quadrant에 * 표시
    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            p = cluster_summary.iloc[j][p_cols[i]]
            if isinstance(p, (int, float)) and p < ALPHA and not np.isnan(p):
                ax.text(j + 0.5, i + 0.8, "*", ha="center", fontsize=14, color="black", fontweight="bold")
    
    ax.set_title(f"Cluster × Quadrant: Mean Error Rate (Kruskal-Wallis * p<{ALPHA})")
    plt.tight_layout()
    plt.savefig(PLOTS / "cluster_quadrant_heatmap.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 2. 군집별 사분면 오답률 막대그래프
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(cluster_summary))
    w = 0.2
    
    for i, q in enumerate(QUADRANT_ORDER):
        ax.bar(
            x + i * w,
            cluster_summary[f"{q}_mean"],
            width=w,
            label=QUADRANT_LABELS_EN.get(q, q),
            color=COLOR_MAP.get(q, "gray")
        )
    
    ax.set_xticks(x + w * 1.5)
    ax.set_xticklabels([f"Cluster {int(c)}" for c in cluster_summary["cluster_4"]])
    ax.set_ylabel("Mean error rate")
    ax.set_title("Mean Error Rate by Student Cluster and Question Quadrant")
    ax.legend(loc="upper right", fontsize=8)
    plt.tight_layout()
    plt.savefig(PLOTS / "cluster_quadrant_bars.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    # 3. 상관관계 히트맵
    if len(all_corr) > 0:
        num_cols = ["theta", "accuracy"] + [f"{q}_error_rate" for q in QUADRANT_ORDER]
        df = user_quadrant_wide[num_cols].dropna()
        
        if len(df) >= 10:
            corr_mat = df.corr()
            
            # p-value 행렬 계산
            p_mat = np.ones((len(num_cols), len(num_cols)))
            for i in range(len(num_cols)):
                for j in range(len(num_cols)):
                    if i != j:
                        try:
                            _, p = pearsonr(df[num_cols[i]], df[num_cols[j]])
                            p_mat[i, j] = float(p)
                        except:
                            pass
            
            # 유의성 마크
            def sig_mark(p):
                if p < ALPHA:
                    return "*"
                if p < 0.1:
                    return "°"
                return ""
            
            annot_mat = np.array([
                [
                    f"{corr_mat.iloc[i, j]:.2f}" + sig_mark(p_mat[i, j]) if i > j else ""
                    for j in range(len(num_cols))
                ]
                for i in range(len(num_cols))
            ])
            
            fig, ax = plt.subplots(figsize=(8, 6))
            mask_upper = np.triu(np.ones_like(corr_mat.values, dtype=bool), k=1)
            sns.heatmap(
                corr_mat,
                mask=mask_upper,
                annot=annot_mat,
                fmt="",
                center=0,
                cmap="RdBu_r",
                vmin=-0.5,
                vmax=0.5,
                ax=ax,
            )
            ax.set_title(f"Correlation (* p<{ALPHA}, ° p<0.1)")
            plt.tight_layout()
            plt.savefig(PLOTS / "correlation_heatmap.png", dpi=150, bbox_inches="tight")
            plt.close()
    
    # 4. Dominant wrong quadrant by cluster
    fig, ax = plt.subplots(figsize=(10, 5))
    ct = pd.crosstab(
        user_quadrant_wide["cluster_4"],
        user_quadrant_wide["dominant_wrong_quadrant"],
        normalize="index"
    )
    ct = ct.reindex(columns=QUADRANT_ORDER).fillna(0)
    ct.plot(
        kind="bar",
        stacked=False,
        ax=ax,
        color=[COLOR_MAP.get(q, "gray") for q in QUADRANT_ORDER]
    )
    ax.set_ylabel("Proportion")
    ax.set_xlabel("Cluster")
    ax.set_title("Proportion of Dominant Wrong Quadrant by Cluster")
    ax.legend(title="Quadrant", labels=[QUADRANT_LABELS_EN.get(q, q) for q in QUADRANT_ORDER])
    ax.set_xticklabels([f"Cluster {int(c)}" for c in ct.index], rotation=0)
    plt.tight_layout()
    plt.savefig(PLOTS / "dominant_wrong_by_cluster.png", dpi=150, bbox_inches="tight")
    plt.close()
    
    print(f"\nVisualizations saved to: {PLOTS}")


def save_results(user_quadrant_wide, cluster_summary, quadrant_mapping, all_corr, sig_corr):
    """결과를 CSV로 저장"""
    Q2_CODE = find_project_root()
    OUT = Q2_CODE / "outputs"
    
    # 유저별 사분면 오답률
    user_path = OUT / "user_quadrant_errors.csv"
    user_quadrant_wide.to_csv(user_path, index=False, encoding="utf-8-sig")
    print(f"\nUser quadrant errors saved to: {user_path}")
    
    # 군집별 사분면 요약
    cluster_path = OUT / "cluster_quadrant_summary.csv"
    cluster_summary.to_csv(cluster_path, index=False, encoding="utf-8-sig")
    print(f"Cluster quadrant summary saved to: {cluster_path}")
    
    # 문항 사분면 매핑
    quadrant_path = OUT / "question_quadrant_mapping.csv"
    quadrant_mapping.to_csv(quadrant_path, index=False, encoding="utf-8-sig")
    print(f"Question quadrant mapping saved to: {quadrant_path}")
    
    # 상관관계
    if len(all_corr) > 0:
        all_corr_path = OUT / "all_correlations.csv"
        all_corr.to_csv(all_corr_path, index=False, encoding="utf-8-sig")
        print(f"All correlations saved to: {all_corr_path}")
        
        if len(sig_corr) > 0:
            sig_corr_path = OUT / "significant_correlations.csv"
            sig_corr.to_csv(sig_corr_path, index=False, encoding="utf-8-sig")
            print(f"Significant correlations saved to: {sig_corr_path}")


def main():
    """메인 실행 함수"""
    print("="*60)
    print("Cluster-wise Question Error Rate Analysis (Quadrant-based)")
    print("="*60)
    
    # 1. 클러스터링된 학생 데이터 로드 (한 번만)
    clustered_df = load_clustered_data()
    user_ids = clustered_df["user_id"].astype(str).tolist()
    
    # 2. KT1 데이터 로드 (user_ids 전달하여 중복 로드 방지)
    kt1_df = load_kt1_data(user_ids=user_ids)
    
    # 3. KT4 데이터 로드 (user_ids 전달하여 중복 로드 방지)
    kt4_df = load_kt4_data(user_ids=user_ids)
    
    # 4. 문항별 통계 계산 (KT4 데이터 사용)
    question_stats = calculate_question_statistics(kt1_df, kt4_df)
    
    # 5. 문항 사분면 매핑 생성
    quadrant_mapping = build_quadrant_mapping(question_stats)
    
    # 6. 유저별 사분면 오답률 계산
    user_quadrant_wide = calculate_user_quadrant_errors(kt1_df, quadrant_mapping, clustered_df)
    
    # 7. 군집별 사분면 요약 및 유의성 검정
    cluster_summary = calculate_cluster_quadrant_summary(user_quadrant_wide)
    
    # 8. 상관관계 분석
    all_corr, sig_corr = analyze_correlation(user_quadrant_wide)
    
    # 9. 시각화
    visualize_results(user_quadrant_wide, cluster_summary, all_corr, sig_corr)
    
    # 10. 결과 저장
    save_results(user_quadrant_wide, cluster_summary, quadrant_mapping, all_corr, sig_corr)
    
    print("\n" + "="*60)
    print("Analysis completed successfully!")
    print("="*60)
    
    return user_quadrant_wide, cluster_summary, quadrant_mapping, all_corr, sig_corr


if __name__ == "__main__":
    user_quadrant_wide, cluster_summary, quadrant_mapping, all_corr, sig_corr = main()
