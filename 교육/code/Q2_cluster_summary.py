"""
학생 클러스터별 요약(Summary) 생성

클러스터링 결과(student_features_clustered.csv)를 읽어
각 군집별 인원 수와 변수별 한 수치(평균)만 담은 요약 파일을 생성합니다.
"""

from pathlib import Path
import pandas as pd


def find_project_root():
    _cwd = Path.cwd()
    if (_cwd / "theta_feature_vis.ipynb").exists():
        return _cwd
    if (_cwd / "Q2_code" / "theta_feature_vis.ipynb").exists():
        return _cwd / "Q2_code"
    return _cwd


def get_summary_columns(df):
    """요약에 사용할 수치형 컬럼 (실제 존재하는 것만)"""
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


def build_cluster_summary(df):
    """클러스터별 요약: 변수당 한 수치(평균)만."""
    num_cols = get_summary_columns(df)
    if not num_cols:
        return pd.DataFrame()

    out = df.groupby("cluster_4", as_index=False).agg(
        n_students=("cluster_4", "count"),
        **{c: (c, "mean") for c in num_cols}
    )
    out = out.round(4)
    out = out.rename(columns={"cluster_4": "cluster"})
    return out


def main():
    Q2_CODE = find_project_root()
    OUT = Q2_CODE / "outputs"

    print("Loading clustered student data...")
    df = load_clustered_data()
    print(f"Loaded {len(df):,} students in {df['cluster_4'].nunique()} clusters.")

    summary = build_cluster_summary(df)

    if summary.empty:
        print("No numeric columns found for summary.")
        return

    print("\nCluster summary (each variable = mean):")
    print(summary.to_string(index=False))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "cluster_summary.csv"
    summary.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()
