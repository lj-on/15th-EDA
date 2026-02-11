"""
결측치에 최저점(0.1) 부여 (특수 케이스 제외)
- 특수 케이스: 전체 레이스가 비정상적인 경우 → 결측 유지
- 일반 케이스: 개별 드라이버 문제 (DNF/예선불참) → 최저점 0.1 부여
"""

import pandas as pd
import numpy as np

print("=" * 80)
print("결측치에 최저점(0.1) 부여 (특수 케이스 제외)")
print("=" * 80)

# 데이터 로드
df = pd.read_csv('data/main/driver_performance_metrics.csv')

print(f"\n원본 데이터: {len(df):,}개 레코드")
print(f"QP_Score 결측: {df['QP_Score'].isna().sum()}개")
print(f"RP_Score 결측: {df['RP_Score'].isna().sum()}개")
print(f"Overall_Pace_Score 결측: {df['Overall_Pace_Score'].isna().sum()}개")

# 특수 케이스 정의
SPECIAL_CASES = [
    # (Year, Round, EventName, 문제 유형)
    (2021, 12, 'Belgian Grand Prix', 'RP'),      # 폭우로 2랩만 진행
    (2022, 21, 'São Paulo Grand Prix', 'RP'),    # 스프린트 포맷 + 데이터 문제
    (2024, 21, 'São Paulo Grand Prix', 'QP')     # 예선 데이터 전체 누락
]

print("\n" + "=" * 80)
print("특수 케이스 (결측 유지)")
print("=" * 80)

special_mask_qp = pd.Series([False] * len(df))
special_mask_rp = pd.Series([False] * len(df))

for year, round_num, event_name, issue_type in SPECIAL_CASES:
    mask = (df['Year'] == year) & (df['Round'] == round_num) & (df['EventName'] == event_name)
    count = mask.sum()
    
    if issue_type == 'QP':
        special_mask_qp |= mask
        print(f"{year} R{round_num:2d} {event_name:30s} - QP 결측 유지 ({count}명)")
    elif issue_type == 'RP':
        special_mask_rp |= mask
        print(f"{year} R{round_num:2d} {event_name:30s} - RP 결측 유지 ({count}명)")

print(f"\n특수 케이스 제외 대상: QP {special_mask_qp.sum()}개, RP {special_mask_rp.sum()}개")

# 결측치 처리
df_filled = df.copy()

# 1. QP_Score 처리: 특수 케이스 아니면서 QP_Score가 결측인 경우 → 0.1
qp_fill_mask = df_filled['QP_Score'].isna() & ~special_mask_qp
qp_fill_count = qp_fill_mask.sum()
df_filled.loc[qp_fill_mask, 'QP_Score'] = 0.1

print(f"\n✓ QP_Score에 0.1 부여 (예선 불참): {qp_fill_count}개")

# 2. RP_Score 처리: 특수 케이스 아니면서 RP_Score가 결측인 경우 → 0.1  
rp_fill_mask = df_filled['RP_Score'].isna() & ~special_mask_rp
rp_fill_count = rp_fill_mask.sum()
df_filled.loc[rp_fill_mask, 'RP_Score'] = 0.1

print(f"✓ RP_Score에 0.1 부여 (DNF): {rp_fill_count}개")

# 3. Overall_Pace_Score 재계산
def calculate_overall_score(row):
    qp = row['QP_Score']
    rp = row['RP_Score']
    
    if pd.notna(qp) and pd.notna(rp):
        return (qp + rp) / 2
    elif pd.notna(qp):
        return qp
    elif pd.notna(rp):
        return rp
    else:
        return np.nan

df_filled['Overall_Pace_Score'] = df_filled.apply(calculate_overall_score, axis=1)

print(f"✓ Overall_Pace_Score 재계산 완료")

# 결과 통계
print("\n" + "=" * 80)
print("처리 후 결측치 현황")
print("=" * 80)

print(f"\nQP_Score 결측: {df_filled['QP_Score'].isna().sum()}개")
print(f"RP_Score 결측: {df_filled['RP_Score'].isna().sum()}개")
print(f"Overall_Pace_Score 결측: {df_filled['Overall_Pace_Score'].isna().sum()}개")

# Score 통계
print("\n" + "=" * 80)
print("Score 통계")
print("=" * 80)

print(f"\nQP_Score:")
print(f"  평균: {df_filled['QP_Score'].mean():.2f}")
print(f"  최소: {df_filled['QP_Score'].min():.2f}")
print(f"  최대: {df_filled['QP_Score'].max():.2f}")
print(f"  0.1 (예선 불참): {(df_filled['QP_Score'] == 0.1).sum()}개")

print(f"\nRP_Score:")
print(f"  평균: {df_filled['RP_Score'].mean():.2f}")
print(f"  최소: {df_filled['RP_Score'].min():.2f}")
print(f"  최대: {df_filled['RP_Score'].max():.2f}")
print(f"  0.1 (DNF): {(df_filled['RP_Score'] == 0.1).sum()}개")

print(f"\nOverall_Pace_Score:")
print(f"  평균: {df_filled['Overall_Pace_Score'].mean():.2f}")
print(f"  최소: {df_filled['Overall_Pace_Score'].min():.2f}")
print(f"  최대: {df_filled['Overall_Pace_Score'].max():.2f}")

# 저장
output_path = 'data/main/driver_performance_metrics.csv'
df_filled.to_csv(output_path, index=False)

print("\n" + "=" * 80)
print(f"✓ 저장 완료: {output_path}")
print("=" * 80)

# driver_pace.csv도 업데이트
print("\ndriver_pace.csv 업데이트 중...")

selected_columns = [
    'Driver',
    'Year',
    'Round',
    'EventName',
    'Team',
    'Qualifying_Pace',
    'Race_Pace',
    'QP_Score',
    'RP_Score',
    'Overall_Pace_Score'
]

pace_df = df_filled[selected_columns].copy()
pace_df.columns = [
    'driver',
    'year',
    'round',
    'eventname',
    'team',
    'qualifying_pace',
    'race_pace',
    'qp_score',
    'rp_score',
    'overall_pace_score'
]

pace_output_path = 'data/main/driver_pace.csv'
pace_df.to_csv(pace_output_path, index=False)

print(f"✓ 저장 완료: {pace_output_path}")

# 최종 검증
print("\n" + "=" * 80)
print("최종 검증: DNF vs 완주 구별")
print("=" * 80)

rp_01 = pace_df[pace_df['rp_score'] == 0.1]
rp_1 = pace_df[pace_df['rp_score'] == 1.0]

print(f"\nRP_Score = 0.1 (DNF): {len(rp_01)}개")
print(f"  Race_Pace 없음: {rp_01['race_pace'].isna().sum()}개 ✓")

print(f"\nRP_Score = 1.0 (완주, 가장 느림): {len(rp_1)}개")
print(f"  Race_Pace 있음: {rp_1['race_pace'].notna().sum()}개 ✓")

if len(rp_1) > 0:
    print(f"\n완주 최저 점수 샘플 (처음 5개):")
    print(rp_1[['driver', 'year', 'round', 'eventname', 'race_pace', 'rp_score']].head(5).to_string(index=False))

print("\n" + "=" * 80)
print("완료!")
print("=" * 80)
