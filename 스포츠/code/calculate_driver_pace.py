"""
드라이버 실력 지표 계산: Qualifying Pace & Race Pace
"""

import pandas as pd
import numpy as np
from pathlib import Path

def calculate_qualifying_pace(df):
    """
    Qualifying Pace 계산
    
    QP_{i,r,t} = Q_{i,r,t}^{best} - median(Q_{r,t})
    
    Args:
        df: merged_f1_data.csv 데이터프레임
    
    Returns:
        DataFrame with columns: Driver, Year, Round, EventName, Team, Qualifying_Pace, 
                                 BestQualifyingTime, FieldMedian
    """
    print("\n=== Qualifying Pace 계산 시작 ===")
    
    # 각 드라이버의 레이스별 퀄리파잉 데이터 추출 (중복 제거)
    quali_cols = ['Driver', 'Year', 'Round', 'EventName', 'Team',
                  'Q1_Time', 'Q2_Time', 'Q3_Time',
                  'QualifyingPosition', 'GridPosition']
    
    quali_df = df[quali_cols].drop_duplicates(subset=['Driver', 'Year', 'Round', 'EventName'])
    
    # 베스트 랩 선택: Q1, Q2, Q3 중 가장 빠른(최소) 랩타임 선택
    # NaN은 무시하고 있는 값들 중에서 최소값 선택
    quali_df['BestQualifyingTime'] = quali_df[['Q1_Time', 'Q2_Time', 'Q3_Time']].min(axis=1)
    
    # NaN 제거 (퀄리파잉 기록이 없는 경우)
    quali_df = quali_df.dropna(subset=['BestQualifyingTime'])
    
    print(f"퀄리파잉 기록 수: {len(quali_df)}")
    
    # 각 레이스별 필드 중앙값 계산
    race_median = quali_df.groupby(['Year', 'Round', 'EventName'])['BestQualifyingTime'].median().reset_index()
    race_median.rename(columns={'BestQualifyingTime': 'FieldMedian'}, inplace=True)
    
    # 병합
    quali_pace = quali_df.merge(race_median, on=['Year', 'Round', 'EventName'])
    
    # Qualifying Pace 계산 (음수 = 빠름, 양수 = 느림)
    quali_pace['Qualifying_Pace'] = quali_pace['BestQualifyingTime'] - quali_pace['FieldMedian']
    
    # 정리
    result = quali_pace[[
        'Driver', 'Year', 'Round', 'EventName', 'Team',
        'Qualifying_Pace', 'BestQualifyingTime', 'FieldMedian',
        'QualifyingPosition', 'GridPosition'
    ]].copy()
    
    print(f"계산 완료: {len(result)} 레코드")
    print(f"평균 Qualifying Pace: {result['Qualifying_Pace'].mean():.3f}초")
    print(f"표준편차: {result['Qualifying_Pace'].std():.3f}초")
    
    return result


def calculate_race_pace(df):
    """
    Race Pace 계산
    
    RP_{i,r,t} = sum_{s=1}^{S} w_{s} * median(L_{s}^{clean})
    
    Args:
        df: merged_f1_data.csv 데이터프레임
    
    Returns:
        DataFrame with columns: Driver, Year, Round, EventName, Team, Race_Pace,
                                 TotalCleanLaps, NumStints
    """
    print("\n=== Race Pace 계산 시작 ===")
    
    # 1. 클린랩 필터링
    race_df = df.copy()
    
    # 기본 필터: Deleted=False, IsAccurate=True
    clean_laps = race_df[
        (race_df['Deleted'] == False) &
        (race_df['IsAccurate'] == True) &
        (race_df['LapTimeSeconds'].notna())
    ].copy()
    
    print(f"기본 필터 후: {len(clean_laps)} 랩")
    
    # TrackStatus 필터: '1'이 포함된 경우 (그린 플래그)
    # TrackStatus가 '1', '12', '14' 등 1이 포함되면 그린 플래그 상황
    clean_laps['TrackStatus_str'] = clean_laps['TrackStatus'].astype(str)
    clean_laps = clean_laps[clean_laps['TrackStatus_str'].str.startswith('1')]
    
    print(f"TrackStatus 필터 후: {len(clean_laps)} 랩")
    
    # 피트스톱 랩 제외
    clean_laps = clean_laps[
        (clean_laps['PitInTime'].isna()) &
        (clean_laps['PitOutTime'].isna())
    ]
    
    print(f"피트스톱 제외 후: {len(clean_laps)} 랩")
    
    # 이상치 제거: IQR (Interquartile Range) 방법 사용
    # 각 레이스별로 IQR 계산하여 이상치 제거
    def remove_outliers_iqr(group):
        """IQR 방법으로 이상치 제거"""
        lap_times = group['LapTimeSeconds']
        
        Q1 = lap_times.quantile(0.25)
        Q3 = lap_times.quantile(0.75)
        IQR = Q3 - Q1
        
        # 상한선: Q3 + 1.5 × IQR
        # 하한선: Q1 - 1.5 × IQR (랩타임은 음수가 없으므로 실질적으로 불필요)
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # 범위 내의 랩만 반환
        mask = (lap_times >= lower_bound) & (lap_times <= upper_bound)
        return group[mask]
    
    # 레이스별로 그룹화하여 IQR 필터링 적용
    clean_laps = clean_laps.groupby(['Year', 'Round', 'EventName'], group_keys=False).apply(
        remove_outliers_iqr
    )
    
    print(f"이상치 제거 후 (IQR 방법): {len(clean_laps)} 랩")
    
    # [주석 처리] 기존 방법: Q1 × 1.20 기준
    # quali_baseline = df.groupby(['Year', 'Round', 'EventName'])['Quali_Q1Seconds'].min().reset_index()
    # quali_baseline.rename(columns={'Quali_Q1Seconds': 'Q1_Baseline'}, inplace=True)
    # clean_laps = clean_laps.merge(quali_baseline, on=['Year', 'Round', 'EventName'], how='left')
    # clean_laps = clean_laps[
    #     (clean_laps['LapTimeSeconds'] <= clean_laps['Q1_Baseline'] * 1.20) |
    #     (clean_laps['Q1_Baseline'].isna())
    # ]
    # print(f"이상치 제거 후 (Q1 × 1.20): {len(clean_laps)} 랩")
    
    if len(clean_laps) == 0:
        print("경고: 클린랩이 없습니다!")
        return pd.DataFrame()
    
    # 2. 각 레이스의 총 랩 수 계산 (연료 보정을 위해)
    # TotalLaps 컬럼이 이미 존재하므로 사용
    race_total_laps = df[['Year', 'Round', 'EventName', 'TotalLaps']].drop_duplicates()
    race_total_laps.rename(columns={'TotalLaps': 'RaceTotalLaps'}, inplace=True)
    clean_laps = clean_laps.merge(race_total_laps, on=['Year', 'Round', 'EventName'], how='left')
    
    # 3. 연료 보정 계수 계산
    # 연료는 선형적으로 감소하므로, 각 랩의 상대적 무게를 계산
    # Fuel_Factor = 1 - (LapNumber - 1) / RaceTotalLaps
    # LapNumber 1 (시작) = 1.0 (100% 연료), LapNumber = RaceTotalLaps (끝) ≈ 0.0 (0% 연료)
    clean_laps['FuelFactor'] = 1 - (clean_laps['LapNumber'] - 1) / clean_laps['RaceTotalLaps']
    
    # 연료 보정 적용: 연료가 많을수록 차가 무거워 랩타임이 느려짐
    # 연료 1kg당 약 0.03초 영향, F1 차량 최대 연료 110kg 가정
    # 따라서 최대 영향 = 110kg * 0.03초/kg = 3.3초
    # 보정된 랩타임 = 실제 랩타임 - (연료 영향)
    # 연료 영향 = FuelFactor * 최대_연료_영향
    MAX_FUEL_EFFECT = 3.3  # 초
    clean_laps['FuelEffect'] = clean_laps['FuelFactor'] * MAX_FUEL_EFFECT
    clean_laps['CorrectedLapTime'] = clean_laps['LapTimeSeconds'] - clean_laps['FuelEffect']
    
    # 4. 스틴트별 평균 계산 (중앙값 → 평균으로 변경)
    stint_stats = clean_laps.groupby(
        ['Driver', 'Year', 'Round', 'EventName', 'Team', 'Stint']
    ).agg({
        'CorrectedLapTime': ['mean', 'count']  # median → mean
    }).reset_index()
    
    stint_stats.columns = ['Driver', 'Year', 'Round', 'EventName', 'Team', 'Stint', 
                           'StintMeanLapTime', 'StintLapCount']
    
    print(f"스틴트 수: {len(stint_stats)}")
    
    # 5. 드라이버별 레이스별 가중 평균 계산
    def weighted_average_pace(group):
        total_laps = group['StintLapCount'].sum()
        if total_laps == 0:
            return pd.Series({
                'Race_Pace': np.nan,
                'TotalCleanLaps': 0,
                'NumStints': 0
            })
        
        weights = group['StintLapCount'] / total_laps
        race_pace = (group['StintMeanLapTime'] * weights).sum()
        
        return pd.Series({
            'Race_Pace': race_pace,
            'TotalCleanLaps': total_laps,
            'NumStints': len(group)
        })
    
    race_pace = stint_stats.groupby(
        ['Driver', 'Year', 'Round', 'EventName', 'Team']
    ).apply(weighted_average_pace, include_groups=False).reset_index()
    
    # NaN 제거
    race_pace = race_pace.dropna(subset=['Race_Pace'])
    
    print(f"계산 완료: {len(race_pace)} 레코드")
    print(f"평균 Race Pace: {race_pace['Race_Pace'].mean():.3f}초")
    print(f"표준편차: {race_pace['Race_Pace'].std():.3f}초")
    print(f"평균 클린랩 수: {race_pace['TotalCleanLaps'].mean():.1f}")
    
    return race_pace


def calculate_pace_scores(performance_df):
    """
    Qualifying Pace와 Race Pace를 1~10 점수로 정규화
    
    - 트랙별(EventName)로 정규화하여 공정한 비교
    - 낮은 Pace 값 = 빠름 = 높은 점수
    - Min-Max 정규화 후 1~10 스케일 변환
    """
    print("\n=== Pace 점수 계산 (1~10 스케일) ===")
    
    df = performance_df.copy()
    
    # 1. Qualifying Pace Score 계산 (트랙별)
    def normalize_quali_pace(group):
        """트랙별 QP 정규화"""
        qp = group['Qualifying_Pace']
        
        # 유효한 값만
        valid_qp = qp.dropna()
        
        if len(valid_qp) < 2:
            # 데이터가 부족하면 기본값
            return pd.Series(5.0, index=group.index)
        
        qp_min = valid_qp.min()
        qp_max = valid_qp.max()
        
        if qp_max == qp_min:
            # 모두 같은 값이면 중간 점수
            return pd.Series(5.5, index=group.index)
        
        # Min-Max 정규화 후 역변환 (작을수록 좋으므로)
        # Score = 1 + 9 * (max - value) / (max - min)
        # 가장 빠른 값(min) = 10점, 가장 느린 값(max) = 1점
        normalized = 1 + 9 * (qp_max - qp) / (qp_max - qp_min)
        
        return normalized
    
    # 트랙별로 그룹화하여 정규화
    df['QP_Score'] = df.groupby(['Year', 'EventName'], group_keys=False).apply(
        normalize_quali_pace, include_groups=False
    )
    
    print(f"Qualifying Pace Score 계산 완료")
    print(f"  - 평균: {df['QP_Score'].mean():.2f}")
    print(f"  - 범위: {df['QP_Score'].min():.2f} ~ {df['QP_Score'].max():.2f}")
    
    # 2. Race Pace Score 계산 (트랙별)
    def normalize_race_pace(group):
        """트랙별 RP 정규화"""
        rp = group['Race_Pace']
        
        # 유효한 값만
        valid_rp = rp.dropna()
        
        if len(valid_rp) < 2:
            return pd.Series(5.0, index=group.index)
        
        rp_min = valid_rp.min()
        rp_max = valid_rp.max()
        
        if rp_max == rp_min:
            return pd.Series(5.5, index=group.index)
        
        # Min-Max 정규화 후 역변환 (작을수록 좋으므로)
        # 가장 빠른 값(min) = 10점, 가장 느린 값(max) = 1점
        normalized = 1 + 9 * (rp_max - rp) / (rp_max - rp_min)
        
        return normalized
    
    df['RP_Score'] = df.groupby(['Year', 'EventName'], group_keys=False).apply(
        normalize_race_pace, include_groups=False
    )
    
    print(f"Race Pace Score 계산 완료")
    print(f"  - 평균: {df['RP_Score'].mean():.2f}")
    print(f"  - 범위: {df['RP_Score'].min():.2f} ~ {df['RP_Score'].max():.2f}")
    
    # 3. 종합 점수 계산 (QP + RP 평균)
    # 둘 다 있는 경우에만 계산
    both_valid = (~df['QP_Score'].isna()) & (~df['RP_Score'].isna())
    df.loc[both_valid, 'Overall_Pace_Score'] = (
        df.loc[both_valid, 'QP_Score'] + df.loc[both_valid, 'RP_Score']
    ) / 2
    
    print(f"Overall Pace Score 계산 완료")
    print(f"  - 평균: {df['Overall_Pace_Score'].mean():.2f}")
    print(f"  - 범위: {df['Overall_Pace_Score'].min():.2f} ~ {df['Overall_Pace_Score'].max():.2f}")
    
    return df


def main():
    """메인 실행 함수"""
    print("=" * 70)
    print("드라이버 Pace 지표 계산")
    print("=" * 70)
    
    # 데이터 로드
    data_path = Path('data/main/f1_main_analysis.csv')
    print(f"\n데이터 로드: {data_path}")
    df = pd.read_csv(data_path, low_memory=False)
    print(f"전체 데이터: {len(df)} 행, {len(df.columns)} 열")
    print(f"연도 범위: {df['Year'].min()} - {df['Year'].max()}")
    print(f"레이스 수: {df.groupby(['Year', 'Round']).ngroups}")
    print(f"드라이버 수: {df['Driver'].nunique()}")
    
    # Qualifying Pace 계산
    quali_pace = calculate_qualifying_pace(df)
    
    # Race Pace 계산
    race_pace = calculate_race_pace(df)
    
    # 결합
    print("\n=== 데이터 결합 ===")
    performance = quali_pace.merge(
        race_pace[['Driver', 'Year', 'Round', 'EventName', 'Race_Pace', 'TotalCleanLaps', 'NumStints']],
        on=['Driver', 'Year', 'Round', 'EventName'],
        how='outer'
    )
    
    print(f"결합 후: {len(performance)} 레코드")
    print(f"Qualifying Pace만 있는 경우: {performance['Race_Pace'].isna().sum()}")
    print(f"Race Pace만 있는 경우: {performance['Qualifying_Pace'].isna().sum()}")
    print(f"둘 다 있는 경우: {(~performance['Race_Pace'].isna() & ~performance['Qualifying_Pace'].isna()).sum()}")
    
    # 점수 계산 (1~10 스케일)
    performance = calculate_pace_scores(performance)
    
    # 정렬
    performance = performance.sort_values(['Year', 'Round', 'Qualifying_Pace'])
    
    # 저장
    output_path = Path('data/main/driver_performance_metrics.csv')
    performance.to_csv(output_path, index=False)
    print(f"\n결과 저장: {output_path}")
    
    # 샘플 출력
    print("\n=== 결과 샘플 (2024 Bahrain GP 상위 5명) ===")
    sample = performance[
        (performance['Year'] == 2024) & 
        (performance['Round'] == 1)
    ].head(10)
    
    print(sample[[
        'Driver', 'Team', 'Qualifying_Pace', 'QP_Score', 
        'Race_Pace', 'RP_Score', 'Overall_Pace_Score'
    ]].to_string(index=False))
    
    # 통계 요약
    print("\n=== 전체 통계 ===")
    print("\nQualifying Pace 통계 (초):")
    print(performance['Qualifying_Pace'].describe())
    
    print("\nQualifying Pace Score 통계 (1~10):")
    print(performance['QP_Score'].describe())
    
    print("\nRace Pace 통계 (초):")
    print(performance['Race_Pace'].describe())
    
    print("\nRace Pace Score 통계 (1~10):")
    print(performance['RP_Score'].describe())
    
    print("\nOverall Pace Score 통계 (1~10):")
    print(performance['Overall_Pace_Score'].describe())
    
    # 상관관계
    if (~performance['Qualifying_Pace'].isna() & ~performance['Race_Pace'].isna()).sum() > 0:
        corr = performance[['Qualifying_Pace', 'Race_Pace']].corr().iloc[0, 1]
        print(f"\nQualifying Pace vs Race Pace 상관계수: {corr:.3f}")
    
    if (~performance['QP_Score'].isna() & ~performance['RP_Score'].isna()).sum() > 0:
        corr_score = performance[['QP_Score', 'RP_Score']].corr().iloc[0, 1]
        print(f"QP Score vs RP Score 상관계수: {corr_score:.3f}")
    
    # Top 10 드라이버 (Overall Score 기준)
    print("\n=== Top 10 드라이버 (Overall Pace Score 평균) ===")
    top_drivers = performance.groupby('Driver').agg({
        'QP_Score': 'mean',
        'RP_Score': 'mean',
        'Overall_Pace_Score': 'mean',
        'Team': 'first'
    }).sort_values('Overall_Pace_Score', ascending=False).head(10)
    print(top_drivers.round(2).to_string())
    
    print("\n" + "=" * 70)
    print("계산 완료!")
    print("=" * 70)
    
    return performance


if __name__ == '__main__':
    performance_df = main()
