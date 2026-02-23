"""
시대별 K-POP 가사 의미 변화 분석 (Semantic Shift Analysis)
=============================================================
전처리된 멜론 차트 데이터(1996-2025) 활용

워크플로우:
1. Data Loading: 전처리된 토큰 데이터 로드
2. Training: 시대별 독립 Word2Vec 모델 학습
3. Comparison: 핵심 키워드의 Nearest Neighbors 비교
4. Visualization: t-SNE 2차원 시각화
"""

import os
import glob
import ast
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots


# ============================================================
# Step 1. 전처리된 데이터 로드
# ============================================================

def load_processed_data(lyrics_dir='processed_lyrics'):
    """전처리된 Excel 파일들을 병합"""
    all_files = glob.glob(os.path.join(lyrics_dir, 'Processed_Melon_Chart_*.xlsx'))

    df_list = []
    for file in all_files:
        df = pd.read_excel(file)
        df_list.append(df)

    combined_df = pd.concat(df_list, ignore_index=True)

    print(f"✓ 총 {len(combined_df):,}개 로드")
    print(f"  - 연도 범위: {combined_df['Year'].min()} ~ {combined_df['Year'].max()}")
    print(f"  (중복 제거는 시대별로 수행)")

    return combined_df


def parse_tokens(token_str):
    """문자열 형태의 토큰 리스트를 파싱"""
    if pd.isna(token_str):
        return []
    try:
        # 문자열을 리스트로 변환
        tokens = ast.literal_eval(token_str)
        # 한글 토큰만 필터링 (영어/의성어 제거)
        korean_tokens = [t for t in tokens if is_valid_korean(t)]
        return korean_tokens
    except:
        return []


def is_valid_korean(word):
    """유효한 한글 단어인지 확인"""
    if not word or len(word) < 2:
        return False
    # 한글이 포함되어 있는지 확인
    has_korean = any('가' <= c <= '힣' for c in word)
    if not has_korean:
        return False
    # 의성어/반복어 필터링
    if is_onomatopoeia(word):
        return False
    # 불용어 필터링
    if word in STOPWORDS:
        return False
    return True


def is_onomatopoeia(word):
    """의성어/반복 어구 감지"""
    if len(word) < 2:
        return False
    # 동일 글자 반복 (바바바, 라라라)
    if len(set(word)) == 1:
        return True
    # 2글자 반복 (짠짠, 둠칫둠칫)
    if len(word) >= 4 and len(word) % 2 == 0:
        half = len(word) // 2
        if word[:half] == word[half:]:
            return True
    return False


# 불용어 리스트
STOPWORDS = {
    '하다', '되다', '있다', '없다', '같다', '보다', '주다', '오다', '가다',
    '이다', '아니다', '않다', '못하다', '싶다', '알다', '모르다',
    '그렇다', '어떻다', '이렇다', '저렇다', '이러다', '저러다' , '이번', '저번',
    '나', '너', '너희', '우리', '저', '그', '이것', '그것', '저것',
    '때', '것', '수', '듯', '더', '또', '다시', '너무', '정말', '진짜',
    '아이고', '훨씬', '전혀', '몹시', '매우', '존나', '미처', '완전', '완전히', '일단',
    '일단', '막상' , '그새', '아예', '어쨌든','감히','무지','실은','확실히','이후','너만이','그동안','무심코','왼손','거리다',
    '뒷자리','레드','놓이다','충분히','감히','허다','지나오다','진행','연속','수없이','며칠','하나같이','별로','정도','곤란','그동안',
    '떠다','깔다','적히다','부분','족하다','여태껏','신다','갈라놓다','등장','어떠','일루'
}


# ============================================================
# Step 2. 시대별 데이터 분할 및 모델 학습
# ============================================================

def slice_by_era(df):
    """데이터를 5개 시대로 분할 (기술/미디어 변화 기준, 시대별 중복 제거)"""
    era_ranges = [
        ('Era 1 (1996-2005)', 1996, 2005),     
        ('Era 2 (2006-2015)', 2006, 2015),     
        ('Era 3 (2016-2025)', 2016, 2025),  
    ]

    eras = {}
    print("\n✓ 시대별 데이터 분할 (시대 내 중복 제거):")

    for name, year_start, year_end in era_ranges:
        era_df = df[(df['Year'] >= year_start) & (df['Year'] <= year_end)]
        original = len(era_df)

        # 시대 내에서 중복 곡 제거
        if 'Song_ID' in era_df.columns:
            era_df = era_df.drop_duplicates(subset=['Song_ID'], keep='first')

        deduped = len(era_df)
        removed = original - deduped
        eras[name] = era_df
        print(f"  - {name}: {original:,} → {deduped:,}곡 (중복 {removed:,}개 제거)")

    return eras


def train_era_model(era_df, era_name, vector_size=100, window=4,
                    min_count=5, min_doc_count=5):
    """시대별 Word2Vec 모델 학습
    - min_count: 전체에서 최소 등장 횟수
    - min_doc_count: 최소 등장 곡 수
    """
    from collections import Counter

    print(f"\n[{era_name}] 모델 학습 시작...")

    # 전처리된 토큰 파싱
    tokenized_data = []
    for tokens_str in era_df['Processed_Tokens']:
        tokens = parse_tokens(tokens_str)
        if len(tokens) >= 3:
            tokenized_data.append(tokens)

    if len(tokenized_data) < 10:
        print(f"  ⚠ 데이터 부족 (토큰화된 곡: {len(tokenized_data)})")
        return None

    # 단어별 등장 곡 수(document frequency) 계산
    doc_freq = Counter()
    for tokens in tokenized_data:
        unique_tokens = set(tokens)  # 곡 내 중복 제거
        doc_freq.update(unique_tokens)

    # min_doc_count 미만인 단어 제거
    rare_words = {w for w, cnt in doc_freq.items() if cnt < min_doc_count}
    filtered_data = [
        [w for w in tokens if w not in rare_words]
        for tokens in tokenized_data
    ]
    filtered_data = [t for t in filtered_data if len(t) >= 3]

    before_vocab = len(set(w for tokens in tokenized_data for w in tokens))
    after_vocab = len(set(w for tokens in filtered_data for w in tokens))
    print(f"  - 곡 수 필터(≥{min_doc_count}곡): {before_vocab:,} → {after_vocab:,} 단어")

    # Word2Vec 학습
    model = Word2Vec(
        sentences=filtered_data,
        vector_size=vector_size,
        window=window,
        min_count=min_count,
        workers=4,
        sg=1,  # Skip-gram
        epochs=10
    )

    print(f"  ✓ 학습 완료! (어휘 수: {len(model.wv):,}개)")
    return model


def train_all_models(eras):
    """모든 시대별 모델 학습"""
    models = {}
    for name, data in eras.items():
        model = train_era_model(data, name)
        if model:
            models[name] = model
    return models


# ============================================================
# Step 3. 의미 변화 분석
# ============================================================

def analyze_semantic_shift(models, target_word, topn=15):
    """특정 단어의 시대별 의미 변화 분석"""
    print(f"\n{'='*60}")
    print(f"  '{target_word}'의 시대별 의미 변화 분석")
    print(f"{'='*60}")

    results = {}

    for name, model in models.items():
        try:
            similar_words = model.wv.most_similar(target_word, topn=topn)
            words_only = [w[0] for w in similar_words]

            results[name] = {
                'similar_words': similar_words,
                'words_list': words_only
            }

            print(f"\n[{name}]")
            print(f"  유사 단어: {', '.join(words_only[:10])}")

        except KeyError:
            print(f"\n[{name}] '{target_word}'가 이 시대 데이터에 없습니다.")
            results[name] = None

    return results


def find_word_sources(df, target_word, era_range=None, max_results=10):
    """특정 단어가 포함된 원문 가사 출처 확인"""
    print(f"\n{'='*60}")
    print(f"  '{target_word}' 단어 출처 분석")
    print(f"{'='*60}")

    if era_range:
        filtered_df = df[(df['Year'] >= era_range[0]) & (df['Year'] <= era_range[1])]
        print(f"  분석 범위: {era_range[0]} ~ {era_range[1]}년")
    else:
        filtered_df = df

    matches = []
    for idx, row in filtered_df.iterrows():
        tokens_str = str(row.get('Processed_Tokens', ''))
        if target_word in tokens_str:
            context = extract_context(str(row.get('Lyrics', '')), target_word)
            matches.append({
                'Year': row['Year'],
                'Title': row['Title'],
                'Artist': row['Artist'],
                'Context': context
            })

    # 연도별 빈도
    year_counts = {}
    for m in matches:
        year = m['Year']
        year_counts[year] = year_counts.get(year, 0) + 1

    print(f"\n  총 {len(matches)}곡에서 '{target_word}' 발견")

    if year_counts:
        print(f"\n  [연도별 빈도]")
        for year in sorted(year_counts.keys()):
            bar = '█' * min(year_counts[year], 30)
            print(f"  {year}: {bar} ({year_counts[year]}곡)")

    print(f"\n  [대표 곡 목록]")
    for i, m in enumerate(matches[:max_results]):
        print(f"  {i+1}. [{m['Year']}] {m['Artist']} - {m['Title']}")
        if m['Context']:
            print(f"      \"{m['Context'][:60]}...\"")

    return matches


def extract_context(lyrics, word, window=30):
    """단어 주변 문맥 추출"""
    if not lyrics:
        return ""
    idx = lyrics.find(word)
    if idx == -1:
        return ""
    start = max(0, idx - window)
    end = min(len(lyrics), idx + len(word) + window)
    return lyrics[start:end].replace('\n', ' ')


def deep_dive_word(models, df, target_word, era_name=None):
    """특정 단어 심층 분석"""
    print(f"\n{'#'*60}")
    print(f"  '{target_word}' 심층 분석")
    print(f"{'#'*60}")

    # 유사 단어 분석
    if era_name and era_name in models:
        model = models[era_name]
        try:
            similar = model.wv.most_similar(target_word, topn=15)
            print(f"\n  [{era_name}] '{target_word}'와 유사한 단어:")
            for w, score in similar:
                print(f"    - {w}: {score:.3f}")
        except KeyError:
            print(f"  '{target_word}'가 해당 시대에 없습니다.")
    else:
        for name, model in models.items():
            try:
                similar = model.wv.most_similar(target_word, topn=5)
                words_str = ', '.join([f"{w}({s:.2f})" for w, s in similar])
                print(f"\n  [{name}] {words_str}")
            except KeyError:
                print(f"\n  [{name}] 없음")

    # 출처 확인
    if era_name:
        import re
        years = re.findall(r'\d{4}', era_name)
        if len(years) == 2:
            find_word_sources(df, target_word, era_range=(int(years[0]), int(years[1])), max_results=5)
    else:
        find_word_sources(df, target_word, max_results=10)


# ============================================================
# Step 4. t-SNE 시각화
# ============================================================

def setup_korean_font():
    """한글 폰트 설정"""
    import platform
    system = platform.system()

    if system == 'Darwin':
        fonts = ['AppleGothic', 'Apple SD Gothic Neo']
    elif system == 'Windows':
        fonts = ['Malgun Gothic', 'NanumGothic']
    else:
        fonts = ['NanumGothic', 'UnDotum']

    for font in fonts:
        try:
            plt.rc('font', family=font)
            plt.rcParams['axes.unicode_minus'] = False
            return True
        except:
            continue
    return False


def plot_semantic_field(target_word, models, topn=8, save_path=None):
    """시대별 의미장 3D 시각화 (matplotlib 정적 이미지)"""
    from mpl_toolkits.mplot3d import Axes3D
    setup_korean_font()

    # 시대별 색상 팔레트
    era_colors = {
        'Era 1 (1996-2005)': '#E74C3C',
        'Era 2 (2006-2015)': '#F39C12',
        'Era 3 (2016-2025)': '#27AE60'
    }

    # 3D 플롯 생성 (흰색 배경)
    fig = plt.figure(figsize=(16, 12), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')

    # 모든 시대 데이터 수집
    all_data = []
    for era_name, model in models.items():
        color = era_colors.get(era_name, '#999999')
        try:
            similar_words = model.wv.most_similar(target_word, topn=topn)
            words = [target_word] + [w[0] for w in similar_words]  # 타겟 단어 포함
            scores = [1.0] + [s for _, s in similar_words]

            word_vectors = np.array([model.wv[w] for w in words])

            # 3D t-SNE
            perplexity = min(5, len(words) - 1)
            tsne = TSNE(n_components=3, random_state=42, perplexity=perplexity)
            coords = tsne.fit_transform(word_vectors)

            # 타겟 단어를 원점(0,0,0)으로 이동
            target_coord = coords[0]
            coords = coords - target_coord  # 모든 좌표를 이동

            all_data.append({
                'era': era_name,
                'color': color,
                'coords': coords,
                'words': words,
                'scores': scores
            })

        except KeyError:
            print(f"  ⚠ '{target_word}'가 {era_name}에 없음")
            continue

    # 범례에 사용할 era별로 한 번만 표시
    legend_added = set()

    # 시대별로 플롯
    for data in all_data:
        coords = data['coords']
        words = data['words']
        scores = data['scores']
        color = data['color']
        era_name = data['era']

        # 타겟 단어 (중심, 0번째)
        ax.scatter(
            [coords[0, 0]], [coords[0, 1]], [coords[0, 2]],
            c=[color],
            s=200,
            alpha=1.0,
            edgecolors='black',
            linewidths=3,
            marker='*',
            label=era_name if era_name not in legend_added else ""
        )
        legend_added.add(era_name)

        # 타겟 단어 라벨 (크게, 위쪽으로 오프셋)
        ax.text(
            coords[0, 0], coords[0, 1], coords[0, 2] + 2,  # z축으로 올림
            target_word,
            fontsize=22,
            color='black',
            weight='bold',
            ha='center',
            va='bottom',
            zorder=100,  # 다른 요소들 위에 표시
            
        )

        # 유사 단어들 (1번째부터)
        sizes = [40 for _ in scores[1:]]  # 작은 점

        ax.scatter(
            coords[1:, 0], coords[1:, 1], coords[1:, 2],
            c=[color] * len(coords[1:]),
            s=sizes,
            alpha=0.8,
            edgecolors='black',
            linewidths=1
        )

        # 모든 유사 단어에 라벨 표시 (겹침 방지)
        for i, word in enumerate(words[1:], start=1):
            # 방향은 유지하되 거리는 일정하게
            direction = np.array([coords[i, 0], coords[i, 1], coords[i, 2]])
            distance = np.linalg.norm(direction)

            if distance > 0:
                # 단위 벡터 (방향만 유지)
                unit_vector = direction / distance
                # 일정한 거리만큼 떨어뜨리기
                offset = unit_vector * 35 # 50 유닛 일정하게
            else:
                # 원점에 있으면 z축으로
                offset = np.array([0, 0, 3])

            # 반투명 배경 박스로 가독성 향상
            ax.text(
                coords[i, 0] + offset[0],
                coords[i, 1] + offset[1],
                coords[i, 2] + offset[2],
                word,
                fontsize=12,
                color='black',
                ha='center',
                va='center',
                bbox=dict(
                    boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor='none',
                    alpha=0.85
                )
            )

    # 축 설정 (명확히 표시)
    ax.set_xlabel('X축', fontsize=12, color='black', weight='bold')
    ax.set_ylabel('Y축', fontsize=12, color='black', weight='bold')
    ax.set_zlabel('Z축', fontsize=12, color='black', weight='bold')
    ax.grid(True, alpha=0.3)
    ax.xaxis.pane.fill = True
    ax.yaxis.pane.fill = True
    ax.zaxis.pane.fill = True
    ax.xaxis.pane.set_alpha(0.1)
    ax.yaxis.pane.set_alpha(0.1)
    ax.zaxis.pane.set_alpha(0.1)
    ax.tick_params(colors='black')

    # 타이틀 (그래프에 가깝게)
    plt.title(
        f"'{target_word}'의 시대별 의미 변화 (3D t-SNE)",
        fontsize=40,
        color='black',
        weight='bold',
        pad=-50 # 음수로 더 가깝게
    )

    # 범례 (색깔 박스만 표시, 그래프 안쪽으로)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor=era_colors['Era 1 (1996-2005)'], label='Era 1 (1996-2005)'),
        Patch(facecolor=era_colors['Era 2 (2006-2015)'], label='Era 2 (2006-2015)'),
        Patch(facecolor=era_colors['Era 3 (2016-2025)'], label='Era 3 (2016-2025)')
    ]
    ax.legend(
        handles=legend_elements,
        loc='upper right',
        bbox_to_anchor=(0.9, 0.9),  # 그래프 안쪽으로 이동
        fontsize=11,
        frameon=True,
        facecolor='white',
        edgecolor='black',
        labelcolor='black'
    )

    # 카메라 각도
    ax.view_init(elev=20, azim=45)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
        print(f"  ✓ 저장: {save_path}")

    plt.close()


# ============================================================
# Step 5. 리포트 생성
# ============================================================

def trace_similar_word_sources(models, eras_data, df, target_words,
                               output_file='output/word_sources_report.txt'):
    """
    유사 단어별 출처 곡 추적 리포트 (전체 곡 + 원문 가사 근거)
    - 원문 가사(Lyrics)에서 해당 단어가 등장하는 문맥을 추출
    - 모든 곡 목록 포함
    """
    import re as re_module

    print("\n[유사 단어 출처 추적]")

    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 80 + "\n")
        f.write("  유사 단어 출처 추적 리포트 (전곡 + 원문 근거)\n")
        f.write("=" * 80 + "\n")

        for target_word in target_words:
            f.write(f"\n\n{'#' * 80}\n")
            f.write(f"  '{target_word}' 유사 단어 출처\n")
            f.write(f"{'#' * 80}\n")

            for era_name, model in models.items():
                years = re_module.findall(r'\d{4}', era_name)
                if len(years) != 2:
                    continue
                year_start, year_end = int(years[0]), int(years[1])
                era_df = df[(df['Year'] >= year_start) & (df['Year'] <= year_end)]

                f.write(f"\n\n  [{era_name}]\n")
                f.write(f"  {'=' * 60}\n")

                try:
                    similar = model.wv.most_similar(target_word, topn=10)
                except KeyError:
                    f.write(f"  '{target_word}' 데이터 없음\n")
                    continue

                for sim_word, score in similar:
                    # Processed_Tokens + 원문 가사(Lyrics) 결합 검색
                    matches = []
                    search_terms = _get_search_stems(sim_word)

                    for _, row in era_df.iterrows():
                        # 1차: Processed_Tokens에서 원형(lemma) 매칭
                        token_str = row.get('Processed_Tokens', '')
                        tokens = parse_tokens(token_str) if not pd.isna(token_str) else []
                        in_tokens = sim_word in tokens

                        # 2차: 원문 가사에서 어간 검색
                        lyrics = str(row.get('Lyrics', ''))
                        if pd.isna(lyrics):
                            lyrics = ''
                        found_term = None
                        for st in search_terms:
                            if _find_with_boundary(lyrics, st) != -1:
                                found_term = st
                                break

                        if not in_tokens and not found_term:
                            continue

                        # 문맥 추출 (원문에서)
                        contexts = []
                        if found_term:
                            contexts = _extract_all_contexts(lyrics, found_term)
                        elif in_tokens and lyrics:
                            # 토큰에는 있지만 어간 검색으론 못 찾은 경우
                            # → 원형으로 한번 더 시도
                            if _find_with_boundary(lyrics, sim_word) != -1:
                                contexts = _extract_all_contexts(lyrics, sim_word)

                        matches.append({
                            'Year': row['Year'],
                            'Month': row.get('Month', ''),
                            'Rank': row.get('Rank', ''),
                            'Title': row['Title'],
                            'Artist': row['Artist'],
                            'Contexts': contexts
                        })

                    f.write(f"\n  ■ {sim_word} (유사도: {score:.3f}) - 총 {len(matches)}곡\n")
                    f.write(f"  {'-' * 55}\n")

                    if not matches:
                        f.write(f"    (원문에서 발견되지 않음)\n")
                        continue

                    for m in matches:
                        rank_info = f"#{m['Rank']}" if m['Rank'] else ''
                        f.write(f"    [{m['Year']}] {m['Artist']} - {m['Title']} {rank_info}\n")
                        if m['Contexts']:
                            for ctx in m['Contexts'][:2]:
                                f.write(f"      → \"{ctx}\"\n")
                        else:
                            f.write(f"      → (토큰 매칭, 원문 문맥 미발견)\n")

                    f.write("\n")

            print(f"  ✓ '{target_word}' 출처 추적 완료")

    print(f"✓ 출처 리포트 저장: {output_file}")


def _is_korean_syllable(ch):
    """한글 음절(가~힣) 여부 확인"""
    return '\uAC00' <= ch <= '\uD7A3'


def _find_with_boundary(text, word, start=0):
    """
    텍스트에서 단어를 찾되, 앞 글자가 한글이 아닌 경우만 매칭
    (예: '군대'를 찾을 때 '수군대'는 매칭하지 않음)
    """
    while True:
        idx = text.find(word, start)
        if idx == -1:
            return -1
        # 앞 글자 경계 확인: 앞 글자가 한글 음절이면 다른 단어의 일부
        if idx > 0 and _is_korean_syllable(text[idx - 1]):
            start = idx + 1
            continue
        return idx


def _get_search_stems(word):
    """
    동사/형용사의 어간(stem)을 추출하여 검색어 목록 생성
    예: '앓다' → ['앓다', '앓아', '앓고', '앓']
        '헤아리다' → ['헤아리다', '헤아리', '헤아려', '헤아릴']
        '기약' (명사) → ['기약']
    """
    stems = [word]  # 원형 그대로도 포함

    # '~다'로 끝나는 동사/형용사 → 어간 추출
    if word.endswith('다') and len(word) >= 2:
        stem = word[:-1]  # '앓다' → '앓', '헤아리다' → '헤아리'
        stems.append(stem)

        # 주요 활용 어미 추가
        last_char = stem[-1] if stem else ''
        if last_char:
            # 받침 유무에 따라 다른 활용형
            code = ord(last_char) - 0xAC00
            if code >= 0:
                jong = code % 28  # 종성 (0이면 받침 없음)
                if jong == 0:  # 받침 없음: 가다→가, 서다→서
                    stems.append(stem + '고')
                    stems.append(stem + '서')
                    stems.append(stem + '지')
                    stems.append(stem + 'ㄹ')
                else:  # 받침 있음: 앓→앓아, 먹→먹어
                    stems.append(stem + '아')
                    stems.append(stem + '어')
                    stems.append(stem + '고')
                    stems.append(stem + '은')
                    stems.append(stem + '을')
    else:
        # 명사: 그대로 + 앞 2글자 (3글자 이상일 때)
        if len(word) >= 3:
            stems.append(word[:2])

    return stems


def _extract_all_contexts(lyrics, word, window=25, max_contexts=2):
    """원문 가사에서 단어가 등장하는 모든 문맥 추출 (단어 경계 확인)"""
    contexts = []
    start = 0
    while True:
        idx = _find_with_boundary(lyrics, word, start)
        if idx == -1:
            break
        s = max(0, idx - window)
        e = min(len(lyrics), idx + len(word) + window)
        ctx = lyrics[s:e].replace('\n', ' ').strip()
        if s > 0:
            ctx = '...' + ctx
        if e < len(lyrics):
            ctx = ctx + '...'
        contexts.append(ctx)
        start = idx + len(word)
        if len(contexts) >= max_contexts:
            break
    return contexts

    print(f"✓ 출처 리포트 저장: {output_file}")


def generate_report(models, target_words, output_file='analysis_report.txt'):
    """분석 결과 텍스트 리포트 생성"""
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("  K-POP 가사 시대별 의미 변화 분석 리포트\n")
        f.write("=" * 70 + "\n\n")

        for word in target_words:
            f.write(f"\n■ '{word}'의 시대별 의미 변화\n")
            f.write("-" * 50 + "\n")

            for era_name, model in models.items():
                try:
                    similar = model.wv.most_similar(word, topn=10)
                    words_str = ', '.join([w[0] for w in similar])
                    f.write(f"\n[{era_name}]\n  {words_str}\n")
                except KeyError:
                    f.write(f"\n[{era_name}] 해당 단어 없음\n")

    print(f"\n✓ 리포트 저장: {output_file}")


# ============================================================
# 메인 실행
# ============================================================

def main():
    print("\n" + "=" * 60)
    print("  K-POP 가사 시대별 의미 변화 분석")
    print("  (Melon Chart 1996-2025, 전처리 데이터)")
    print("=" * 60)

    # 1. 전처리된 데이터 로드
    print("\n[Step 1] 전처리된 데이터 로드")
    df = load_processed_data('processed_lyrics')

    # 2. 시대별 분할
    print("\n[Step 2] 시대별 데이터 분할")
    eras = slice_by_era(df)

    # 3. 모델 학습
    print("\n[Step 3] 시대별 Word2Vec 모델 학습")
    models = train_all_models(eras)

    # 4. 의미 변화 분석
    print("\n[Step 4] 의미 변화 분석")
    target_words = ['침대','만남','여자','첫사랑','완벽']

    for word in target_words:
        analyze_semantic_shift(models, word)

    # 5. 시각화 (파일 저장만)
    print("\n[Step 5] t-SNE 시각화 저장")
    os.makedirs('output', exist_ok=True)
    for word in target_words:
        try:
            plot_semantic_field(word, models, save_path=f'output/tsne_{word}.png')
        except Exception as e:
            print(f"  ⚠ '{word}' 실패: {e}")

    # 6. 리포트 생성
    generate_report(models, target_words, 'output/analysis_report.txt')

    # 7. 유사 단어 출처 추적
    print("\n[Step 6] 유사 단어 출처 추적")
    trace_similar_word_sources(models, eras, df, target_words,
                               'output/word_sources_report.txt')

    print("\n" + "=" * 60)
    print("  분석 완료!")
    print("=" * 60)

    return models, eras, df


if __name__ == "__main__":
    models, eras, df = main()

    # 추가 분석 예시:
    # deep_dive_word(models, df, '할아버지', 'Era6 (2021-2025)')
    # find_word_sources(df, '여의도', era_range=(2016, 2020))
