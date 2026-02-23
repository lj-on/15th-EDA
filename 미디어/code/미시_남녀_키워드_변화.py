import pandas as pd
import glob
import os
import ast
import re
import numpy as np
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
from sklearn.manifold import TSNE
from wordcloud import WordCloud
from matplotlib import font_manager, rc
import matplotlib.pyplot as plt


# 1. 설정 및 불용어 정의
STOPWORDS = {
    '하다', '되다', '있다', '없다', '같다', '보다', '주다', '오다', '가다',
    '이다', '아니다', '않다', '못하다', '싶다', '알다', '모르다',
    '그렇다', '어떻다', '이렇다', '저렇다',
    '저', '그', '이것', '그것', '저것',
    '때', '것', '수', '듯', '더', '또', '다시', '너무', '정말', '진짜','워우워','아니야'
}

# 2. 유틸리티 함수 (사용자 작성 코드 포함)
def is_onomatopoeia(word):
    """의성어/반복 어구 감지"""
    if len(word) < 2: return False
    if len(set(word)) == 1: return True
    if len(word) >= 4 and len(word) % 2 == 0:
        half = len(word) // 2
        if word[:half] == word[half:]: return True
    return False

def is_valid_korean(word):
    """유효한 한글 단어인지 확인"""
    if not word or len(word) < 2: return False
    has_korean = any('가' <= c <= '힣' for c in word)
    if not has_korean: return False
    if is_onomatopoeia(word): return False
    if word in STOPWORDS: return False
    return True

def parse_tokens(token_str):
    """문자열 형태의 토큰 리스트를 파싱 및 필터링"""
    if pd.isna(token_str): return []
    try:
        # 엑셀의 문자열 리스트를 실제 객체로 변환
        tokens = ast.literal_eval(token_str)
        return [t for t in tokens if is_valid_korean(t)]
    except:
        return []

def normalize_gender_value(gender):
    """성별 데이터 정규화 (필요에 따라 수정 가능)"""
    gender = str(gender).lower()
    if 'female' in gender or '여' in gender: return 'female'
    if 'male' in gender or '남' in gender: return 'male'
    return 'mixed/unknown'

#3
def load_and_preprocess_kpop(lyrics_dir):
    # 1. ~$로 시작하는 임시 파일은 제외하고 실제 엑셀 파일만 리스트업
    all_files = [f for f in glob.glob(os.path.join(lyrics_dir, '*.xlsx')) 
                 if not os.path.basename(f).startswith('~$')]
    
    if not all_files:
        print(f"❌ 경로 내에 유효한 엑셀 파일이 없습니다: {lyrics_dir}")
        return None

    df_list = []
    for file in all_files:
        try:
            # 2. engine='openpyxl'을 명시적으로 지정하여 에러 방지
            df = pd.read_excel(file, engine='openpyxl')
            df_list.append(df)
            print(f"  - 로드 완료: {os.path.basename(file)}")
        except Exception as e:
            # 어떤 파일에서 문제가 생겼는지 출력하고 다음 파일로 진행
            print(f"⚠️ 파일 로드 실패 ({os.path.basename(file)}): {e}")

    if not df_list:
        return None

    combined_df = pd.concat(df_list, ignore_index=True)
    
    # 중복 제거 (Song_ID 기준)
    if 'Song_ID' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['Song_ID'], keep='first')
    
    # 성별 정규화
    if 'Artist_Gender' in combined_df.columns:
        combined_df['Gender'] = combined_df['Artist_Gender'].apply(normalize_gender_value)
    
    print("⏳ 가사 토큰 정제 중...")
    # Processed_Tokens 컬럼 존재 여부 확인 후 처리
    if 'Processed_Tokens' in combined_df.columns:
        combined_df['Clean_Tokens'] = combined_df['Processed_Tokens'].apply(parse_tokens)
    else:
        print("❌ 'Processed_Tokens' 컬럼을 찾을 수 없습니다.")
        return combined_df
    
    combined_df['Total_Words'] = combined_df['Clean_Tokens'].apply(len)
    combined_df['Unique_Words'] = combined_df['Clean_Tokens'].apply(lambda x: len(set(x)))
    combined_df['Era'] = (combined_df['Year'] // 10 * 10).astype(str) + 's'

    print(f"✅ 완료! 총 {len(combined_df):,}개 곡 데이터 전처리 완료.")
    return combined_df

# 4. 실행
data_path = r'C:\Users\User\Desktop\DSL\eda\kpop_lyrics_analysis-main\final_dataset'
final_df = load_and_preprocess_kpop(data_path)

# 결과 확인
if final_df is not None:
    print(final_df[['Year', 'Title', 'Total_Words', 'Era']].head())

# Step 2. 시대별 데이터 분할 및 모델 학습
# ============================================================

ERA_DEFS = [
    ('Era1 (1996-2005)', 1996, 2005),
    ('Era2 (2006-2015)', 2006, 2015),
    ('Era3 (2016-2025)', 2016, 2025),
]


def slice_by_era(df):
    """데이터를 3개 시대로 분할"""
    eras = {
        name: df[(df['Year'] >= start) & (df['Year'] <= end)]
        for name, start, end in ERA_DEFS
    }

    print("\n✓ 시대별 데이터 분할:")
    for name, data in eras.items():
        print(f"  - {name}: {len(data):,}곡")

    return eras


def slice_by_era_and_gender(df, gender_col='Gender'):
    """시대+성별 분할"""
    groups = {}
    for name, start, end in ERA_DEFS:
        era_df = df[(df['Year'] >= start) & (df['Year'] <= end)]
        if gender_col not in era_df.columns:
            groups[name] = era_df
            continue
        for gender, gdf in era_df.groupby(gender_col):
            label = f"{name} | {gender}"
            groups[label] = gdf

    print("\n✓ 시대+성별 데이터 분할:")
    for name, data in groups.items():
        print(f"  - {name}: {len(data):,}곡")

    return groups


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


def plot_semantic_field(target_word, models, topn=15, save_path=None):
    """시대별 의미장 시각화 (개선된 버전)"""
    setup_korean_font()

    # 시대별 색상 팔레트
    era_colors = ['#E74C3C', '#3498DB', '#9B59B6']

    fig, axes = plt.subplots(1, 3, figsize=(21, 7))
    axes = axes.flatten()

    fig.suptitle(f"'{target_word}'의 시대별 의미 변화 (t-SNE)",
                 fontsize=18, fontweight='bold', y=0.98)

    for i, (name, model) in enumerate(models.items()):
        ax = axes[i]
        color = era_colors[i]

        try:
            similar_words = model.wv.most_similar(target_word, topn=topn)
            words = [target_word] + [w[0] for w in similar_words]
            scores = [1.0] + [s for _, s in similar_words]  # 유사도 점수

            word_vectors = np.array([model.wv[w] for w in words])

            perplexity = min(5, len(words) - 1)
            tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
            coords = tsne.fit_transform(word_vectors)

            # 배경 스타일
            ax.set_facecolor('#FAFAFA')

            # 유사도에 따른 점 크기 (가까울수록 큼)
            sizes = [300] + [80 + scores[j] * 150 for j in range(1, len(scores))]

            # 유사 단어들 (연한 색)
            ax.scatter(coords[1:, 0], coords[1:, 1],
                      c=color, s=sizes[1:], alpha=0.4, edgecolors='white', linewidths=1.5)

            # 타겟 단어 (진한 색, 별 모양)
            ax.scatter(coords[0, 0], coords[0, 1],
                      c=color, s=400, marker='*', edgecolors='black', linewidths=2, zorder=5)

            # 상위 5개만 라벨 표시 (겹침 방지)
            for j, txt in enumerate(words[:6]):
                if j == 0:
                    ax.annotate(txt, (coords[j, 0], coords[j, 1]),
                               fontsize=14, fontweight='bold', color='black',
                               ha='center', va='bottom',
                               bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
                else:
                    ax.annotate(txt, (coords[j, 0], coords[j, 1]),
                               fontsize=10, color='#333333',
                               ha='center', va='bottom')

            # 나머지 단어들 (작은 폰트)
            for j, txt in enumerate(words[6:], start=6):
                ax.annotate(txt, (coords[j, 0], coords[j, 1]),
                           fontsize=8, color='#666666', alpha=0.7,
                           ha='center', va='bottom')

            # 시대 이름 (년도만 추출)
            era_short = name.split('(')[1].replace(')', '') if '(' in name else name
            ax.set_title(era_short, fontsize=14, fontweight='bold',
                        color=color, pad=10)

            ax.axis('off')

            # 테두리 추가
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color(color)
                spine.set_linewidth(2)

        except KeyError:
            ax.set_facecolor('#F5F5F5')
            ax.text(0.5, 0.5, f"'{target_word}'\n데이터 없음",
                   ha='center', va='center', fontsize=12, color='#999999')
            era_short = name.split('(')[1].replace(')', '') if '(' in name else name
            ax.set_title(era_short, fontsize=14, fontweight='bold', color='#CCCCCC')
            ax.axis('off')

    for j in range(i + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    plt.subplots_adjust(top=0.92)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='white')
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


def _write_report_header(f, title):
    f.write("=" * 70 + "\n")
    f.write(f"  {title}\n")
    f.write("=" * 70 + "\n\n")


def generate_report(models, target_words, output_file='analysis_report.txt',
                    title="K-POP 가사 시대별 의미 변화 분석 리포트",
                    mode='w'):
    """분석 결과 텍스트 리포트 생성"""
    with open(output_file, mode, encoding='utf-8') as f:
        if mode == 'w':
            _write_report_header(f, title)
        else:
            f.write("\n")
            _write_report_header(f, title)

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


def _extract_gender_from_key(model_key):
    if '|' not in model_key:
        return None
    return model_key.split('|', 1)[1].strip()


def _filter_models_by_gender(models, gender):
    return {
        name: model for name, model in models.items()
        if _extract_gender_from_key(name) == gender
    }


def _get_genders_from_models(models):
    genders = set()
    for name in models.keys():
        g = _extract_gender_from_key(name)
        if g:
            genders.add(g)
    return sorted(genders)


def append_gender_report(base_report_file, gender_models, target_words):
    """analysis_report.txt에 성별 섹션 추가"""
    if not gender_models:
        return
    generate_report(
        gender_models,
        target_words,
        output_file=base_report_file,
        title="성별(시대+성별) 의미 변화 분석",
        mode='a'
    )


def generate_gender_reports(gender_models, target_words, output_dir='output'):
    """성별별 리포트 파일 생성"""
    genders = _get_genders_from_models(gender_models)
    for gender in genders:
        models = _filter_models_by_gender(gender_models, gender)
        if not models:
            continue
        output_file = os.path.join(output_dir, f'analysis_report_{gender}.txt')
        title = f"K-POP 가사 의미 변화 분석 리포트 ({gender})"
        generate_report(models, target_words, output_file, title=title, mode='w')


def generate_gender_tsne_plots(gender_models, target_words, output_dir='output'):
    """성별 기반 t-SNE 시각화 생성"""
    genders = _get_genders_from_models(gender_models)
    for gender in genders:
        models = _filter_models_by_gender(gender_models, gender)
        if not models:
            continue
        for word in target_words:
            try:
                save_path = os.path.join(output_dir, f'tsne_{word}_{gender}.png')
                plot_semantic_field(word, models, save_path=save_path)
            except Exception as e:
                print(f"  ⚠ '{word}' ({gender}) 시각화 실패: {e}")


def generate_gender_word_sources(gender_models, df, target_words, output_dir='output'):
    """성별 기반 word_sources 리포트 생성"""
    if not gender_models:
        return
    trace_similar_word_sources(
        gender_models,
        None,
        df,
        target_words,
        os.path.join(output_dir, 'word_sources_report_gender.txt')
    )

    genders = _get_genders_from_models(gender_models)
    for gender in genders:
        models = _filter_models_by_gender(gender_models, gender)
        if not models:
            continue
        output_file = os.path.join(output_dir, f'word_sources_report_{gender}.txt')
        trace_similar_word_sources(models, None, df, target_words, output_file)

# ============================================================
# 메인 실행부 (안전 경로 및 필터링 적용 버전)
# ============================================================


def print_gender_distribution(df, title="Overall"):
    """
    데이터프레임 내의 성별 분포를 출력하는 함수입니다.
    """
    print(f"--- {title} Gender Distribution ---")
    if 'Gender' in df.columns:
        dist = df['Gender'].value_counts()
        print(dist)
    else:
        print("Error: 'Gender' column not found in DataFrame.")
    print("-" * 30) 

    
def main():
    print("\n" + "=" * 60)
    print("  K-POP 가사 시대별 의미 변화 분석 시작")
    print("  (분석 대상: 사랑, 이별")
    print("=" * 60)

    # 1. 데이터 로드 및 정제
    print("\n[Step 1] 전처리된 데이터 로드 및 정제")
    df = load_and_preprocess_kpop('C:\\Users\\User\\Desktop\\DSL\\eda\\kpop_lyrics_analysis-main\\final_dataset') 

    if df is None:
        print("❌ 데이터를 찾을 수 없어 종료합니다.")
        return None, None, None, None

    # [중요] 파일명 에러 방지를 위해 성별 값 내의 슬래시(/)를 언더바(_)로 치환
    if 'Gender' in df.columns:
        df['Gender'] = df['Gender'].replace('mixed/unknown', 'mixed_unknown')

    # 1b. 성별 분포 출력
    print("\n[Step 1b] Gender Distribution")
    print_gender_distribution(df, title="Overall")
    for name, start, end in ERA_DEFS:
        era_df = df[(df['Year'] >= start) & (df['Year'] <= end)]
        print_gender_distribution(era_df, title=name)

    # 2. 시대별/성별 분할
    print("\n[Step 2] 데이터 분할 (시대 및 성별)")
    eras = slice_by_era(df)
    gender_groups = slice_by_era_and_gender(df, gender_col='Gender')

    # 3. 모델 학습 (Word2Vec)
    print("\n[Step 3] Word2Vec 모델 학습 진행 중...")
    models = train_all_models(eras)
    gender_models = train_all_models(gender_groups)

    # 4. 분석 대상 단어 확정 (23개)
    target_words = [
       '사랑','이별'
    ]

    # 4a. 시대별 의미 변화 분석 출력
    print("\n[Step 4] 시대별 의미 변화 분석 결과")
    for word in target_words:
        analyze_semantic_shift(models, word)

    # 4b. 성별 기반 의미 변화 분석 출력 (리포트에는 저장되나 화면 출력은 선택사항)
    print("\n[Step 4b] 성별 기반 의미 변화 분석 진행 중...")

    # 5. 시각화 및 리포트 파일 저장
    print("\n[Step 5] 시각화 및 리포트 파일 생성 중...")
    os.makedirs('output', exist_ok=True)
    
    # 시대별 t-SNE 시각화
    for word in target_words:
        try:
            plot_semantic_field(word, models, save_path=f'output/tsne_{word}.png')
        except Exception as e:
            print(f"  ⚠️ '{word}' 시대별 시각화 실패: {e}")

    # 성별 기반 t-SNE 시각화 (수정된 함수 호출: mixed/unknown 제외 로직 포함)
    generate_gender_tsne_plots(gender_models, target_words, output_dir='output')

    # 리포트 생성
    generate_report(models, target_words, 'output/analysis_report.txt')
    append_gender_report('output/analysis_report.txt', gender_models, target_words)
    
    # 성별 리포트 개별 생성 (수정된 함수 호출: mixed/unknown 제외 로직 포함)
    generate_gender_reports(gender_models, target_words, output_dir='output')

    # 6. 유사 단어 출처 추적 (원문 가사 매칭)
    print("\n[Step 6] 원문 가사 기반 출처 추적 리포트 생성 중...")
    trace_similar_word_sources(models, eras, df, target_words, 'output/word_sources_report.txt')
    generate_gender_word_sources(gender_models, df, target_words, output_dir='output')

    print("\n" + "=" * 60)
    print("  🎉 분석 완료! 모든 결과는 'output' 폴더에 저장되었습니다.")
    print("=" * 60)

    return models, eras, gender_models, df

# 스크립트 실행
if __name__ == "__main__":
    models, eras, gender_models, df = main()

# 1. 한글 폰트 설정 (Windows/Mac 공용)
def set_korean_font():
    if os.name == 'nt': # 윈도우
        font_path = "C:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        return font_path
    else: # 맥
        rc('font', family='AppleGothic')
        return "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

# 2. 리포트에서 단어 뭉치 추출 함수
def get_all_words_from_report(filename, keyword):
    if not os.path.exists(filename):
        # 파일이 없을 경우 (예: mixed_unknown 제외 등)를 대비한 예외 처리
        return ""
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 키워드별 섹션 분리 (섹션 구분자 '■ ' 기준)
    sections = re.split(r'■ ', content)
    combined_words = []
    
    for section in sections:
        if f"'{keyword}'" in section:
            # 해당 키워드 섹션 내의 유사 단어 리스트(각 Era별)를 모두 추출
            # 정규표현식으로 [Era...] 이후 등장하는 단어들을 긁어옵니다.
            matches = re.findall(r'\[.*?\]\n\s+(.*?)\n', section)
            for m in matches:
                # 쉼표를 제거하고 공백으로 분리하여 단어 리스트에 추가
                combined_words.extend(m.replace(',', '').split())
                
    return " ".join(combined_words)

# 3. 워드클라우드 시각화 실행
font_p = set_korean_font()
output_dir = 'output' # 리포트가 저장된 경로

# 우리가 최종 확정한 23개 타겟 단어 중 주요 단어 선택 (혹은 전체 순회)
# 시각화하고 싶은 단어만 추려도 되고, target_words를 그대로 써도 됩니다.
selected_keywords = [
      '사랑','이별'
    ]

for kw in selected_keywords:
    # 성별 리포트 경로 설정 (파일명 안전화 로직 반영)
    male_report = os.path.join(output_dir, 'analysis_report_male.txt')
    female_report = os.path.join(output_dir, 'analysis_report_female.txt')
    
    male_text = get_all_words_from_report(male_report, kw)
    female_text = get_all_words_from_report(female_report, kw)
    
    # 데이터가 부족한 경우 스킵
    if not male_text.strip() or not female_text.strip():
        print(f"⚠️ '{kw}'에 대한 성별 비교 데이터가 부족하여 건너뜁니다.")
        continue

    # 워드클라우드 객체 생성
    wc_params = {
        'font_path': font_p,
        'background_color': 'white',
        'width': 600,
        'height': 600,
        'max_words': 100,
        'prefer_horizontal': 0.9
    }
    
    wc_male = WordCloud(**wc_params, colormap='Blues').generate(male_text)
    wc_female = WordCloud(**wc_params, colormap='Reds').generate(female_text)

    # 시각화 레이아웃
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))
    
    axes[0].imshow(wc_male, interpolation='bilinear')
    axes[0].set_title(f"남성 아티스트(Male): '{kw}'", fontsize=20, color='blue', pad=20)
    axes[0].axis('off')
    
    axes[1].imshow(wc_female, interpolation='bilinear')
    axes[1].set_title(f"여성 아티스트(Female): '{kw}'", fontsize=20, color='red', pad=20)
    axes[1].axis('off')

    plt.suptitle(f"K-POP 30년 성별 가사 의미장 대조: '{kw}'", fontsize=26, fontweight='bold', y=1.05)
    
    # 결과 저장
    save_name = f"{output_dir}/wordcloud_{kw}_gender_comparison.png"
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"✅ 완료: {save_name} 저장됨.")



# 1. 한글 폰트 설정 (기존과 동일)
def set_korean_font():
    if os.name == 'nt':
        font_path = "C:/Windows/Fonts/malgun.ttf"
        font_name = font_manager.FontProperties(fname=font_path).get_name()
        rc('font', family=font_name)
        return font_path
    else:
        rc('font', family='AppleGothic')
        return "/System/Library/Fonts/Supplemental/AppleGothic.ttf"

# 2. 리포트에서 [시대별] 단어 뭉치를 딕셔너리로 추출하는 함수
def get_era_words_from_report(filename, keyword):
    if not os.path.exists(filename):
        return {}
    
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # 해당 키워드 섹션만 추출
    sections = re.split(r'■ ', content)
    target_section = ""
    for section in sections:
        if f"'{keyword}'" in section:
            target_section = section
            break
            
    if not target_section:
        return {}

    # 시대별로 단어 매칭 (Era1, Era2, Era3 추출)
    era_data = {}
    # [Era...] 이후 다음 [Era...] 혹은 섹션 끝 전까지의 단어들을 긁어옴
    matches = re.findall(r'\[(Era\d.*?)\]\n\s+(.*?)\n', target_section)
    
    for era_name, words in matches:
        era_data[era_name] = words.replace(',', '')
        
    return era_data

# 3. 시각화 실행
font_p = set_korean_font()
output_dir = 'output'
selected_keywords = ['사랑', '이별']
eras = ['Era1', 'Era2', 'Era3'] # 시대 리스트

for kw in selected_keywords:
    male_report = os.path.join(output_dir, 'analysis_report_male.txt')
    female_report = os.path.join(output_dir, 'analysis_report_female.txt')
    
    male_era_data = get_era_words_from_report(male_report, kw)
    female_era_data = get_era_words_from_report(female_report, kw)

    # 3행(시대) 2열(성별) 레이아웃 생성
    fig, axes = plt.subplots(3, 2, figsize=(15, 18))
    
    wc_params = {
        'font_path': font_p, 'background_color': 'white',
        'width': 400, 'height': 400, 'max_words': 50
    }

    for idx, era in enumerate(eras):
        # 해당 에라 명칭이 포함된 키 찾기 (예: 'Era1 (1996-2005)')
        m_key = [k for k in male_era_data.keys() if era in k]
        f_key = [k for k in female_era_data.keys() if era in k]
        
        # 남성 워드클라우드
        if m_key and male_era_data[m_key[0]].strip():
            wc_m = WordCloud(**wc_params, colormap='Blues').generate(male_era_data[m_key[0]])
            axes[idx, 0].imshow(wc_m, interpolation='bilinear')
            axes[idx, 0].set_title(f"MALE | {m_key[0]}", fontsize=15, color='blue')
        axes[idx, 0].axis('off')

        # 여성 워드클라우드
        if f_key and female_era_data[f_key[0]].strip():
            wc_f = WordCloud(**wc_params, colormap='Reds').generate(female_era_data[f_key[0]])
            axes[idx, 1].imshow(wc_f, interpolation='bilinear')
            axes[idx, 1].set_title(f"FEMALE | {f_key[0]}", fontsize=15, color='red')
        axes[idx, 1].axis('off')

    plt.suptitle(f"K-POP 30년 '{kw}' 의미 변화 (시대별/성별 대조)", fontsize=22, fontweight='bold', y=1.02)
    
    save_name = f"{output_dir}/wordcloud_{kw}_timeline_comparison.png"
    plt.tight_layout()
    plt.savefig(save_name, dpi=300, bbox_inches='tight')
    plt.show()

passive_seeds = [
    # 기다림 / 수용
    "기다리다", "버티다", "견디다", "참다", "맡기다",

    # 상실 / 후퇴
    "놓치다", "잃다", "떠나보내다", "멀어지다", "사라지다",

    # 회피 / 머무름
    "숨기다", "피하다", "머뭇거리다", "망설이다", "남기다"
]

active_seeds = [
    # 접근 / 시도
    "다가가다", "잡다", "만나다", "말하다", "고백하다",

    # 결정 / 선택
    "결정하다", "선택하다", "정하다", "끊다", "바꾸다",

    # 주도 / 표현
    "지키다", "외치다", "요구하다", "이끌다", "시작하다"
]
def compare_gender_attitude(gender_models, target_word='사랑'):
    eras = ['Era1 (1996-2005)', 'Era2 (2006-2015)', 'Era3 (2016-2025)']
    results = []

    for era in eras:
        for gender in ['male', 'female']:
            model = gender_models.get(f"{era} | {gender}")
            if not model or target_word not in model.wv:
                continue
            
            # 해당 성별/시대의 '사랑' 유사어 10개
            similar_words = [w for w, s in model.wv.most_similar(target_word, topn=10)]
            
            # 수동성/능동성 점수 계산
            p_score = np.mean([model.wv.similarity(word, p) for word in similar_words for p in passive_seeds if p in model.wv])
            a_score = np.mean([model.wv.similarity(word, a) for word in similar_words for a in active_seeds if a in model.wv])
            
            results.append({
                '시대': era,
                '성별': gender,
                '수동성(Passive)': p_score,
                '능동성(Active)': a_score,
                '주체성 지수(A-P)': a_score - p_score
            })

    return pd.DataFrame(results)

# 실행
gender_diff_df = compare_gender_attitude(gender_models)
display(gender_diff_df)

def compare_gender_attitude(gender_models, target_word='이별'):
    eras = ['Era1 (1996-2005)', 'Era2 (2006-2015)', 'Era3 (2016-2025)']
    results = []

    for era in eras:
        for gender in ['male', 'female']:
            model = gender_models.get(f"{era} | {gender}")
            if not model or target_word not in model.wv:
                continue
            
            # 해당 성별/시대의 '사랑' 유사어 10개
            similar_words = [w for w, s in model.wv.most_similar(target_word, topn=10)]
            
            # 수동성/능동성 점수 계산
            p_score = np.mean([model.wv.similarity(word, p) for word in similar_words for p in passive_seeds if p in model.wv])
            a_score = np.mean([model.wv.similarity(word, a) for word in similar_words for a in active_seeds if a in model.wv])
            
            results.append({
                '시대': era,
                '성별': gender,
                '수동성(Passive)': p_score,
                '능동성(Active)': a_score,
                '주체성 지수(A-P)': a_score - p_score
            })

    return pd.DataFrame(results)

# 실행
gender_diff_df = compare_gender_attitude(gender_models)
display(gender_diff_df)


# 데이터 설정 (여성 전용)
eras = ['Era1', 'Era2', 'Era3']
love_female = [-0.0288, 0.0052, 0.0533]    # '사랑' 주체성 지수
breakup_female = [-0.0642, -0.0240, 0.0092] # '이별' 주체성 지수

# 2개의 그래프를 세로로 배치 (1행 2열로 하고 싶다면 subplots(1, 2)로 수정)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# --- 첫 번째 그래프: '사랑' 주체성 변화 ---
axes[0].plot(eras, love_female, marker='o', color='darkorange', linewidth=3, markersize=10)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1.5) # 기준선
axes[0].set_title("Female Agency Index: 'Love'", fontsize=16, pad=15)
axes[0].set_ylabel("Agency Index (A-P)", fontsize=12)
axes[0].set_ylim(-0.08, 0.08)
axes[0].grid(True, linestyle=':', alpha=0.6)
# 수치 표시
for i, txt in enumerate(love_female):
    axes[0].annotate(f'{txt:.4f}', (eras[i], love_female[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

# --- 두 번째 그래프: '이별' 주체성 변화 ---
axes[1].plot(eras, breakup_female, marker='s', color='crimson', linewidth=3, markersize=10)
axes[1].axhline(0, color='black', linestyle='--', linewidth=1.5) # 기준선
axes[1].set_title("Female Agency Index: 'Breakup'", fontsize=16, pad=15)
axes[1].set_ylabel("Agency Index (A-P)", fontsize=12)
axes[1].set_ylim(-0.08, 0.08)
axes[1].grid(True, linestyle=':', alpha=0.6)
# 수치 표시
for i, txt in enumerate(breakup_female):
    axes[1].annotate(f'{txt:.4f}', (eras[i], breakup_female[i]), textcoords="offset points", xytext=(0,10), ha='center', fontweight='bold')

plt.tight_layout()
plt.savefig('female_agency_separated.png', dpi=300)
plt.show()

# 1. 데이터 설정
eras = ['Era1', 'Era2', 'Era3']

# [메인] 여성 가사 데이터 (이전 분석 결과)
love_female = [-0.0288, 0.0052, 0.0533]
breakup_female = [-0.0642, -0.0240, 0.0092]

# [배경] 전체 시대별 주체성 지수 (방금 주신 데이터)
overall_agency = [-0.028816, 0.005243, 0.053326]

# 2. 그래프 생성 (2행 1열)
fig, axes = plt.subplots(2, 1, figsize=(10, 12))

# --- 첫 번째 그래프: '사랑' (여성 vs 전체) ---
# 전체 흐름 (희미한 선)
axes[0].plot(eras, overall_agency, color='gray', linestyle='--', alpha=0.3, label='Overall Trend')
# 여성 흐름 (진한 선)
axes[0].plot(eras, love_female, marker='o', color='darkorange', linewidth=3, markersize=10, label='Female: Love')

axes[0].axhline(0, color='black', linestyle='-', linewidth=1)
axes[0].set_title("Female 'Love' Agency vs. Overall Trend", fontsize=16, pad=15)
axes[0].set_ylabel("Agency Index (A-P)", fontsize=12)
axes[0].legend()
axes[0].grid(True, linestyle=':', alpha=0.6)

# --- 두 번째 그래프: '이별' (여성 vs 전체) ---
# 전체 흐름 (희미한 선)
axes[1].plot(eras, overall_agency, color='gray', linestyle='--', alpha=0.3, label='Overall Trend')
# 여성 흐름 (진한 선)
axes[1].plot(eras, breakup_female, marker='s', color='crimson', linewidth=3, markersize=10, label='Female: Breakup')

axes[1].axhline(0, color='black', linestyle='-', linewidth=1)
axes[1].set_title("Female 'Breakup' Agency vs. Overall Trend", fontsize=16, pad=15)
axes[1].set_ylabel("Agency Index (A-P)", fontsize=12)
axes[1].legend()
axes[1].grid(True, linestyle=':', alpha=0.6)

plt.tight_layout()
plt.savefig('female_vs_overall_comparison.png', dpi=300)
plt.show()