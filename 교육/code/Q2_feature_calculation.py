#!/usr/bin/env python3
"""
EDNet feature extraction pipeline (data -> student_features.csv).

역할:
- KT1/KT2/KT2_pay_res/KT4를 사용해 `features.csv`에 정의된 주요 feature들을
  학생 단위로 계산
- 미리 계산된 IRT 능력치(user_theta_optimized.csv)의 theta를 merge
- 권장 처리(clip/log1p 등)를 적용한 최종 테이블을 student_features.csv로 저장

분리된 후속 단계:
- 상관/인사이트 요약: summarize_insights.py
- 시각화(SVG): visualize_theta_features.py
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


# 스크립트 위치 기준 경로 (노트북/데스크탑 어디서든 동작)
_SCRIPT_DIR = Path(__file__).resolve().parent
OUT_DEFAULT = _SCRIPT_DIR / "outputs"
DATASET_DEFAULT = _SCRIPT_DIR.parent / "Dataset"  # DSL/Dataset 또는 프로젝트 루트의 Dataset

KT1_COLS = {"timestamp", "solving_id", "question_id", "user_answer", "elapsed_time", "correct"}
KT2_COLS = {"timestamp", "action_type", "item_id", "source", "user_answer", "platform"}
KT4_COLS = {"timestamp", "action_type", "item_id", "cursor_time", "source", "user_answer", "platform"}
KT2_PAY_RES_COLS = {"submit_idx", "question_id", "response_change"}

DWELL_CLIP_MS_DEFAULT = 10 * 60 * 1000  # 10분 (해설 체류 상한)
AFTER_WRONG_WINDOW_MS_DEFAULT = 10 * 60 * 1000  # 오답 후 해설 진입 윈도우 (기본 10분)

# NOTE: EDNet 데이터셋에서 item_id가 b%로 태깅되어 있어도,
# 실제로는 번들이 아닌 파트(1,2,5)가 섞여 있는 경우가 있음.
# 아래 helper는 "실제 번들"로 취급할 파트만 필터링하기 위한 규칙이다.
#   - 관례: item_id 형식이 b{part}{...} 라고 가정하면,
#     두 번째 문자(숫자)가 파트 번호가 된다.
#   - 여기서는 요구사항에 맞춰 파트 3,4,6,7만 번들로 인정한다.
VALID_BUNDLE_PARTS = {"3", "4", "6", "7"}


def _is_finite(x: float) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _safe_float(x: str, default: float = math.nan) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _safe_int(x: str, default: int = 0) -> int:
    try:
        return int(float(x))
    except Exception:
        return default


def _mean(values: Iterable[float]) -> float:
    s = 0.0
    n = 0
    for v in values:
        if _is_finite(v):
            s += float(v)
            n += 1
    return (s / n) if n > 0 else math.nan


def _safe_log1p(x: float) -> float:
    if not _is_finite(x) or x < 0:
        return math.nan
    return math.log1p(float(x))


def _is_real_bundle(item_id: str) -> bool:
    """
    EDNet에서 b%로 태깅된 모든 item이 진짜 '번들'은 아니므로,
    파트 3,4,6,7에 해당하는 번들만 사용하기 위한 필터.

    가정:
    - item_id는 \"b1234\" 처럼 b + 숫자들로 구성
    - 두 번째 문자(숫자)가 파트 번호
    """
    item_id = (item_id or "").strip()
    if not item_id.startswith("b") or len(item_id) < 2:
        return False
    part_digit = item_id[1]
    return part_digit in VALID_BUNDLE_PARTS


def _quantile(values: List[float], q: float) -> float:
    v = [float(x) for x in values if _is_finite(x)]
    if not v:
        return math.nan
    v.sort()
    if len(v) == 1:
        return v[0]
    idx = int(round(q * (len(v) - 1)))
    idx = max(0, min(len(v) - 1, idx))
    return v[idx]


def _clip_upper(x: float, upper: float) -> float:
    if not _is_finite(x) or not _is_finite(upper):
        return x
    return float(min(x, upper))


def _entropy(counter: Counter) -> float:
    total = sum(counter.values())
    if total <= 0:
        return math.nan
    ent = 0.0
    for c in counter.values():
        if c <= 0:
            continue
        p = c / total
        ent -= p * math.log(p)
    return float(ent)


def _dates_from_ts_ms(ts_ms: int) -> Optional[datetime.date]:
    try:
        return datetime.utcfromtimestamp(ts_ms / 1000.0).date()
    except Exception:
        return None


def _max_consecutive_days(days: List[datetime.date]) -> int:
    if not days:
        return 0
    days = sorted(set(days))
    best = 1
    cur = 1
    for i in range(1, len(days)):
        if (days[i] - days[i - 1]).days == 1:
            cur += 1
            best = max(best, cur)
        else:
            cur = 1
    return best


def _read_header(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        return header or []


def _validate_cols(path: Path, required: set) -> None:
    header = set(_read_header(path))
    missing = sorted(required - header)
    if missing:
        raise ValueError(f"{path} missing columns: {missing}. Found: {sorted(header)}")


def _iter_dict_rows(path: Path) -> Iterable[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            yield row


def _load_theta_map(dataset_root: Path, prefer_existing: bool = True) -> Dict[str, float]:
    theta_path = dataset_root / "user_theta_optimized.csv"
    if prefer_existing and theta_path.exists():
        _validate_cols(theta_path, {"user_id", "theta"})
        m: Dict[str, float] = {}
        for r in _iter_dict_rows(theta_path):
            uid = (r.get("user_id") or "").strip()
            th = _safe_float(r.get("theta") or "")
            if uid:
                m[uid] = th
        return m
    raise FileNotFoundError(
        "No precomputed theta file found (user_theta_optimized.csv). "
        "현재 스크립트는 외부 라이브러리 없이 동작하도록 IRT 재추정은 제외했습니다."
    )


@dataclass
class StudentFeaturesRaw:
    user_id: str
    accuracy: float
    avg_response_time_ms: float
    n_problems: int
    abandon_rate: float
    time_on_task_total_ms: float
    consecutive_days: int
    active_days: int
    source_entropy: float
    adaptive_offer: float
    enter_b_count: int
    erase_rate: float
    undo_rate: float
    explanation_adoption_rate: float
    explanation_after_wrong_rate: float
    explanation_time_per_problem_ms: float
    media_play_rate: float
    answer_change_rate: float
    accuracy_per_time: float
    accuracy_per_problem: float


def compute_student_features(
    *,
    user_id: str,
    kt1_path: Path,
    kt2_dir: Path,
    kt2_pay_res_dir: Path,
    kt4_dir: Path,
    dwell_clip_ms: int = DWELL_CLIP_MS_DEFAULT,
    after_wrong_window_ms: int = AFTER_WRONG_WINDOW_MS_DEFAULT,
) -> StudentFeaturesRaw:
    # --- KT1: 정오답, 응답 시간, 오답 목록 ---
    _validate_cols(kt1_path, KT1_COLS)
    correct_vals: List[float] = []
    elapsed_vals: List[float] = []
    wrong_attempts: List[Tuple[int, str]] = []  # (submit_ts_ms, question_id)
    n_problems = 0
    for r in _iter_dict_rows(kt1_path):
        n_problems += 1
        c = _safe_float(r.get("correct") or "")
        correct_vals.append(c)
        elapsed_vals.append(_safe_float(r.get("elapsed_time") or ""))
        if _is_finite(c) and float(c) == 0.0:
            ts = _safe_int(r.get("timestamp") or "", default=-1)
            qid = (r.get("question_id") or "").strip()
            if ts >= 0 and qid:
                wrong_attempts.append((ts, qid))
    accuracy = _mean(correct_vals)
    if _is_finite(accuracy):
        accuracy = max(0.0, min(1.0, float(accuracy)))
    avg_response_time_ms = _mean(elapsed_vals)

    # 기본값 초기화
    abandon_rate = math.nan
    time_on_task_total_ms = 0.0
    consecutive_days = 0
    active_days = 0
    source_entropy = math.nan
    adaptive_offer = math.nan
    enter_b_count = 0
    erase_rate = math.nan
    undo_rate = math.nan
    explanation_adoption_rate = math.nan
    explanation_after_wrong_rate = math.nan
    explanation_time_per_problem_ms = math.nan
    media_play_rate = math.nan
    answer_change_rate = math.nan

    # --- KT2: 진입/제출, source, bundle 시간, question-bundle 매핑 ---
    kt2_path = kt2_dir / f"{user_id}.csv"
    qid_to_bundle: Dict[str, str] = {}
    if kt2_path.exists():
        _validate_cols(kt2_path, KT2_COLS)
        enter_b = 0
        submit_b = 0
        last_enter: Dict[str, int] = {}
        enter_days: List[datetime.date] = []
        source_counts = Counter()
        adaptive_enter = 0
        qid_bundle_counts: Dict[str, Counter] = defaultdict(Counter)
        last_respond_qid: Optional[str] = None

        for r in _iter_dict_rows(kt2_path):
            action = (r.get("action_type") or "").strip()
            item_id = (r.get("item_id") or "").strip()
            ts = _safe_int(r.get("timestamp") or "", default=-1)

            if action == "respond" and item_id.startswith("q"):
                last_respond_qid = item_id

            # 실제 번들(파트 3,4,6,7)만 번들 feature 계산에 사용
            if _is_real_bundle(item_id):
                if action == "enter":
                    enter_b += 1
                    if ts >= 0:
                        last_enter[item_id] = ts
                        d = _dates_from_ts_ms(ts)
                        if d is not None:
                            enter_days.append(d)
                    src = (r.get("source") or "").strip()
                    if src:
                        source_counts[src] += 1
                        if src == "adaptive_offer":
                            adaptive_enter += 1
                elif action == "submit":
                    submit_b += 1
                    if last_respond_qid:
                        qid_bundle_counts[last_respond_qid][item_id] += 1
                    if ts >= 0 and item_id in last_enter:
                        dt = ts - last_enter[item_id]
                        if dt >= 0:
                            time_on_task_total_ms += float(dt)
                        last_enter.pop(item_id, None)

        enter_b_count = enter_b
        if enter_b > 0:
            abandon_rate = 1.0 - (submit_b / enter_b)
            abandon_rate = max(0.0, min(1.0, float(abandon_rate)))
            adaptive_offer = adaptive_enter / enter_b
        if enter_days:
            active_days = len(set(enter_days))
            consecutive_days = _max_consecutive_days(enter_days)
        source_entropy = _entropy(source_counts) if source_counts else math.nan

        # question_id -> bundle_id 매핑 (최다 등장 bundle 기준)
        for qid, cnt in qid_bundle_counts.items():
            if not cnt:
                continue
            bundle, _ = cnt.most_common(1)[0]
            qid_to_bundle[qid] = bundle

    # --- KT2_pay_res: 답변 변경 횟수 ---
    kt2pr_path = kt2_pay_res_dir / f"{user_id}.csv"
    if kt2pr_path.exists():
        _validate_cols(kt2pr_path, KT2_PAY_RES_COLS)
        total_rc = 0.0
        for r in _iter_dict_rows(kt2pr_path):
            total_rc += _safe_float(r.get("response_change") or "0", default=0.0)
        if n_problems > 0:
            answer_change_rate = total_rc / n_problems

    # --- KT4: erase/undo/media, 해설 진입/체류, 오답 후 해설 진입 ---
    kt4_path = kt4_dir / f"{user_id}.csv"
    if kt4_path.exists():
        _validate_cols(kt4_path, KT4_COLS)
        erase_cnt = 0
        undo_cnt = 0
        media_cnt = 0
        expl_enter_cnt = 0
        last_enter_e: Dict[str, int] = {}
        expl_total_ms = 0.0
        expl_enter_times: Dict[str, List[int]] = defaultdict(list)  # e#### -> enter ts 리스트

        for r in _iter_dict_rows(kt4_path):
            action = (r.get("action_type") or "").strip()
            item_id = (r.get("item_id") or "").strip()
            ts = _safe_int(r.get("timestamp") or "", default=-1)

            if action in ("erase_choice", "eliminate_choice"):
                erase_cnt += 1
            elif action == "undo_erase_choice":
                undo_cnt += 1
            elif action in ("play_audio", "play_video"):
                media_cnt += 1

            if item_id.startswith("e"):
                if action == "enter":
                    expl_enter_cnt += 1
                    if ts >= 0:
                        last_enter_e[item_id] = ts
                        expl_enter_times[item_id].append(ts)
                elif action == "quit":
                    if ts >= 0 and item_id in last_enter_e:
                        dt = ts - last_enter_e[item_id]
                        if dt >= 0:
                            expl_total_ms += float(min(dt, dwell_clip_ms))
                        last_enter_e.pop(item_id, None)

        if n_problems > 0:
            erase_rate = erase_cnt / n_problems
            undo_rate = undo_cnt / n_problems
            media_play_rate = media_cnt / n_problems
            explanation_adoption_rate = expl_enter_cnt / n_problems
            explanation_time_per_problem_ms = expl_total_ms / n_problems

        # 오답 후 해설 진입 비율 (explanation_after_wrong_rate)
        if wrong_attempts:
            for k in expl_enter_times.keys():
                expl_enter_times[k].sort()
            import bisect

            adopted = 0
            total_wrong = len(wrong_attempts)
            for submit_ts, qid in wrong_attempts:
                bundle = qid_to_bundle.get(qid)
                # 실제 번들(파트 3,4,6,7)에만 해설 진입 매핑
                if not bundle or not _is_real_bundle(bundle):
                    continue
                expl_id = "e" + bundle[1:]
                times = expl_enter_times.get(expl_id)
                if not times:
                    continue
                lo = submit_ts
                hi = submit_ts + int(after_wrong_window_ms)
                j = bisect.bisect_left(times, lo)
                if j < len(times) and times[j] <= hi:
                    adopted += 1
            if total_wrong > 0:
                explanation_after_wrong_rate = adopted / total_wrong
                if _is_finite(explanation_after_wrong_rate):
                    explanation_after_wrong_rate = max(0.0, min(1.0, float(explanation_after_wrong_rate)))

    # 파생 feature
    accuracy_per_time = math.nan
    accuracy_per_problem = math.nan
    if _is_finite(accuracy):
        denom_t = _safe_log1p(time_on_task_total_ms)
        denom_n = _safe_log1p(float(n_problems))
        if _is_finite(denom_t) and denom_t > 0:
            accuracy_per_time = accuracy / denom_t
        if _is_finite(denom_n) and denom_n > 0:
            accuracy_per_problem = accuracy / denom_n

    return StudentFeaturesRaw(
        user_id=user_id,
        accuracy=accuracy,
        avg_response_time_ms=avg_response_time_ms,
        n_problems=n_problems,
        abandon_rate=abandon_rate,
        time_on_task_total_ms=time_on_task_total_ms,
        consecutive_days=consecutive_days,
        active_days=active_days,
        source_entropy=source_entropy,
        adaptive_offer=adaptive_offer,
        enter_b_count=enter_b_count,
        erase_rate=erase_rate,
        undo_rate=undo_rate,
        explanation_adoption_rate=explanation_adoption_rate,
        explanation_after_wrong_rate=explanation_after_wrong_rate,
        explanation_time_per_problem_ms=explanation_time_per_problem_ms,
        media_play_rate=media_play_rate,
        answer_change_rate=answer_change_rate,
        accuracy_per_time=accuracy_per_time,
        accuracy_per_problem=accuracy_per_problem,
    )


def _apply_recommended_transforms(rows: List[Dict[str, object]]) -> Tuple[List[Dict[str, object]], Dict[str, float]]:
    """
    features.csv의 권장 처리 로직을 반영한 변환을 적용한다.
    - avg_response_time_ms: log1p + p99 clip
    - n_problems: log1p
    - time_on_task_total_ms: log1p + p99 clip
    - answer_change_rate / explanation_time_per_problem_ms / erase_rate / undo_rate /
      media_play_rate / accuracy_per_*: p99 clip
    원본 값은 *_raw 컬럼으로 함께 보존한다.
    """
    cand: Dict[str, List[float]] = defaultdict(list)

    for r in rows:
        for c in (
            "avg_response_time_ms",
            "n_problems",
            "time_on_task_total_ms",
            "answer_change_rate",
            "explanation_time_per_problem_ms",
            "erase_rate",
            "undo_rate",
            "media_play_rate",
        ):
            r[f"{c}_raw"] = r.get(c)

        art = _safe_float(str(r.get("avg_response_time_ms") or "nan"))
        art_t = _safe_log1p(art)
        if _is_finite(art_t):
            cand["avg_response_time_ms"].append(art_t)

        tot = _safe_float(str(r.get("time_on_task_total_ms") or "nan"))
        tot_t = _safe_log1p(tot)
        if _is_finite(tot_t):
            cand["time_on_task_total_ms"].append(tot_t)

        for c in (
            "answer_change_rate",
            "explanation_time_per_problem_ms",
            "erase_rate",
            "undo_rate",
            "media_play_rate",
            "accuracy_per_time",
            "accuracy_per_problem",
        ):
            v = _safe_float(str(r.get(c) or "nan"))
            if _is_finite(v):
                cand[c].append(v)

    thresholds: Dict[str, float] = {}
    for k, vals in cand.items():
        thresholds[k] = _quantile(vals, 0.99) if vals else math.nan

    for r in rows:
        v = _safe_float(str(r.get("avg_response_time_ms") or "nan"))
        v = _safe_log1p(v)
        v = _clip_upper(v, thresholds.get("avg_response_time_ms", math.nan))
        r["avg_response_time_ms"] = v

        npb = _safe_float(str(r.get("n_problems") or "nan"))
        r["n_problems"] = _safe_log1p(npb)

        tot = _safe_float(str(r.get("time_on_task_total_ms") or "nan"))
        tot = _safe_log1p(tot)
        tot = _clip_upper(tot, thresholds.get("time_on_task_total_ms", math.nan))
        r["time_on_task_total_ms"] = tot

        for c in (
            "answer_change_rate",
            "explanation_time_per_problem_ms",
            "erase_rate",
            "undo_rate",
            "media_play_rate",
            "accuracy_per_time",
            "accuracy_per_problem",
        ):
            vv = _safe_float(str(r.get(c) or "nan"))
            r[c] = _clip_upper(vv, thresholds.get(c, math.nan))

    return rows, thresholds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", type=str, default=str(DATASET_DEFAULT))
    ap.add_argument("--out-dir", type=str, default=str(OUT_DEFAULT))
    ap.add_argument("--limit-users", type=int, default=0, help="0이면 전체. >0이면 상위 N명만 처리(디버깅).")
    ap.add_argument("--prefer-existing-theta", action="store_true", default=True)
    ap.add_argument("--no-prefer-existing-theta", dest="prefer_existing_theta", action="store_false")
    ap.add_argument("--dwell-clip-ms", type=int, default=DWELL_CLIP_MS_DEFAULT)
    ap.add_argument("--after-wrong-window-ms", type=int, default=AFTER_WRONG_WINDOW_MS_DEFAULT)
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    kt1_dir = dataset_root / "KT1"
    kt2_dir = dataset_root / "KT2"
    kt2pr_dir = dataset_root / "KT2_pay_res"
    kt4_dir = dataset_root / "KT4"

    if not kt1_dir.exists():
        raise FileNotFoundError(f"KT1 dir not found: {kt1_dir}")

    theta_map = _load_theta_map(dataset_root, prefer_existing=args.prefer_existing_theta)

    kt1_files = sorted(kt1_dir.glob("u*.csv"))
    if args.limit_users and args.limit_users > 0:
        kt1_files = kt1_files[: args.limit_users]

    records: List[Dict[str, object]] = []
    for idx, p in enumerate(kt1_files, start=1):
        user_id = p.stem
        try:
            f = compute_student_features(
                user_id=user_id,
                kt1_path=p,
                kt2_dir=kt2_dir,
                kt2_pay_res_dir=kt2pr_dir,
                kt4_dir=kt4_dir,
                dwell_clip_ms=args.dwell_clip_ms,
                after_wrong_window_ms=args.after_wrong_window_ms,
            )
            d = f.__dict__.copy()
            d["user_id"] = user_id
            d["theta"] = theta_map.get(user_id, math.nan)
            records.append(d)
        except Exception as e:
            print(f"[WARN] failed user {user_id}: {e}")
        if idx % 500 == 0:
            print(f"processed {idx}/{len(kt1_files)} users...")

    # theta_z 계산
    thetas = [_safe_float(str(r.get("theta") or "nan")) for r in records]
    finite_thetas = [t for t in thetas if _is_finite(t)]
    t_mean = _mean(finite_thetas) if finite_thetas else math.nan
    t_std = math.sqrt(_mean([(t - t_mean) ** 2 for t in finite_thetas])) if finite_thetas and _is_finite(t_mean) else math.nan
    for r in records:
        th = _safe_float(str(r.get("theta") or "nan"))
        if _is_finite(th) and _is_finite(t_mean) and _is_finite(t_std) and t_std > 0:
            r["theta_z"] = (th - t_mean) / t_std
        else:
            r["theta_z"] = math.nan

    # 권장 처리/clip 적용
    records, thresholds = _apply_recommended_transforms(records)

    out_features = out_dir / "student_features.csv"
    if records:
        fieldnames = list(records[0].keys())
        with out_features.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            for r in records:
                w.writerow(r)

    print("Done.")
    print(f"- features: {out_features}")


if __name__ == "__main__":
    main()

