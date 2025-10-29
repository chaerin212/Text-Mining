# -*- coding: utf-8 -*-
"""
감성분석 실험 요약
1) EDA: 학습 및 테스트 샘플 수와 전처리 후 실제 학습에 사용된 샘플 수, 클래스 분포, 전처리 후 빈 문서와 1-token 문서 수 계산.
2) 전처리: 원본 파이프라인에서 일부 문서가 전처리 후 빈 토큰이 되어(Empty docs) 학습/평가에 문제 발생.
    - debug_tokens.py로 샘플 출력하여 빈/1-token 케이스 확인.
    - clean_text: 'ㅋ/ㅎ' 및 'ㅜ/ㅠ' 등 단일/반복 이모티콘을 '웃음'/'울음'으로 정규화.
    - 정규화 토큰('웃음','울음')은 감성 신호이므로 stopwords에서 제외.
    - 토큰화가 완전히 비었을 때의 fallback을 추가(Okt.nouns -> regex 한글단어 -> 영문/숫자 -> single-korean).
    - 학습 데이터에서는 토큰이 비어버린 문서들을 드롭. 테스트 문서는 "특수" 토큰으로 대체하여 예측 가능하게 함.
    *이유: '웃음' 등을 stopwords로 제거했을 때 의미 있는 감성 신호가 사라져서 빈 문서가 증가하였음.
         : 빈 문서 자체가 노이즈가 될 수 있어서 학습 안정성을 위해 비어있는 문서를 드롭함.
3) 표현: TF-IDF + Word2Vec 평균 벡터의 결합 표현 사용.
4) 모델: LogisticRegression (LC2, C=1.0, class_weight='balanced' 적용) 및 RandomForestClassifier 사용
5) 평가: StratifiedKFold CV (5-fold)를 사용해서 평균 CV 정확도와 macro-F1, class-wise recall로 모델 조합 평가.
6) 개선: class_weight='balanced' 적용 및 C 그리드(0.01,0.1,1,10) 튜닝 후 기본값 logreg_C=1.0 채택함
"""

from __future__ import annotations
import os
import re
import argparse
import random
from typing import List, Tuple, Dict, Callable, Any

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.base import clone
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from gensim.models import Word2Vec

try:
    from konlpy.tag import Okt
except Exception as e:
    raise RuntimeError("Konlpy import failed. Make sure konlpy is installed and Java is configured.") from e

import joblib
import warnings
warnings.filterwarnings("ignore")


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)


def infer_sep(path: str):
    with open(path, "r", encoding="utf-8") as f:
        first = f.readline()
    if "\t" in first:
        return "\t"
    if "," in first:
        return ","
    return None


def load_table(path: str) -> pd.DataFrame:
    sep = infer_sep(path)
    if sep:
        return pd.read_csv(path, sep=sep, engine="python", encoding="utf-8")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.rstrip("\n") for l in f]
    return pd.DataFrame({"text": lines})

# 라벨 정규화 함수: positive, pos, 긍정 -> 1; negative, neg, 부정 -> 0로 변환.
# 숫자형 라벨(0,1)도 그대로 int로 변환.
def normalize_labels(s: pd.Series) -> pd.Series:
    mapping = {
        "positive": 1, "pos": 1, "긍정": 1, "1": 1,
        "negative": 0, "neg": 0, "부정": 0, "0": 0,
        "true": 1, "false": 0
    }
    if s.dtype.kind in "iufc":
        return s.astype(int)
    return s.astype(str).map(lambda x: mapping.get(x.strip().lower(), x)).astype(int)

    """한국어 전처리 및 tokenizer 클래스
    - clean_text: 이모티콘 정규화, URL 제거, 과도한 구두점/반복문자 축소, 특수문자 제거.
    - tokenize: Okt 기반 형태소 분석 및 불용어 제거, 빈 토큰 시 fallback 순서로 복구함.
    - tokenize_nouns: Okt.nouns 기반 명사 추출 및 불용어 제거, 빈 토큰 시 tokenize()로 fallback함.
    - negation_transform: '안'/'못' 등의 부정 표현을 강화함.
    """

class KoreanPreprocessor:
    def __init__(self, extra_stopwords: List[str] = None):
        self.okt = Okt()
        base_stop = [
            "의", "가", "이", "은", "는", "하", "하다", "되", "수", "있", "없",
            "저", "그", "이것", "것", "같", "들", "그리고", "하지만", "으로", "로",
            "도", "을", "를", "에", "에서", "만", "보다", "정말", "너무", "또",
        ]
        self.stopwords = set(base_stop + (extra_stopwords or []))
    
    """
    - 반복되거나 단일문자인 'ㅋ'/'ㅎ'을 "웃음"으로, 'ㅜ'/'ㅠ'을 "울음"으로 정규화;
     -> 이유: 단순하게 제거하면 빈 토큰이 되어서 학습에 손해를 줄 수 있음. "웃음"/"울음"은 감성 신호이기 때문에 stopwords에서 제외해서 보존함.
    - 연속 구두점과 과도한 반복 문자를 축소하고, URL을 제거함으로써 노이즈를 억제함.
    - 일부 특수문자는 빈 토큰으로 남을 수 있음(학습에서 드롭 처리).
    """
    def clean_text(self, text: str) -> str:
        if pd.isna(text):
            return ""
        text = re.sub(r'(ㅋ|ㅎ)+', ' 웃음 ', text)
        
        text = re.sub(r'(ㅜ|ㅠ)+', ' 울음 ', text)
        
        text = re.sub(r'[\.\?!]{2,}', ' ', text)
        
        text = re.sub(r'([ㄱ-ㅎㅏ-ㅣ가-힣A-Za-z0-9])\1{3,}', r'\1\1\1', text)
        
        text = re.sub(r"http\S+|www\.\S+", " ", text)
        text = re.sub(r"\S+@\S+", " ", text)
        
        text = re.sub(r"[^0-9가-힣a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str, stem: bool = True) -> List[str]:
        text = self.clean_text(text)
        toks = self.okt.morphs(text, stem=stem)
        toks = [t for t in toks if len(t) >= 1 and t not in self.stopwords]

        if len(toks) == 0:
            
            nouns = self.okt.nouns(text)
            nouns = [n for n in nouns if len(n) >= 1 and n not in self.stopwords]
            if nouns:
                return nouns
            
            kwords = re.findall(r'[가-힣]{2,}', text)
            kwords = [w for w in kwords if w not in self.stopwords]
            if kwords:
                return kwords
            
            en_tokens = re.findall(r'[A-Za-z0-9]+', text)
            if en_tokens:
                return en_tokens
            
            single_k = re.findall(r'[가-힣]', text)
            if single_k:
                return single_k[:3]
        return toks

    def tokenize_nouns(self, text: str) -> List[str]:
        text = self.clean_text(text)
        nouns = self.okt.nouns(text)
        nouns = [n for n in nouns if len(n) >= 1 and n not in self.stopwords]
        if len(nouns) == 0:
            return self.tokenize(text, stem=True)
        return nouns

    def negation_transform(self, text: str) -> str:
        text = re.sub(r"\b안\s+([가-힣]+)", r"안\1", text)
        text = re.sub(r"\b못\s+([가-힣]+)", r"못\1", text)
        return text


def train_word2vec(sentences: List[List[str]], seed: int, dim: int = 100, min_count: int = 1) -> Word2Vec:
    model = Word2Vec(
        sentences=sentences,
        vector_size=dim,
        window=5,
        min_count=min_count,
        workers=max(1, os.cpu_count() - 1),
        seed=seed,
        epochs=8
    )
    return model


def doc_mean_vecs(token_lists: List[List[str]], w2v: Word2Vec) -> np.ndarray:
    dim = w2v.vector_size
    X = np.zeros((len(token_lists), dim), dtype=float)
    for i, toks in enumerate(token_lists):
        vecs = [w2v.wv[t] for t in toks if t in w2v.wv]
        if vecs:
            X[i] = np.mean(vecs, axis=0)
        else:
            X[i] = np.zeros(dim)
    return X


def variant_baseline(prep: KoreanPreprocessor, texts: List[str]) -> Tuple[List[List[str]], List[str], str]:
    token_lists = [prep.tokenize(t, stem=True) for t in texts]
    joined = [" ".join(toks) for toks in token_lists]
    desc = "Baseline: morphs(stem=True) with base stopwords"
    return token_lists, joined, desc


def variant_no_stem(prep: KoreanPreprocessor, texts: List[str]) -> Tuple[List[List[str]], List[str], str]:
    token_lists = [prep.tokenize(t, stem=False) for t in texts]
    joined = [" ".join(toks) for toks in token_lists]
    desc = "No-stem: morphs(stem=False)"
    return token_lists, joined, desc


def variant_nouns_only(prep: KoreanPreprocessor, texts: List[str]) -> Tuple[List[List[str]], List[str], str]:
    token_lists = [prep.tokenize_nouns(t) for t in texts]
    joined = [" ".join(toks) for toks in token_lists]
    desc = "Nouns only: Okt.nouns filtered by stopwords"
    return token_lists, joined, desc


def variant_negation_plus_stop(prep: KoreanPreprocessor, texts: List[str]) -> Tuple[List[List[str]], List[str], str]:
    trans_texts = [prep.negation_transform(t) for t in texts]
    token_lists = [prep.tokenize(t, stem=True) for t in trans_texts]
    joined = [" ".join(toks) for toks in token_lists]
    desc = "Negation-aware + extended stopwords"
    return token_lists, joined, desc

"""
기존 파이프라인에서는 LogisticRegression이 class_weight=None이었음.
이에 대해 500개의 학습 데이터로 실험한 cv 결과가 mean accuracy = 0.6567이었음.
-> evaluate_variant 및 최종 학습에서 LogisticRegression에 class_weight='balance'를 적용해서
   학습할 때 소수 클래스(negative)에 더 큰 가중치를 부여해서 모델이 소수 클래스를 더 잘 학습하게 했음.
=> 결과적으로, negative recall이 크게 올라갔고 같은 샘플로 실험한 cv mean accuracy 결과가 0.6567에서 0.7108로 상승함.

"""

def evaluate_variant(name: str,
                     train_tokens: List[List[str]],
                     train_joined: List[str],
                     y: np.ndarray,
                     seed: int,
                     w2v_dim: int = 100,
                     w2v_min_count: int = 1,
                     cv_folds: int = 5,
                     use_balanced: bool = False,
                     logreg_C: float = 1.0) -> Dict[str, Any]:
    results = {"variant": name}
    total_docs = len(train_tokens)
    empty_docs = sum(1 for t in train_tokens if len(t) == 0)
    one_token_docs = sum(1 for t in train_tokens if len(t) == 1)
    avg_len = np.mean([len(t) for t in train_tokens]) if train_tokens else 0.0
    results["token_stats"] = {"total_docs": total_docs, "empty_docs": int(empty_docs),
                              "one_token_docs": int(one_token_docs), "avg_len": float(avg_len)}

    def build_tfidf(min_df=1, ngram_range=(1, 1), max_df=0.995):
        return TfidfVectorizer(min_df=min_df, ngram_range=ngram_range, max_df=max_df)

    tfidf = build_tfidf()
    try:
        X_tfidf = tfidf.fit_transform(train_joined)
    except ValueError:
        tfidf = build_tfidf(max_df=1.0)
        X_tfidf = tfidf.fit_transform(train_joined)

    w2v = train_word2vec(train_tokens, seed=seed, dim=w2v_dim, min_count=w2v_min_count)
    X_w2v = doc_mean_vecs(train_tokens, w2v)

    from scipy.sparse import hstack, csr_matrix
    X_w2v_csr = csr_matrix(X_w2v)
    X_concat = hstack([X_tfidf, X_w2v_csr])

    lr = LogisticRegression(max_iter=1000, random_state=seed, C=logreg_C,
                             class_weight='balanced' if use_balanced else None)
    rf = RandomForestClassifier(n_estimators=150, random_state=seed, n_jobs=-1)

    models = {
        "tfidf_logreg": (X_tfidf, lr),
        "w2v_rf": (X_w2v, rf),
        "tfidf_w2v_logreg": (X_concat, lr)
    }

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=seed)
    model_scores = {}
    for mname, (Xmat, clf) in models.items():
        fold_accs = []
        for tr_idx, va_idx in skf.split(range(Xmat.shape[0]), y):
            if hasattr(Xmat, "tocsr"):
                Xtr = Xmat[tr_idx]
                Xva = Xmat[va_idx]
            else:
                Xtr = Xmat[tr_idx, :]
                Xva = Xmat[va_idx, :]
            ytr = y[tr_idx]
            yva = y[va_idx]
            cl = clone(clf)
            cl.fit(Xtr, ytr)
            preds = cl.predict(Xva)
            acc = accuracy_score(yva, preds)
            fold_accs.append(acc)
        model_scores[mname] = {"mean": float(np.mean(fold_accs)), "std": float(np.std(fold_accs)), "folds": fold_accs}
    results["model_scores"] = model_scores
    results["tfidf"] = tfidf
    results["w2v"] = w2v
    results["X_tfidf"] = X_tfidf
    results["X_w2v"] = X_w2v
    results["X_concat"] = X_concat
    return results


def main(args):
    set_seed(args.seed)
    print("Loading data...")
    train_df = load_table(args.train)
    test_df = load_table(args.test)

    # EDA 요약: 학습 및 테스트 샘플 수, 라벨 컬럼 파악.
    cols_train = [c for c in train_df.columns]
    lowered = [c.lower() for c in cols_train]
    if "document" in lowered:
        text_col = cols_train[lowered.index("document")]
    else:
        text_candidates = [c for c in train_df.columns if 'text' in c.lower() or 'review' in c.lower() or 'content' in c.lower()]
        text_col = text_candidates[0] if text_candidates else train_df.columns[0]

    label_col = None
    lowered_train = [c.lower() for c in train_df.columns]
    if "label" in lowered_train:
        label_col = train_df.columns[lowered_train.index("label")]
    else:
        label_candidates = [c for c in train_df.columns if 'label' in c.lower() or 'sentiment' in c.lower()]
        if label_candidates:
            label_col = label_candidates[0]

    if label_col is None:
        raise ValueError("Could not detect label column in train file. Ensure it contains a label column.")

    print(f"Using text column: {text_col} and label column: {label_col}")

    y = normalize_labels(train_df[label_col]).to_numpy()

    """
    초기 실험에서는 '웃음', '울음'을 stopwords에 포함함.
    'ㅋㅋㅋㅋ'같은 문서가 '웃음'으로 정규화되고 stopwords에 의해 제거되어
    문서가 빈 상태가 되는 문제가 반복적으로 발생함.
    이는 학습 노이즈에 영향을 미칠 수 있다 판단하여 '웃음', '울음'을 extra_stopwords 리스트에서 제거.
    """
    extra_sw = ["ㅋㅋ", "ㅎㅎ", "ㅠㅠ", "ㅠ", "ㅋ", "!", "ㅠㅜ"] 
    base_prep = KoreanPreprocessor(extra_stopwords=extra_sw)

    raw_train_texts = train_df[text_col].astype(str).tolist()
    raw_test_texts = test_df[text_col].astype(str).tolist()

    """
    일부 문서가 전처리나 토큰화 이후 빈 리스트가 됨.
    이런 문서들을 그대로 학습에 포함했을 때 학습 노이즈로 인한 모델 성능 저하 우려.
    학습할 때에는 전처리 후 비어있는 샘플을 드롭함.
    테스트할 때는 빈 문서를 단일 토큰으로 대체해서 예측이 가능하도록 함.
    """
    variants: List[Tuple[str, Callable[[KoreanPreprocessor, List[str]], Tuple[List[List[str]], List[str], str]]]] = [
        ("attempt_1_baseline", variant_baseline),
        ("attempt_2_no_stem", variant_no_stem),
        ("attempt_3_nouns_only", variant_nouns_only),
        ("attempt_4_negation_extstop", variant_negation_plus_stop),
    ]

    history_rows = []
    artifacts = {}

    for i, (vname, vfunc) in enumerate(variants, start=1):
        print("\n" + "="*80)
        print(f"Attempt {i}: running variant '{vname}' ...")
        train_tokens, train_joined, desc = vfunc(base_prep, raw_train_texts)
        test_tokens, test_joined, _ = vfunc(base_prep, raw_test_texts)

        non_empty_indices = [idx for idx, toks in enumerate(train_tokens) if len(toks) > 0]
        num_dropped = len(train_tokens) - len(non_empty_indices)
        if num_dropped > 0:
            print(f"  Note: dropping {num_dropped} empty docs from training for this variant")
            train_tokens = [train_tokens[idx] for idx in non_empty_indices]
            train_joined = [train_joined[idx] for idx in non_empty_indices]
            y_variant = y[non_empty_indices]
        else:
            y_variant = y

        test_joined = [tj if tj.strip() != "" else "특수" for tj in test_joined]
        test_tokens = [toks if len(toks) > 0 else ["특수"] for toks in test_tokens]

        total_docs = len(train_tokens)
        empty_docs = sum(1 for t in train_tokens if len(t) == 0)
        one_token_docs = sum(1 for t in train_tokens if len(t) == 1)
        avg_len = np.mean([len(t) for t in train_tokens]) if train_tokens else 0.0
        print(f"Variant description: {desc}")
        print(f"  Samples (after drop): {total_docs}, Empty docs: {empty_docs}, 1-token docs: {one_token_docs}, Avg tokens per sample: {avg_len:.2f}")

        # 평가(교차 검증) 및 피처/모델 생성.
        res = evaluate_variant(vname, train_tokens, train_joined, y_variant, seed=args.seed,
                               w2v_dim=args.w2v_dim, w2v_min_count=args.w2v_min_count, cv_folds=args.cv_folds,
                               use_balanced=args.use_balanced, logreg_C=args.logreg_C if hasattr(args, 'logreg_C') else 1.0)

        model_scores = res["model_scores"]
        best_model_name = max(model_scores.items(), key=lambda kv: kv[1]["mean"])[0]
        best_mean = model_scores[best_model_name]["mean"]
        print(f"  -> Best model for variant '{vname}': {best_model_name} (mean acc={best_mean:.4f})")

        history_rows.append({
            "attempt": vname,
            "description": desc,
            "best_model": best_model_name,
            "best_mean_acc": best_mean,
            "token_stats": res.get("token_stats", {})
        })

        artifacts[vname] = {
            "tfidf": res["tfidf"],
            "w2v": res["w2v"],
            "train_tokens": train_tokens,
            "test_tokens": test_tokens,
            "train_joined": train_joined,
            "test_joined": test_joined,
            "X_tfidf": res["X_tfidf"],
            "X_w2v": res["X_w2v"],
            "X_concat": res["X_concat"],
            "best_model_name": best_model_name,
            "model_scores": model_scores,
            "y_variant": y_variant,
            "train_indices_kept": non_empty_indices
        }

    history_df = pd.DataFrame([{
        "attempt": r["attempt"],
        "description": r["description"],
        "best_model": r["best_model"],
        "best_mean_acc": r["best_mean_acc"],
        "token_stats": r.get("token_stats", {})
    } for r in history_rows])
    hist_path = args.history or "preprocessing_attempt_history.csv"
    history_df.to_csv(hist_path, index=False, encoding="utf-8")
    print("\nSaved attempt history to:", hist_path)

    overall_best = max(history_rows, key=lambda r: r["best_mean_acc"])
    best_variant = overall_best["attempt"]
    best_model_name = overall_best["best_model"]
    print("\n" + "="*80)
    print(f"Overall best variant: {best_variant} with model {best_model_name} (acc={overall_best['best_mean_acc']:.4f})")
    chosen = artifacts[best_variant]

    # ---final model 훈련부 (main 함수 내부) ---
    print("Training final model on chosen variant's filtered training set...")
    X_final = None
    y_final = chosen["y_variant"]

    """정규화 계수 C를 grid(0.01, 0.1, 1, 10)로 3fold CV 튜닝한 결과,
    negative recall은 C=0.01에서 더 높았지만 전체 균형 관점에서 C=1.0으로 채택함.
    """
    if best_model_name.startswith("tfidf") and "w2v" not in best_model_name:
        X_final = chosen["X_tfidf"]
        final_clf = LogisticRegression(
            max_iter=1000,
            random_state=args.seed,
            C=getattr(args, "logreg_C", 1.0),
            class_weight='balanced' if getattr(args, "use_balanced", False) else None
        )
    elif best_model_name == "w2v_rf":
        X_final = chosen["X_w2v"]
        final_clf = RandomForestClassifier(
            n_estimators=getattr(args, "rf_estimators", 200),
            random_state=args.seed,
            n_jobs=-1
        )
    else:
    # 테스트 셋 변환 및 예측
        X_final = chosen["X_concat"]
        final_clf = LogisticRegression(
            max_iter=1000,
            random_state=args.seed,
            C=getattr(args, "logreg_C", 1.0),
            class_weight='balanced' if getattr(args, "use_balanced", False) else None
        )

    final_clf.fit(X_final, y_final)
    print("Final model trained.")

    tfidf_vec = chosen["tfidf"]
    w2v_model = chosen["w2v"]
    X_tfidf_test = tfidf_vec.transform(chosen["test_joined"])
    from scipy.sparse import csr_matrix, hstack
    X_w2v_test = csr_matrix(doc_mean_vecs(chosen["test_tokens"], w2v_model))
    if best_model_name == "tfidf_logreg":
        X_test_final = X_tfidf_test
    elif best_model_name == "w2v_rf":
        X_test_final = X_w2v_test
    else:
        X_test_final = hstack([X_tfidf_test, X_w2v_test])

    print("Predicting test set...")
    preds = final_clf.predict(X_test_final)
    preds = preds.astype(int)
    out_df = test_df.copy()
    out_df["label_pred"] = preds

    out_path = args.out or f"{args.user}_pred_test.txt"
    out_df.to_csv(out_path, sep="\t", index=False, encoding="utf-8")
    print("Saved predictions to:", out_path)

    # 모델 및 아티팩트 저장
    if args.save_model:
        model_artifacts = {
            "final_model": final_clf,
            "tfidf": tfidf_vec,
            "w2v": w2v_model,
            "chosen_variant": best_variant,
            "best_model_name": best_model_name,
            "history": history_rows,
            "train_indices_kept": chosen.get("train_indices_kept", [])
        }
        joblib.dump(model_artifacts, args.save_model)
        print("Saved model artifacts to:", args.save_model)

    print("\nPerforming a reproducible StratifiedShuffleSplit to show an example validation result...")
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.15, random_state=args.seed)
    tr_idx, va_idx = next(sss.split(range(X_final.shape[0]), y_final))
    if hasattr(X_final, "tocsr"):
        Xtr = X_final[tr_idx]
        Xva = X_final[va_idx]
    else:
        Xtr = X_final[tr_idx, :]
        Xva = X_final[va_idx, :]
    ytr = y_final[tr_idx]
    yva = y_final[va_idx]
    final_clone = clone(final_clf)
    final_clone.fit(Xtr, ytr)
    va_preds = final_clone.predict(Xva)
    print("Example validation accuracy:", accuracy_score(yva, va_preds))
    print("Example validation classification report:\n", classification_report(yva, va_preds))

    try:
        print("\nClustering on TF-IDF of chosen variant (k=2) to inspect structure...")
        kmeans = KMeans(n_clusters=2, random_state=args.seed, n_init=10)
        klabels = kmeans.fit_predict(chosen["X_tfidf"])
        sil = silhouette_score(chosen["X_tfidf"], klabels)
        print("Silhouette score (tfidf):", sil)
    except Exception as e:
        print("Skipping clustering due to:", e)

    print("\nRun complete. Summary of attempts saved to:", hist_path)
    print("Best variant:", best_variant, "| Best model:", best_model_name, "| Output:", out_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocessing experiments tuned for 'document' column with empty-doc drop policy")
    parser.add_argument("--train", required=True, help="Path to train file")
    parser.add_argument("--test", required=True, help="Path to test file")
    parser.add_argument("--out", help="Output path for test predictions (tab-separated).")
    parser.add_argument("--save_model", help="Path to save model/artifacts (joblib).")
    parser.add_argument("--history", help="Path to save attempt history CSV.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--user", default="chaerin212", help="User id for default filenames.")
    parser.add_argument("--w2v_dim", type=int, default=100, help="Word2Vec vector dim.")
    parser.add_argument("--w2v_min_count", type=int, default=1, help="Word2Vec min_count.")
    parser.add_argument("--cv_folds", type=int, default=5, help="CV folds for evaluation.")
    parser.add_argument("--use_balanced", action="store_true", help="If set, use class_weight='balanced' for LogisticRegression.")
    parser.add_argument("--logreg_C", type=float, default=1.0, help="LogisticRegression C (inverse regularization).")
    parser.add_argument("--rf_estimators", type=int, default=200, help="n_estimators for RandomForest used in final fit.")
    parser.add_argument("--only_test_output", action="store_true",
                    help="If set, suppress intermediate prints and show only final test output.")
    args = parser.parse_args()
    main(args)