"""
Gradescope submission generator: hybrid rules + ML.

1. Rules assign forced 0 (empty text, hard report / structure gates) or a score
   ``delta = pos - neg``.
2. If ``delta >= margin + HYBRID_CONFIDENCE_BAND`` -> 1; if
   ``delta <= margin - HYBRID_CONFIDENCE_BAND`` -> 0; else ML decides.
3. ML is trained on official seed + ``extra_labeled.csv`` (if present), else
   seed + first ``PRIVATE_TRAIN_MAX_ROWS`` rows of ``mimic_labeled_private.csv``.
   Word + char TF-IDF + LogisticRegression.

If no extra/private file exists, falls back to **rule-only** ``predict_text``.

Outputs: test01/02/03-pred.csv (row_id, prediction).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import FeatureUnion, Pipeline

DECISION_MARGIN = 0.60
HYBRID_CONFIDENCE_BAND = 1.2

PRIVATE_TRAIN_MAX_ROWS = 100


def read_text_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=None, engine="python")
    if "sentence" in df.columns and "text" not in df.columns:
        df = df.rename(columns={"sentence": "text"})
    if "text" not in df.columns:
        raise ValueError(f"Expected a text column in {path}")
    return df


def rule_pipeline(text: str) -> tuple[str, float]:
    """
    Returns ``('zero', 0.)`` for forced label 0, else ``('delta', pos - neg)``.
    """
    if not isinstance(text, str) or not text.strip():
        return "zero", 0.0

    t = text.lower()

    pos = 0.0
    neg = 0.0

    pos += 1.25

    positive = [
        "discharge diagnosis",
        "history of present illness",
        "active issues",
        "past medical history",
        "chief complaint",
        "hospital course",
        "followup instructions",
        "follow-up instructions",
        "admission date",
        "discharge date",
        "discharge instructions",
        "primary diagnosis",
        "secondary diagnosis",
        "diagnosis:",
        "condition:",
        "history:",
    ]

    negative_substrings = [
        "impression:",
        "findings:",
        "cta head",
        "cta neck",
        " cta ",
        "ct angiography",
        "ct ",
        "ultrasound",
        "arteriogram",
        "gram stain",
        "final report",
        "tablet sig:",
        "disp:",
        "vs on arrival",
        "portable chest",
    ]

    negative_regex = [
        (r"\bekg\b", 2.0),
        (r"\bcxr\b", 2.0),
        (r"\bmg/dl\b", 2.0),
        (r"\bmeq/l\b", 2.0),
        (r"\btablet\b", 2.0),
        (r"\bdose\b", 2.0),
        (r"exam:", 2.0),
    ]

    for w in positive:
        if w in t:
            pos += 2.0

    for w in negative_substrings:
        if w in t:
            neg += 2.0

    for pattern, w in negative_regex:
        if re.search(pattern, t):
            neg += w

    neg += 0.24 * min(t.count(":"), 55)
    neg += 0.14 * min(t.count("-"), 70)

    if len(t) > 900:
        pos += 1.0

    weak_positive = [
        "admitted for",
        "presented with",
        "came in with",
        "found to have",
        "treated with",
        "started on",
        "continued on",
        "discharged on",
        "status post",
        "s/p",
        "complicated by",
        "likely due to",
        "secondary to",
        "concerning for",
        "consistent with",
        "history of",
        "diagnosed with",
        "patient was admitted",
        "patient presented",
        "patient reports",
        "patient complains",
        "was found to have",
        "was treated with",
        "was started on",
        "was discharged",
        "will be discharged",
        "plan to",
        "recommend",
        "follow up",
        "follow-up",
        "continue",
        "improved with",
        "resolved",
        "stable for discharge",
    ]

    for w in weak_positive:
        if w in t:
            pos += 0.75

    strong_pos_sections = [
        "discharge diagnosis",
        "history of present illness",
        "hospital course",
        "past medical history",
        "assessment and plan",
        "brief hospital course",
        "chief complaint",
    ]

    hard_report_cues = [
        "impression:",
        "findings:",
        "final report",
        "portable chest",
        "ct ",
        "cta head",
        "cta neck",
        "ultrasound",
        "x-ray",
        "radiograph",
        "arteriogram",
        "exam:",
        "technique:",
        "comparison:",
        "indication:",
        "wet read",
        "preliminary report",
        "normal sinus rhythm",
        "ventricular rate",
        "blood pressure",
        "heart rate",
        "respiratory rate",
        "o2 sat",
        "spo2",
        "wbc",
        "hgb",
        "hct",
        "platelet",
        "sodium",
        "potassium",
        "chloride",
        "creatinine",
        "mg/dl",
        "meq/l",
    ]

    has_strong_pos = any(w in t for w in strong_pos_sections)
    has_report = any(w in t for w in hard_report_cues)

    report_structure_score = 0
    report_structure_score += t.count(":")
    report_structure_score += t.count("=")
    report_structure_score += len(re.findall(r"\b\d+(\.\d+)?\b", t)) * 0.25

    if has_report and not has_strong_pos:
        return "zero", 0.0

    if report_structure_score >= 10 and not has_strong_pos and len(t) < 1200:
        return "zero", 0.0

    return "delta", pos - neg


def predict_text(text: str, margin: float = DECISION_MARGIN) -> int:
    """Rule-only: same decision boundary as before hybrid banding."""
    kind, d = rule_pipeline(text)
    if kind == "zero":
        return 0
    return 1 if d >= margin else 0


def predict_hybrid(text: str, margin: float, ml_model: Pipeline) -> int:
    """Rules for confident + forced 0; ML only in the uncertain band."""
    kind, d = rule_pipeline(text)
    if kind == "zero":
        return 0
    if d >= margin + HYBRID_CONFIDENCE_BAND:
        return 1
    if d <= margin - HYBRID_CONFIDENCE_BAND:
        return 0
    return int(ml_model.predict([str(text)])[0])


def load_ml_training_data(base: Path, tests_dir: Path) -> pd.DataFrame | None:
    seed_path = tests_dir / "train_data-text_and_labels.csv"
    seed = read_text_csv(seed_path)
    if "label" not in seed.columns:
        raise ValueError(f"Expected label column in {seed_path}")
    seed = seed[["text", "label"]].copy()
    seed["label"] = seed["label"].astype(int)

    extra_path = base / "extra_labeled.csv"
    private_path = base / "mimic_labeled_private.csv"

    if extra_path.exists():
        add = read_text_csv(extra_path)[["text", "label"]].copy()
        add["label"] = add["label"].astype(int)
        src = "extra_labeled.csv"
    elif private_path.exists():
        add = read_text_csv(private_path)[["text", "label"]].head(PRIVATE_TRAIN_MAX_ROWS).copy()
        add["label"] = add["label"].astype(int)
        src = f"mimic_labeled_private.csv (first {PRIVATE_TRAIN_MAX_ROWS})"
    else:
        return None

    merged = pd.concat([seed, add], ignore_index=True)
    merged = merged.drop_duplicates(subset=["text"], keep="first")
    if merged["label"].nunique() < 2 or len(merged) < 12:
        print(f"ML training data from {src} too small or single-class; hybrid disabled.")
        return None
    print(f"Hybrid ML train: seed + {src} -> {len(merged)} rows after dedupe")
    return merged


def train_ml_model(merged: pd.DataFrame) -> Pipeline:
    X = merged["text"].fillna("").astype(str)
    y = merged["label"].astype(int)
    pipe = Pipeline(
        [
            (
                "features",
                FeatureUnion(
                    [
                        (
                            "word",
                            TfidfVectorizer(
                                analyzer="word",
                                ngram_range=(1, 2),
                                min_df=1,
                                sublinear_tf=True,
                                max_features=20000,
                            ),
                        ),
                        (
                            "char",
                            TfidfVectorizer(
                                analyzer="char_wb",
                                ngram_range=(3, 5),
                                min_df=1,
                                sublinear_tf=True,
                                max_features=20000,
                            ),
                        ),
                    ]
                ),
            ),
            (
                "clf",
                LogisticRegression(
                    C=1.0,
                    class_weight="balanced",
                    max_iter=5000,
                    solver="liblinear",
                ),
            ),
        ]
    )
    pipe.fit(X, y)
    return pipe


def write_gradescope_file(preds: np.ndarray, output_path: Path) -> None:
    out_df = pd.DataFrame(
        {
            "row_id": np.arange(len(preds), dtype=int),
            "prediction": preds.astype(int),
        }
    )
    out_df.to_csv(output_path, index=False)
    print(f"Saved {output_path.name} ({len(out_df)} rows)")


def main() -> None:
    base = Path(__file__).resolve().parent
    tests_dir = base / "from_zip"
    train_path = tests_dir / "train_data-text_and_labels.csv"

    train_df = read_text_csv(train_path)
    if "label" not in train_df.columns:
        raise ValueError(f"Expected label column in {train_path}")
    y = train_df["label"].astype(int)

    train_rule = np.array(
        [predict_text(x) for x in train_df["text"].fillna("").astype(str)],
        dtype=int,
    )
    print(f"Rule-only sanity on official train ({len(train_df)} rows):")
    print("  Acc:", accuracy_score(y, train_rule))
    print("  F1:", f1_score(y, train_rule))

    merged = load_ml_training_data(base, tests_dir)
    ml_model: Pipeline | None = None
    if merged is not None:
        ml_model = train_ml_model(merged)
        hyb = np.array(
            [
                predict_hybrid(x, DECISION_MARGIN, ml_model)
                for x in train_df["text"].fillna("").astype(str)
            ],
            dtype=int,
        )
        print("Hybrid (margin=DECISION_MARGIN) on official train:")
        print("  Acc:", accuracy_score(y, hyb))
        print("  F1:", f1_score(y, hyb))

    test_map = {
        "test01_text_only.csv": ("test01-pred.csv", 0.60),
        "test02_text_only.csv": ("test02-pred.csv", 1.00),
        "test03_text_only.csv": ("test03-pred.csv", 0.85),
    }

    for input_name, (output_name, margin) in test_map.items():
        test_df = read_text_csv(tests_dir / input_name)
        X_test = test_df["text"].fillna("").astype(str)
        if ml_model is not None:
            preds = np.array(
                [predict_hybrid(x, margin, ml_model) for x in X_test], dtype=int
            )
            mode = "hybrid"
        else:
            preds = np.array(
                [predict_text(x, margin=margin) for x in X_test], dtype=int
            )
            mode = "rule-only"
        write_gradescope_file(preds, base / output_name)
        print(f"    {output_name} ({mode}, margin={margin})")


if __name__ == "__main__":
    main()
