"""E0 baselines: majority class, and TF-IDF (word 1-2 grams plus char 2-5 grams) with logistic regression."""
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline, make_union

# sklearn's default token pattern treats Bangla vowel signs (Unicode categories Mc and Mn) as
# separators, which breaks every word apart: 'ভালো লাগলো খুব' becomes ['গল']. Keep the whole
# Bengali block (U+0980 to U+09FF) inside words instead.
BANGLA_WORD = r"[\wঀ-৿]+"


def build_model(name, cfg, seed):
    """An unfitted sklearn model for one E0 condition. `cfg` is that model's block in configs/e0.yaml."""
    if name == "majority":
        return DummyClassifier(strategy="most_frequent")
    if name == "tfidf_lr":
        features = make_union(
            TfidfVectorizer(token_pattern=BANGLA_WORD, ngram_range=tuple(cfg["word_ngrams"]),
                            sublinear_tf=cfg["sublinear_tf"]),
            TfidfVectorizer(analyzer=cfg["char_analyzer"], ngram_range=tuple(cfg["char_ngrams"]),
                            sublinear_tf=cfg["sublinear_tf"]),
        )
        return make_pipeline(features, LogisticRegression(C=cfg["C"], max_iter=cfg["max_iter"], random_state=seed))
    raise ValueError(f"unknown E0 model: {name}")
