import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, classification_report

from preprocess import limpar_texto

CSV = "../data/reviews.csv"
COL_TEXTO = "review_comment_message"
COL_NOTA = "review_score"


def nota_para_sentimento(n):
    if n <= 2:
        return "negativo"
    if n == 3:
        return "neutro"
    return "positivo"


df = pd.read_csv(CSV)
df = df[[COL_TEXTO, COL_NOTA]].dropna()
df["label"] = df[COL_NOTA].apply(nota_para_sentimento)
df["texto_limpo"] = df[COL_TEXTO].apply(limpar_texto)
df = df[df["texto_limpo"].str.len() > 0]

X_train, X_test, y_train, y_test = train_test_split(
    df["texto_limpo"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)

modelos = {
    "TF-IDF + Regressão Logística": LogisticRegression(max_iter=1000, class_weight="balanced"),
    "TF-IDF + SVM linear": LinearSVC(class_weight="balanced"),
}

linhas = []
for nome, clf in modelos.items():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
        ("clf", clf),
    ])
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    acc = accuracy_score(y_test, pred)
    f1 = f1_score(y_test, pred, average="macro")
    print("\n=====", nome, "=====")
    print(classification_report(y_test, pred))
    linhas.append(f"{nome}: accuracy={acc:.3f}, F1 macro={f1:.3f}")

print("\nCOMPARAÇÃO:")
for l in linhas:
    print(l)

with open("../docs/comparacao_modelos.txt", "w", encoding="utf-8") as f:
    f.write("\n".join(linhas))