import pandas as pd
import joblib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, ConfusionMatrixDisplay

from preprocess import limpar_texto

# ---- AJUSTAR AO TEU DATASET ----
CSV = "../data/reviews.csv"
COL_TEXTO = "review_comment_message"
COL_NOTA = "review_score"
# --------------------------------


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

print(df["label"].value_counts())  # vê o desequilíbrio das classes

X_train, X_test, y_train, y_test = train_test_split(
    df["texto_limpo"], df["label"],
    test_size=0.2, random_state=42, stratify=df["label"],
)

modelo = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), max_features=50000)),
    ("clf", LogisticRegression(max_iter=1000, class_weight="balanced")),
])
modelo.fit(X_train, y_train)

pred = modelo.predict(X_test)
print(classification_report(y_test, pred))

ConfusionMatrixDisplay.from_predictions(y_test, pred)
plt.savefig("../docs/matriz_confusao_baseline.png", bbox_inches="tight")

joblib.dump(modelo, "../src/modelo_baseline.joblib")