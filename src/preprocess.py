import re
import nltk
from nltk.corpus import stopwords

nltk.download("stopwords", quiet=True)

# Mantém negações, que são importantes para o sentimento
_NEGACOES = {"não", "nao", "nem", "nunca", "sem", "jamais"}
STOP = set(stopwords.words("portuguese")) - _NEGACOES


def limpar_texto(texto: str) -> str:
    texto = str(texto).lower()
    texto = re.sub(r"http\S+|www\.\S+", " ", texto)   # links
    texto = re.sub(r"@\w+", " ", texto)               # menções
    texto = re.sub(r"[^a-záàâãéêíóôõúç\s]", " ", texto)  # só letras
    palavras = [p for p in texto.split() if p not in STOP and len(p) > 1]
    return " ".join(palavras)