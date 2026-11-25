"""
Kenyan Farmer Query Router - Streamlit app.

Type a farmer question in English, Swahili, Sheng or a mix, and the app shows:
  * the fine-tuned mBERT category (with confidence for every class)
  * what base mBERT predicts, so the effect of local adaptation is visible

Run locally:   streamlit run streamlit_app.py
Deploy:        see DEPLOY.md (GitHub + Streamlit Community Cloud)
"""
import os
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import streamlit as st
import torch
from transformers import AutoModel, AutoModelForSequenceClassification, AutoTokenizer

from config import LABEL_DESCRIPTIONS, LABELS, MAX_LEN, MODEL_DIR, MODEL_NAME, RESULTS_DIR

MODEL_RELEASE_TAG = "model-v1"
MODEL_ASSET_NAME = "mbert-farmer-finetuned.zip"

LABEL_INFO = {
    "pests": ("Pests / Wadudu", "Crop protection officer"),
    "crop_disease": ("Crop disease / Magonjwa ya mimea", "Plant health officer"),
    "soil_fertilizer": ("Soil & fertilizer / Udongo na mbolea", "Soil fertility advisor"),
    "weather_climate": ("Weather & climate / Hali ya hewa", "Agro-meteorology desk"),
    "market_prices": ("Markets & prices / Soko na bei", "Market information officer"),
    "livestock": ("Livestock / Mifugo", "Veterinary / livestock officer"),
}

EXAMPLES = [
    "Mahindi yangu ina viwavi jeshi, nifanye nini?",
    "Madalali wanataka kununua gunia la maharagwe kwa bei ya kutupa.",
    "Ni mara ngapi nipe mbuzi dawa ya minyoo?",
    "Napima udongo wapi hapa Kakamega?",
    "Bro, ngamia yangu ina swelling kwa shingo.",
    "Mvua za vuli zimechelewa, nipande lini?",
]

torch.set_num_threads(max(1, (os.cpu_count() or 2)))


DEFAULT_REPO = "SamuelGithara/AI-Chatbot"   # change if you rename the repository


def _secret(name: str, default: str = "") -> str:
    """Read a secret from Streamlit secrets (top level or any [section]) or env vars."""
    try:
        if name in st.secrets:
            return str(st.secrets[name]).strip()
        for v in st.secrets.values():      # tolerate keys placed under a [section]
            if hasattr(v, "get") and v.get(name):
                return str(v[name]).strip()
    except Exception:  # no secrets.toml locally
        pass
    return os.environ.get(name, default).strip()


def _download(url: str, dest: Path, headers: dict) -> None:
    """Stream a large file to disk (keeps memory low on Streamlit Cloud)."""
    with requests.get(url, headers=headers, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(chunk_size=1 << 20):
                f.write(chunk)


def ensure_model() -> Path:
    """Use the local model folder if present, otherwise download it from the GitHub release."""
    if (MODEL_DIR / "model.safetensors").exists():
        return MODEL_DIR
    repo = _secret("GITHUB_REPO") or DEFAULT_REPO
    repo = repo.replace("https://github.com/", "").strip("/")
    token = _secret("GITHUB_TOKEN")
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    zpath = MODEL_DIR.parent / MODEL_ASSET_NAME
    with st.spinner("Downloading fine-tuned model (about 650 MB, first start only)..."):
        try:
            if token:   # private repository: go through the GitHub API
                h = {"Accept": "application/vnd.github+json", "Authorization": f"Bearer {token}"}
                rel = requests.get(f"https://api.github.com/repos/{repo}/releases/tags/{MODEL_RELEASE_TAG}",
                                   headers=h, timeout=30)
                rel.raise_for_status()
                asset = next(a for a in rel.json()["assets"] if a["name"] == MODEL_ASSET_NAME)
                _download(asset["url"], zpath, {**h, "Accept": "application/octet-stream"})
            else:       # public repository: direct release link
                _download(f"https://github.com/{repo}/releases/download/{MODEL_RELEASE_TAG}/{MODEL_ASSET_NAME}",
                          zpath, {})
        except Exception as e:
            st.error(f"Could not download the model from GitHub repo '{repo}' "
                     f"(release '{MODEL_RELEASE_TAG}'). Check GITHUB_REPO / GITHUB_TOKEN. Details: {e}")
            st.stop()
        with zipfile.ZipFile(zpath) as z:
            z.extractall(MODEL_DIR)
        zpath.unlink(missing_ok=True)
    return MODEL_DIR


@st.cache_resource(show_spinner="Loading models...")
def load_models():
    path = ensure_model()
    ft_tok = AutoTokenizer.from_pretrained(str(path))
    ft = AutoModelForSequenceClassification.from_pretrained(str(path)).eval()
    base_tok = AutoTokenizer.from_pretrained(MODEL_NAME)
    base = AutoModel.from_pretrained(MODEL_NAME).eval()
    head = np.load(RESULTS_DIR / "base_head.npz", allow_pickle=True)
    label_emb = embed([LABEL_DESCRIPTIONS[l] for l in LABELS], base_tok, base)
    return ft_tok, ft, base_tok, base, head, label_emb


@torch.no_grad()
def embed(texts, tok, model):
    enc = tok(texts, truncation=True, padding=True, max_length=MAX_LEN, return_tensors="pt")
    h = model(**enc).last_hidden_state
    m = enc["attention_mask"].unsqueeze(-1).float()
    e = ((h * m).sum(1) / m.sum(1)).numpy()
    return e


@torch.no_grad()
def classify(text):
    ft_tok, ft, base_tok, base, head, label_emb = load_models()
    enc = ft_tok([text], truncation=True, max_length=MAX_LEN, return_tensors="pt")
    probs = ft(**enc).logits.softmax(-1)[0].numpy()
    ft_labels = [ft.config.id2label[i] for i in range(len(probs))]

    e = embed([text], base_tok, base)
    scores = e @ head["coef"].T + head["intercept"]
    base_label = str(head["classes"][int(scores.argmax())])
    cos = (e / np.linalg.norm(e)) @ (label_emb / np.linalg.norm(label_emb, axis=1, keepdims=True)).T
    zs_label = LABELS[int(cos.argmax())]
    return dict(zip(ft_labels, probs)), base_label, zs_label


# ---------------- UI ----------------
st.set_page_config(page_title="Kenyan Farmer Query Router", layout="centered")
st.title("Kenyan Farmer Query Router")
st.caption("Fine-tuned multilingual BERT that routes farmer questions in English, Swahili, "
           "Sheng or a mix to the right advisory desk.")

if "query" not in st.session_state:
    st.session_state.query = EXAMPLES[0]

st.write("Try an example:")
cols = st.columns(2)
for i, ex in enumerate(EXAMPLES):
    if cols[i % 2].button(ex, key=f"ex{i}", use_container_width=True):
        st.session_state.query = ex

query = st.text_area("Farmer question", key="query", height=90,
                     placeholder="e.g. Kuku wangu wanakufa ghafla, nifanye nini?")

if st.button("Classify", type="primary") and query.strip():
    probs, base_label, zs_label = classify(query.strip())
    best = max(probs, key=probs.get)
    name, desk = LABEL_INFO[best]
    st.success(f"**{name}**  ·  confidence {probs[best]:.0%}\n\nRoute to: {desk}")

    df = pd.DataFrame({"Category": [LABEL_INFO[l][0] for l in probs],
                       "Probability": list(probs.values())}).sort_values("Probability")
    st.bar_chart(df, x="Category", y="Probability", horizontal=True, height=260)

    st.subheader("Base model vs fine-tuned")
    def mark(l):
        return f"{LABEL_INFO[l][0]} ({'agrees' if l == best else 'differs'})"
    st.table(pd.DataFrame({
        "Model": ["Base mBERT (zero-shot)", "Base mBERT (frozen + LR head)", "Fine-tuned mBERT"],
        "Prediction": [mark(zs_label), mark(base_label), LABEL_INFO[best][0]],
    }).set_index("Model"))

with st.expander("About this model"):
    st.markdown(
        "- **Model:** `bert-base-multilingual-cased`, fully fine-tuned on 420 labelled Kenyan farmer queries\n"
        "- **Test set (90 queries):** fine-tuned accuracy 0.72 vs 0.33 for zero-shot base mBERT "
        "and 0.64 for frozen base mBERT\n"
        "- **Labels:** pests, crop disease, soil and fertilizer, weather and climate, markets and prices, livestock\n"
        "- Advisory only. Confirm serious crop or animal health issues with a local extension officer or vet."
    )
