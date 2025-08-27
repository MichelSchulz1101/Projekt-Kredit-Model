import pathlib
from typing import Dict, Any

import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.preprocessing import LabelEncoder

st.set_page_config(page_title="Credit Risk Scorer", layout="centered")


# -----------------------------
# Config
# -----------------------------
ARTIFACT_DIR = pathlib.Path("artifacts")
DEFAULT_MODEL_PATH = ARTIFACT_DIR / "extra_trees_credit_model.pkl"         
DEFAULT_TARGET_ENCODER = ARTIFACT_DIR / "target_encoder.pkl"    

ENCODER_PATHS = {
    "Sex": ARTIFACT_DIR / "Sex_encoder.pkl",
    "Housing": ARTIFACT_DIR / "Housing_encoder.pkl",
    "Saving accounts": ARTIFACT_DIR / "Saving accounts_encoder.pkl",
    "Checking account": ARTIFACT_DIR / "Checking account_encoder.pkl",
}


FEATURE_ORDER = [
    "Age", "Sex", "Job", "Housing", "Saving accounts", "Checking account", "Credit amount", "Duration"
]

NUM_FEATURES = ["Age", "Credit amount", "Duration", "Job"]
CAT_FEATURES = ["Sex", "Housing", "Saving accounts", "Checking account"]

# -----------------------------
# Laden der Artefakte
# -----------------------------
@st.cache_resource(show_spinner=False)
def load_model(path: pathlib.Path):
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_label_encoder(path: pathlib.Path) -> LabelEncoder:
    return joblib.load(path)

@st.cache_resource(show_spinner=False)
def load_all_encoders(paths: Dict[str, pathlib.Path]) -> Dict[str, LabelEncoder]:
    return {col: load_label_encoder(p) for col, p in paths.items()}

def safe_label_transform(encoder: LabelEncoder, value: Any, colname: str) -> int:
    try:
        return int(encoder.transform([value])[0])
    except Exception:
        classes = list(getattr(encoder, "classes_", []))
        if "Unknown" in classes:
            return int(encoder.transform(["Unknown"])[0])
        if classes:
            return int(encoder.transform([classes[0]])[0])
        raise ValueError(f"Encoder für {colname} hat keine Klassen.")

def preprocess_input(df_row: pd.Series, encoders: Dict[str, LabelEncoder]) -> pd.DataFrame:
    row = df_row.copy()

    # Numerisch erzwingen
    for f in NUM_FEATURES:
        row[f] = pd.to_numeric(row[f], errors="coerce")
    if row[NUM_FEATURES].isnull().any():
        missing = row[NUM_FEATURES].index[row[NUM_FEATURES].isnull()].tolist()
        raise ValueError(f"Numerische Eingaben fehlen/ungültig: {missing}")

    # Kategorische encodieren (nur die mit Encodern)
    for f in CAT_FEATURES:
        enc = encoders[f]
        row[f] = safe_label_transform(enc, row[f], f)

    # In Modell-Feature-Reihenfolge packen
    X = pd.DataFrame([row[FEATURE_ORDER].values], columns=FEATURE_ORDER)
    for f in CAT_FEATURES:
        X[f] = X[f].astype(int)
    return X

# -----------------------------
# Sidebar
# -----------------------------
st.sidebar.header("Artefakte")
model_path = st.sidebar.text_input("Modell-Pfad", value=str(DEFAULT_MODEL_PATH))
target_enc_path = st.sidebar.text_input("Target-Encoder-Pfad", value=str(DEFAULT_TARGET_ENCODER))

st.sidebar.markdown("**Encoder-Pfade (kategorische Spalten)**")
enc_paths_inputs = {col: st.sidebar.text_input(f"Encoder für {col}", value=str(path))
                    for col, path in ENCODER_PATHS.items()}

load_ok = True
try:
    model = load_model(pathlib.Path(model_path))
except Exception as e:
    load_ok = False
    st.sidebar.error(f"Modell konnte nicht geladen werden: {e}")

try:
    target_encoder = load_label_encoder(pathlib.Path(target_enc_path))
except Exception as e:
    load_ok = False
    st.sidebar.error(f"Target-Encoder konnte nicht geladen werden: {e}")

try:
    encoder_paths_actual = {c: pathlib.Path(p) for c, p in enc_paths_inputs.items()}
    encoders = load_all_encoders(encoder_paths_actual)
except Exception as e:
    load_ok = False
    st.sidebar.error(f"Kategorie-Encoder konnten nicht geladen werden: {e}")

# -----------------------------
# UI
# -----------------------------
st.title("Credit Risk Scorer")
st.caption("Diese App schätzt das Kreditrisiko (0 = bad, 1 = good) basierend auf deinen Eingaben. "
               "Das trainierte ML-Modell und die Encoder werden aus Dateien geladen.")

if not load_ok:
    st.stop()

st.subheader("Neue Kredit-Anfrage")

# Defaults
age = st.number_input("Age", min_value=18, max_value=80, value=35, step=1)
credit_amount = st.number_input("Credit amount", min_value=0, max_value=20000, value=5000, step=1000)
duration = st.number_input("Duration (Monate)", min_value=1, max_value=72, value=24, step=1)

# Klassen direkt aus Encodern lesen (zeigt Trainingklassen)
def encoder_classes(enc: LabelEncoder):
    return list(getattr(enc, "classes_", [])) or []

sex = st.selectbox("Sex", options=encoder_classes(encoders["Sex"]) or ["male", "female"])
job = st.number_input("Job (0–3)", min_value=0, max_value=3, value=1, step=1)
housing = st.selectbox("Housing", options=encoder_classes(encoders["Housing"]) or ["own", "rent", "free"])
saving = st.selectbox("Saving accounts", options=encoder_classes(encoders["Saving accounts"]) or ["little", "moderate", "rich", "quite rich", "unknown"])
checking = st.selectbox("Checking account", options=encoder_classes(encoders["Checking account"]) or ["little", "moderate", "rich", "unknown"])

row = pd.Series({
    "Age": age,
    "Credit amount": credit_amount,
    "Duration": duration,
    "Sex": sex,
    "Job": job,
    "Housing": housing,
    "Saving accounts": saving,
    "Checking account": checking,
})

st.divider()

col1, col2 = st.columns([1, 1])
with col1:
    predict_btn = st.button("Vorhersage starten", use_container_width=True)
with col2:
    proba_toggle = st.toggle("Wahrscheinlichkeit anzeigen", value=True)

if predict_btn:
    try:
        X = preprocess_input(row, encoders)
        y_pred = model.predict(X)

        proba = None
        if proba_toggle and hasattr(model, "predict_proba"):
            try:
                proba = model.predict_proba(X)
            except Exception:
                proba = None

        # Target zurückübersetzen 
        try:
            pred_label = target_encoder.inverse_transform(y_pred.astype(int))[0]
        except Exception:
            pred_label = {0: "bad", 1: "good"}.get(int(y_pred[0]), str(y_pred[0]))

        risk_numeric = int(y_pred[0])
        st.success(f"**Vorhersage:** {risk_numeric}  ({pred_label})")

        if proba is not None:
            prob_good = float(proba[0][1]) if proba.shape[1] > 1 else float(proba[0][0])
            prob_bad = 1.0 - prob_good
            st.metric(label="P(Risk = good)", value=f"{prob_good*100:.1f}%")
            st.progress(min(max(prob_good, 0.0), 1.0))

            with st.expander("Details"):
                st.json({
                    "Eingaben (roh)": row.to_dict(),
                    "Eingaben (modellready)": X.iloc[0].to_dict(),
                    "P(bad)": round(prob_bad, 4),
                    "P(good)": round(prob_good, 4),
                })

    except Exception as e:
        st.error(f"Vorhersage fehlgeschlagen: {e}")

