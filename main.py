import streamlit as st
import os
import joblib
import pandas as pd

@st.cache_resource
def load_artifacts():
    # Modell laden:
    model_files = ["xgb_credit_model.pkl", "extre_trees_credit_model.pkl"]
    model = None
    used_model_path = None
    for mf in model_files:
        if os.path.exists(mf):
            model = joblib.load(mf)
            used_model_path = mf
        break
    if model is None:
        st.error("Kein Modell gefunden, lege z.B. eine 'xgb_credit_model.pkl' Datei an.")
        st.stop()
    
    # Feature Reihenfolge bestimmen:
    feature_names_path = "feature_names.pkl"
    if os.path.exists(feature_names_path):
        feature_order = joblib.load(feature_names_path)
    else:
        st.warning("Keine Feature-Reihenfolge Datei gefunden!")
        st.stop()
    
    # Encoder laden:
    cat_cols = ["Sex", "Housing", "Saving accounts", "Checking account", "Purpose"]
    encoders = {}
    missing = []
    for col in cat_cols:
        pkl = f"{col}_encoder.pkl"
        if os.path.exists(pkl):
            encoders[col] = joblib.load(pkl)
        else:
            missing.append(col)
    if missing:
        st.warning(f"Fehlende Encoder: {missing} - App funktioniert nicht!")
        st.stop()
    
    return model, used_model_path, encoders, feature_order

def options(col):
    enc = ENCODERS.get(col)
    return list(getattr(enc, "classes_")) if enc is not None else []

def encode_value(col, value):
    enc = ENCODERS.get(col)
    if enc is None:
        return value
    return enc.transform([value])[0]

# ------------------------------ #
# MAIN
# ------------------------------ #
# Grundlägende Konfiguration:
st.set_page_config(page_title="Credit Risk Prediction", page_icon="💳" ,layout="wide")
st.title("Credit Risk Prediction")
st.caption("Demo für den German Credit Data Datensatz")

# Modell & Encoders laden:
with st.spinner("Lade ML-Model..."):
    model, user_model_path, ENCODERS, FEATURE_ORDER =  load_artifacts() # (x, y, z), Tupel entpacken
if "model_toast_done" not in st.session_state:
    st.toast("Modell erfolgreich geladen!")
    st.session_state["model_toast_done"] = True

# Sidebar Menü:
with st.sidebar:
    st.header("Eingaben")
    age = st.number_input("Age", min_value=18, max_value=80, value=30, step=1)
    sex = st.selectbox("Sex", options=options("Sex") or ["male", "female"])
    job = st.number_input("Job (0-3)", min_value=0, max_value=3, value=1, step=1)
    housing = st.selectbox("Housing", options=options("Housing"))
    saving_accounts = st.selectbox("Saving accounts", options=options("Saving accounts"))
    checking_account = st.selectbox("Checking account", options=options("Checking account"))
    purpose = st.selectbox("Purpose", options=options("Purpose"))
    credit_amount = st.number_input("Credit amount", min_value=0, value=1000, step=100)
    duration = st.number_input("Duration (months)", min_value=1, value=12, step=1)

# Eine modellkonforme Eingabe erzeugen:
row = {
    "Age": age,
    "Sex": encode_value("Sex", sex),
    "Job": job,
    "Housing": encode_value("Housing", housing),
    "Saving accounts": encode_value("Saving accounts", saving_accounts),
    "Checking account": encode_value("Checking account", checking_account),
    "Credit amount": credit_amount,
    "Duration": duration,
    "Purpose": encode_value("Purpose", purpose)
}
input_df = pd.DataFrame([row])
input_df = input_df[FEATURE_ORDER]

# Navigationsmenü erstellen
tab_pred, tab_whatif, tab_explain, tab_about = st.tabs(["Predictions", "What-if", "Erklärungen", "About"])

# Vorhersagen (Predictions-Menü):
with tab_pred:
    c1, c2 = st.columns([1, 1])
    with c1:
        st.subheader("Modell-konformes Input")
        st.dataframe(input_df)
    with c2:
        st.subheader("Vorhersage")
        if st.button("Predict risk"):
            X = input_df
            proba = model.predict_proba(X)[0]
            print(proba)
            p_bad = float(proba[1])
            is_bad = p_bad >= 0.5
            st.metric("Bad-Risiko", f"{p_bad:.1%}")
        
        


 
