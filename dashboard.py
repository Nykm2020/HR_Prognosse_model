from pathlib import Path
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc

st.set_page_config(page_title="HR Kündigungsprognose", layout="wide")
st.title("💼 HR Kündigungsprognose – Logistic Regression Dashboard")
st.markdown("Erkenne potenzielle Kündigungen, verstehe Einflussfaktoren und teste Eingaben interaktiv.")

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "HR_comma_sep.csv"
MODEL_PATH = BASE_DIR / "model" / "logreg_pipeline.joblib"
LOG_PATH = BASE_DIR / "data" / "prediction_log.csv"

if not MODEL_PATH.exists():
    st.error(f"❌ Modell nicht gefunden: {MODEL_PATH}\n\nBitte zuerst `python train_model.py` ausführen.")
    st.stop()

pipe = joblib.load(MODEL_PATH)

st.sidebar.title("⚙️ Schwellenwert")
threshold = st.sidebar.slider("Klassifikations-Schwellenwert", 0.1, 0.9, 0.5, 0.01)

st.sidebar.markdown("### ℹ️ Schwellenwert-Verhalten")
st.sidebar.markdown("""
- 🔽 **Niedriger Wert** (z. B. 0.3–0.4): mehr Kündiger erkannt (**hoher Recall**), mehr Fehlalarme  
- 🔼 **Höherer Wert** (z. B. 0.7): weniger Fehlalarme, aber mehr Kündiger werden übersehen
""")

if threshold < 0.5:
    st.sidebar.success("🟩 Frühwarnmodus (sensibel)")
elif threshold < 0.6:
    st.sidebar.info("🟨 Ausgewogene Balance")
else:
    st.sidebar.warning("🟥 Strenger Modus (konservativ)")

df = None
if DATA_PATH.exists():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
else:
    st.warning(f"CSV nicht gefunden: {DATA_PATH}. Metriken/Plots werden übersprungen.")

if df is not None and "left" in df.columns:
    X = df.drop(columns=["left"])
    y = df["left"].astype(int)

    y_proba_all = pipe.predict_proba(X)[:, 1]
    y_pred_all = (y_proba_all >= threshold).astype(int)

    report = classification_report(y, y_pred_all, output_dict=True, zero_division=0)

    st.markdown("---")
    st.markdown("### 📊 Modellmetriken (auf dem Datensatz)")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("🔎 Accuracy", f"{report['accuracy']:.2f}")
    c2.metric("🎯 F1 (Klasse 1)", f"{report['1']['f1-score']:.2f}")
    c3.metric("📌 Precision (Klasse 1)", f"{report['1']['precision']:.2f}")
    c4.metric("📈 Recall (Klasse 1)", f"{report['1']['recall']:.2f}")
    st.markdown("---")
    st.markdown("### ✍️ Eigene Eingabe testen")

    with st.form("eingabe_formular"):
        satisfaction_level = st.slider("Zufriedenheit (0–1)", 0.0, 1.0, 0.5, 0.01)
        last_evaluation = st.slider("Letzte Beurteilung (0–1)", 0.0, 1.0, 0.7, 0.01)
        number_project = st.slider("Anzahl Projekte", 1, 10, 3)
        average_montly_hours = st.slider("Monatliche Stunden", 80, 320, 160)
        time_spend_company = st.slider("Jahre im Unternehmen", 1, 10, 3)
        Work_accident = st.selectbox("Arbeitsunfall?", [0, 1])
        promotion_last_5years = st.selectbox("Beförderung in letzten 5 Jahren?", [0, 1])
        Department = st.selectbox("Abteilung", [
            "sales", "technical", "support", "IT", "product_mng", "marketing", "RandD", "accounting", "hr", "management"
        ])
        salary = st.selectbox("Gehalt", ["low", "medium", "high"])

        submit = st.form_submit_button("Vorhersage starten")

        if submit:
            row = pd.DataFrame([{
                "satisfaction_level": satisfaction_level,
                "last_evaluation": last_evaluation,
                "number_project": number_project,
                "average_montly_hours": average_montly_hours,
                "time_spend_company": time_spend_company,
                "Work_accident": Work_accident,
                "promotion_last_5years": promotion_last_5years,
                "Department": Department,
                "salary": salary,
            }])

            proba = float(pipe.predict_proba(row)[0][1])
            pred = int(proba >= threshold)

            st.subheader("📤 Prognose")
            st.write(
                f"**Vorhersage (Modell bei Schwelle {threshold:.2f}):** {'⚠️ Kündigt' if pred == 1 else '✅ Bleibt'}")
            st.progress(int(proba * 100))
            st.write(f"**Wahrscheinlichkeit für Kündigung:** {proba:.2%}")

            st.markdown(
                f"🧾 Die Kündigungswahrscheinlichkeit liegt bei **{proba:.2%}**. "
                f"Da der Schwellenwert bei **{threshold:.2f}** liegt, wird die Person als "
                f"**{'Kündigt' if pred == 1 else 'Bleibt'}** klassifiziert."
            )

            if proba < 0.4:
                st.success("🟢 Geringes Risiko – aktuell kein Handlungsbedarf.")
            elif proba < 0.7:
                st.info("🟡 Mittleres Risiko – beobachten und ggf. im Gespräch ansprechen.")
            else:
                st.error("🔴 Hohes Risiko – proaktiv reagieren, z. B. Feedbackgespräch prüfen.")

            with st.expander("ℹ️ Wie funktioniert die Prognose?"):
                st.markdown("""
    - Das Modell berechnet eine **Wahrscheinlichkeit** (Risiko-Score).
    - Die **Vorhersage** hängt vom **Schwellenwert** ab (Schieberegler links).
    - Die Ampel hilft bei der Einordnung – unabhängig vom Schwellenwert.
    """)

            LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
            log_entry = row.copy()
            log_entry["prediction"] = pred
            log_entry["proba"] = proba
            log_entry["threshold"] = threshold

            if LOG_PATH.exists():
                hist = pd.read_csv(LOG_PATH)
                hist = pd.concat([hist, log_entry], ignore_index=True)
            else:
                hist = log_entry

            hist.to_csv(LOG_PATH, index=False)
            st.success(f"✅ Eingabe gespeichert: {LOG_PATH}")

    st.markdown("---")
    st.markdown("### 🗂️ Geloggte Vorhersagen")

    if LOG_PATH.exists():
        history_df = pd.read_csv(LOG_PATH)
        st.dataframe(history_df, use_container_width=True)
    else:
        st.info("ℹ️ Noch keine Eingaben gespeichert.")

    st.markdown("---")
    st.markdown("### 📊 Visualisierungen")
    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.markdown("#### 📌 Confusion Matrix")
        cm = confusion_matrix(y, y_pred_all)
        fig_cm, ax_cm = plt.subplots()
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm)
        ax_cm.set_xlabel("Vorhergesagt")
        ax_cm.set_ylabel("Tatsächlich")
        st.pyplot(fig_cm)

        tn, fp, fn, tp = cm.ravel()
        st.markdown(f"""
- ✅ **TN:** {tn} — bleibt & korrekt erkannt  
- ❌ **FP:** {fp} — bleibt, aber als kündigt markiert  
- ❌ **FN:** {fn} — kündigt, aber übersehen  
- ✅ **TP:** {tp} — kündigt & korrekt erkannt
""")

    with col2:
        st.markdown("#### 📈 ROC Curve (threshold-unabhängig)")
        fpr, tpr, _ = roc_curve(y, y_proba_all)
        roc_auc = auc(fpr, tpr)
        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
        ax_roc.plot([0, 1], [0, 1], linestyle="--")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.set_title("ROC Curve")
        ax_roc.legend(loc="lower right")
        st.pyplot(fig_roc)
        st.caption("📈 ROC/AUC basieren auf Wahrscheinlichkeiten und ändern sich nicht durch den Schwellenwert.")


