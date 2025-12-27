# HR Kündigungsprognose

Dieses Projekt ist im Rahmen eines privaten Praxisprojekts (u. a. angelehnt an Inhalte eines Google-Zertifikatsprogramms) entstanden.  
Ziel ist es, mit Hilfe eines Machine-Learning-Modells die **Kündigungswahrscheinlichkeit** von Mitarbeitenden zu schätzen und die Ergebnisse in einem **interaktiven Dashboard** verständlich darzustellen.

---

## 🎯 Zielsetzung
- Frühzeitiges Erkennen von Mitarbeitenden mit erhöhtem Kündigungsrisiko (Frühwarnsystem)
- Transparente Erklärung der Modellentscheidung (Wahrscheinlichkeit + Schwellenwert)
- Visualisierung wichtiger Modellmetriken und Ergebnisse für HR-Entscheidungen

---

## 🧠 Modell & Daten
- **Modell:** Logistische Regression (Scikit-Learn)
- **Zielvariable:** `left` (1 = hat das Unternehmen verlassen, 0 = geblieben)
- **Features (u. a.):**
  - Zufriedenheit (`satisfaction_level`)
  - Letzte Beurteilung (`last_evaluation`)
  - Projekte (`number_project`)
  - Monatliche Stunden (`average_montly_hours`)
  - Jahre im Unternehmen (`time_spend_company`)
  - Arbeitsunfall (`Work_accident`)
  - Beförderung in letzten 5 Jahren (`promotion_last_5years`)
  - Abteilung (`Department`)
  - Gehalt (`salary`)

Im Training wird ein Preprocessing-Pipeline genutzt:
- One-Hot-Encoding für `Department`
- Ordinal-Encoding für `salary` (low < medium < high)

---

## 📊 Dashboard (Streamlit)
Das Streamlit-Dashboard ermöglicht:
- Anzeige von **Accuracy, Precision, Recall, F1-Score**
- **Confusion Matrix** zur Fehleranalyse
- **ROC Curve + AUC** (threshold-unabhängig)
- Interaktive **Schwellenwert-Steuerung** (Threshold)
- Testen eigener Eingaben inkl. Risiko-Ampel und erklärendem Text
- Optionales Logging der Eingaben/Prognosen in `data/prediction_log.csv`

---
