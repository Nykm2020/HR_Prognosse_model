# HR Kündigungsprognose – Streamlit Dashboard (Logistische Regression)

Dieses Projekt ist ein privates Praxisprojekt (u. a. im Kontext eines Google-Zertifikatsprogramms).  
Ziel ist es, mit einem Machine-Learning-Modell die **Kündigungswahrscheinlichkeit** von Mitarbeitenden zu schätzen und die Ergebnisse in einem **interaktiven Streamlit-Dashboard** verständlich darzustellen.

---

## 🎯 Zielsetzung
- Frühzeitige Identifikation von Mitarbeitenden mit erhöhtem Kündigungsrisiko (**Frühwarnsystem**)
- Transparente Darstellung von **Wahrscheinlichkeit**, **Schwellenwert (Threshold)** und **Vorhersage**
- Visualisierung zentraler Modellmetriken (z. B. Confusion Matrix, ROC/AUC)

---

## 🧰 Technologien & Pakete
- **Python**
- **scikit-learn** (Logistische Regression, Metriken)
- **pandas / numpy** (Datenverarbeitung)
- **matplotlib / seaborn** (Visualisierungen)
- **Streamlit** (Dashboard / Frontend)
- **joblib** (Modell speichern & laden)

---

## 🧠 Daten & Modell
- **Datensatz:** `data/HR_comma_sep.csv`
- **Zielvariable:** `left`
  - `1` = Mitarbeitende haben das Unternehmen verlassen
  - `0` = Mitarbeitende sind geblieben
- **Preprocessing (Pipeline):**
  - One-Hot-Encoding für `Department`
  - Ordinal-Encoding für `salary` (`low < medium < high`)
- **Modell:** Logistische Regression (mit `class_weight="balanced"`), gespeichert als Pipeline:
  - `model/logreg_pipeline.joblib`

---

## 📊 Dashboard-Funktionen
- Anzeige von **Accuracy, Precision, Recall, F1-Score** (für Klasse „Kündigt“)
- **Confusion Matrix** inkl. TN/FP/FN/TP Erklärung
- **ROC Curve + AUC** (threshold-unabhängig)
- **Schwellenwert-Slider** zur Steuerung der Sensitivität (Recall vs. Precision)
- **Eingabeformular**: Manuelle Eingaben → Wahrscheinlichkeit + Vorhersage + Risiko-Ampel
- Optionales Logging von Vorhersagen nach: `data/prediction_log.csv`

---

## ✅ Beispiel-Ergebnis (abhängig vom Schwellenwert)
Ein typisches Beispiel bei **Threshold = 0.60**:
- Accuracy: **0.80**
- Precision: **0.44**
- Recall: **0.69**
- F1-Score: **0.54**
- ROC-AUC: **0.84**

> Hinweis: Der Threshold beeinflusst Precision/Recall stark.  
> Niedriger Threshold → mehr Kündiger erkannt (höherer Recall), aber mehr Fehlalarme.  
> Höherer Threshold → weniger Fehlalarme, aber mehr Kündiger werden übersehen.

---

## 📁 Projektstruktur (minimal)
```
HR_Kuendigungsprognose/
├─ data/
│  └─ HR_comma_sep.csv
├─ model/
│  └─ logreg_pipeline.joblib
├─ train_model.py
├─ dashboard.py
├─ requirements.txt
└─ README.md
```

---

## 🚀 Installation & Start

### 1) Abhängigkeiten installieren
```bash
pip install -r requirements.txt
```

### 2) CSV ablegen
Lege `HR_comma_sep.csv` in:
```
data/HR_comma_sep.csv
```

### 3) Modell trainieren
```bash
python train_model.py
```
Das Modell wird gespeichert unter:
```
model/logreg_pipeline.joblib
```

### 4) Dashboard starten
```bash
streamlit run dashboard.py
```
Streamlit öffnet üblicherweise:
- http://localhost:8501

---

## 🔎 Interpretation (kurz)
- **Wahrscheinlichkeit** = Risiko-Score (z. B. 54%)
- **Threshold** = Grenze, ab wann „Kündigt“ vorhergesagt wird
- **Vorhersage** = Ergebnis nach Threshold (z. B. bei Threshold 0.57 und Score 0.54 → „Bleibt“)

---

## 📌 Disclaimer
Dieses Projekt ist ein Lern-/Demo-Projekt. In realen HR-Szenarien sollten Vorhersagen immer zusammen mit fachlichem Kontext, HR-Prozessen und Datenschutz-/Ethik-Anforderungen interpretiert werden.
