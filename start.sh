#!/usr/bin/env bash
set -e

echo "🔍 Prüfe Modell..."
if [ ! -f "model/logreg_pipeline.joblib" ]; then
  echo "📦 Modell nicht gefunden -> trainiere Modell..."
  python train_model.py
else
  echo "✅ Modell gefunden -> überspringe Training."
fi

echo "🚀 Starte Streamlit..."
streamlit run dashboard.py --server.port=8501 --server.address=0.0.0.0
