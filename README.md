# Heart Disease Prediction ML System

Proyek ini membangun sistem machine learning untuk prediksi penyakit jantung menggunakan dataset terproses dan model Logistic Regression serta SVC. Tracking eksperimen dilakukan dengan MLflow dan DagsHub.

## Fitur Utama
- Preprocessing data penyakit jantung
- Training model Logistic Regression & SVC (dengan tuning hyperparameter)
- Tracking eksperimen otomatis (MLflow, DagsHub)
- Penyimpanan artefak model

## Dataset
Dataset terproses tersedia di folder `heart_disease_preprocessing/` dengan fitur-fitur medis dan target diagnosis.

## Dependensi Utama
- pandas, numpy, matplotlib, seaborn
- scikit-learn, imbalanced-learn
- mlflow, dagshub

Lihat `requirements.txt` untuk detail.

## Cara Menjalankan
1. Pastikan dependensi terinstall:  
   `pip install -r requirements.txt`
2. Jalankan training dasar:
   ```bash
   python modelling.py
   ```
3. Untuk tuning & tracking lanjut:
   ```bash
   python modelling_tuning.py
   ```

## Tracking Eksperimen
Hasil eksperimen dapat dilihat di DagsHub:  
https://dagshub.com/millkywaay/mlsystem-heart-disease
