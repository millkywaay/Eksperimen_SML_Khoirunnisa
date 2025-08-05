import os
import pandas as pd
import numpy as np
import argparse
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.pipeline import make_pipeline
from imblearn.over_sampling import SMOTE

def handle_outliers(df, columns):
    """
    Menangani outliers pada kolom-kolom numerik yang diberikan dengan
    metode winsorizing (clipping ke 1.5 * IQR).
    """
    df_out = df.copy()
    for col in columns:
        Q1 = df_out[col].quantile(0.25)
        Q3 = df_out[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df_out[col] = df_out[col].clip(lower=lower_bound, upper=upper_bound)
    return df_out

def run_preprocessing(input_path, output_dir):
    """
    Fungsi utama untuk menjalankan seluruh pipeline preprocessing.
    
    Args:
        input_path (str): Path ke file CSV data mentah.
        output_dir (str): Path ke direktori untuk menyimpan hasil preprocessing.
    """
    print("Memulai proses preprocessing...")
    try:
        df = pd.read_csv(input_path)
        print(f"Data mentah dimuat. Bentuk: {df.shape}")
    except FileNotFoundError:
        print(f"Error: File tidak ditemukan di {input_path}")
        return

    df.drop(['id', 'dataset'], axis=1, inplace=True)
    df_clean = df.dropna()
    print(f"Baris dengan nilai null telah dihapus. Bentuk sekarang: {df_clean.shape}")

    df_clean['target'] = df_clean['num'].apply(lambda x: 1 if x > 0 else 0)
    
    X = df_clean.drop(['num', 'target'], axis=1)
    y = df_clean['target']
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Data dibagi menjadi set latih ({X_train.shape}) dan uji ({X_test.shape})")

    numeric_features = X_train.select_dtypes(include=np.number).columns
    X_train = handle_outliers(X_train, numeric_features)
    print("Outlier pada data latih telah ditangani.")

    categorical_features = X_train.select_dtypes(include=['object', 'category']).columns

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

    pipeline = make_pipeline(
        preprocessor,
        SMOTE(random_state=42)
    )
    
    X_train_resampled, y_train_resampled = pipeline.fit_resample(X_train, y_train)
    X_test_processed = pipeline.named_steps['columntransformer'].transform(X_test)
    print("Pipeline telah diterapkan pada data latih (dengan SMOTE) dan data uji.")

    
    os.makedirs(output_dir, exist_ok=True)
    encoded_cat_features = pipeline.named_steps['columntransformer'].named_transformers_['cat'].get_feature_names_out(categorical_features)
    all_feature_names = list(numeric_features) + list(encoded_cat_features)
    
    # Simpan data latih yang sudah diproses
    X_train_df = pd.DataFrame(X_train_resampled, columns=all_feature_names)
    y_train_df = pd.DataFrame(y_train_resampled, columns=['target'])
    train_processed_df = pd.concat([X_train_df, y_train_df], axis=1)
    train_path = os.path.join(output_dir, 'train_processed.csv')
    train_processed_df.to_csv(train_path, index=False)
    
    # Simpan data uji yang sudah diproses
    X_test_df = pd.DataFrame(X_test_processed, columns=all_feature_names)
    y_test_df = pd.DataFrame(y_test.values, columns=['target'])
    test_processed_df = pd.concat([X_test_df, y_test_df], axis=1)
    test_path = os.path.join(output_dir, 'test_processed.csv')
    test_processed_df.to_csv(test_path, index=False)

    print(f"\nPreprocessing selesai. Hasil disimpan di folder '{output_dir}':")
    print(f"- {train_path}")
    print(f"- {test_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script untuk preprocessing data penyakit jantung.")
    
    parser.add_argument(
        '--input', 
        type=str, 
        default='heart_disease_uci.csv', 
        help='Path ke file CSV data mentah.'
    )
    parser.add_argument(
        '--output', 
        type=str, 
        default='heart_disease_preprocessing', 
        help='Nama folder untuk menyimpan data yang sudah diproses.'
    )
    
    args = parser.parse_args()

    run_preprocessing(args.input, args.output)