import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from joblib import dump


def preprocess_crop_data(
        dataset_path,
        target_column,
        train_output_path,
        test_output_path,
        preprocessor_path,
        encoder_path):

    # Load dataset
    data = pd.read_csv(dataset_path)

    # Split feature dan target
    X = data.drop(columns=[target_column])
    y = data[target_column]

    # Label encoding
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Train test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Pipeline
    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="mean")),
        ("scaler", StandardScaler())
    ])

    X_train = numeric_pipeline.fit_transform(X_train)
    X_test = numeric_pipeline.transform(X_test)

    X_train = pd.DataFrame(X_train, columns=X.columns)
    X_test = pd.DataFrame(X_test, columns=X.columns)
    X_train[target_column] = y_train
    X_test[target_column] = y_test
    Path(train_output_path).parent.mkdir(parents=True, exist_ok=True)

    # Simpan dataset hasil preprocess beserta artefaknya
    X_train.to_csv(train_output_path, index=False)
    X_test.to_csv(test_output_path, index=False)
    dump(numeric_pipeline, preprocessor_path)
    dump(label_encoder, encoder_path)

    # Simpan nama fitur
    pd.Series(X.columns).to_csv(
        Path(preprocessor_path).parent / "feature_columns.csv",
        index=False
    )

    print("Train dataset saved to:", train_output_path)
    print("Test dataset saved to:", test_output_path)


if __name__ == "__main__":

    BASE_DIR = Path(__file__).resolve().parent.parent

    preprocess_crop_data(
        dataset_path=BASE_DIR / "CropRecommendation_raw.csv",
        target_column="label",
        train_output_path=BASE_DIR / "preprocessing/CropRecommendation_preprocessing/train_preprocessed.csv",
        test_output_path=BASE_DIR / "preprocessing/CropRecommendation_preprocessing/test_preprocessed.csv",
        preprocessor_path=BASE_DIR / "preprocessing/CropRecommendation_preprocessing/preprocessor.joblib",
        encoder_path=BASE_DIR / "preprocessing/CropRecommendation_preprocessing/label_encoder.joblib"
    )