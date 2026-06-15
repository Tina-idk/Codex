from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import json
import os
import re
import shutil
import urllib.request
import zipfile

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.ensemble import ExtraTreesRegressor, HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, median_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

DATA_DIR = Path("data")
MODEL_PATH = Path("art_price_model.joblib")
SOURCE_INFO_PATH = DATA_DIR / "source_info.json"
ARTIST_INFO_PATH = DATA_DIR / "artist_public_info.csv"
VALID_IMAGE_URLS_PATH = DATA_DIR / "valid_image_urls.csv"
METADATA_CSV_NAMES = {ARTIST_INFO_PATH.name, VALID_IMAGE_URLS_PATH.name}
DEFAULT_KAGGLE_SLUGS = ["amaboh/masterworks-top-10-1m-artists-20182022"]
PRICE_CANDIDATES = ["price", "price_($)", "sale_price", "sold_price", "hammer_price", "realized_price", "auction_price_estimate", "成交价"]
FEATURE_CANDIDATES = ["artist", "title", "description", "year", "medium", "size", "dimensions", "purchase_price", "estimate_low", "estimate_high", "category", "auction_house", "date", "gross_appreciation_period", "artist_name", "artist_birth_year", "artist_death_year", "artwork_year", "holding_period_years", "artist_country", "artist_movement"]


def normalize_column_name(name):
    return str(name).strip().lower().replace(" ", "_")


def artwork_csv_files():
    if not DATA_DIR.exists():
        return []
    return sorted(
        path
        for path in DATA_DIR.glob("*.csv")
        if not path.name.startswith(".") and path.name not in METADATA_CSV_NAMES
    )


def clean_price(value):
    if pd.isna(value):
        return np.nan
    if isinstance(value, (int, float, np.number)):
        return float(value)
    original = str(value).strip().upper()
    multiplier = 1
    if "B" in original:
        multiplier = 1_000_000_000
    elif "M" in original:
        multiplier = 1_000_000
    elif "K" in original:
        multiplier = 1_000
    text = re.sub(r"[^0-9.\-]", "", original.replace(",", ""))
    if text in {"", ".", "-", "-."}:
        return np.nan
    try:
        return float(text) * multiplier
    except ValueError:
        return np.nan


def parse_first_number(value):
    if pd.isna(value):
        return np.nan
    match = re.search(r"(\d+(?:\.\d+)?)", str(value).replace(",", ""))
    return float(match.group(1)) if match else np.nan


def parse_artist_name(value):
    if pd.isna(value):
        return ""
    text = re.sub(r"\s*\(.*?\)\s*", "", str(value))
    return re.sub(r"\s+", " ", text).strip().title()


def normalize_artist_key(value):
    if pd.isna(value):
        return ""
    text = parse_artist_name(value).replace("-", " ")
    return re.sub(r"[^A-Za-z0-9]+", " ", text).strip().lower()


def parse_artist_year(value, marker):
    if pd.isna(value):
        return np.nan
    match = re.search(rf"{marker}\.\s*(\d{{4}})", str(value), flags=re.IGNORECASE)
    return float(match.group(1)) if match else np.nan


def parse_artwork_year(row):
    for col in ["url", "title", "year"]:
        if col in row and pd.notna(row[col]):
            match = re.search(r"(1[5-9]\d{2}|20\d{2})", str(row[col]))
            if match:
                return float(match.group(1))
    return np.nan


def write_source_info(source_type, message, dataset_slug=None):
    DATA_DIR.mkdir(exist_ok=True)
    SOURCE_INFO_PATH.write_text(
        json.dumps({"source_type": source_type, "message": message, "dataset_slug": dataset_slug}, indent=2),
        encoding="utf-8",
    )


def read_source_info():
    if not SOURCE_INFO_PATH.exists():
        return {"source_type": "unknown", "message": "Data source is unknown.", "dataset_slug": None}
    return json.loads(SOURCE_INFO_PATH.read_text(encoding="utf-8"))


def find_first_existing_column(columns, candidates):
    lower_to_original = {normalize_column_name(col): col for col in columns}
    for candidate in candidates:
        normalized = normalize_column_name(candidate)
        if normalized in lower_to_original:
            return lower_to_original[normalized]
    return None


def ensure_artist_public_info():
    DATA_DIR.mkdir(exist_ok=True)
    if not ARTIST_INFO_PATH.exists():
        ARTIST_INFO_PATH.write_text("artist_name,artist_country,artist_movement,artist_summary,artist_source\n", encoding="utf-8")
    return pd.read_csv(ARTIST_INFO_PATH)


def merge_artist_public_info(df):
    artist_info = ensure_artist_public_info()
    if "artist_name" not in df.columns or artist_info.empty:
        return df
    df = df.copy()
    artist_info = artist_info.copy()
    df["_artist_key"] = df["artist_name"].apply(normalize_artist_key)
    artist_info["_artist_key"] = artist_info["artist_name"].apply(normalize_artist_key)
    merged = df.merge(artist_info.drop(columns=["artist_name"], errors="ignore"), on="_artist_key", how="left")
    return merged.drop(columns=["_artist_key"])


def enrich_artwork_features(df):
    df = df.copy()
    if "artist" in df.columns:
        df["artist_name"] = df["artist"].apply(parse_artist_name)
        df["artist_birth_year"] = df["artist"].apply(lambda value: parse_artist_year(value, "b"))
        df["artist_death_year"] = df["artist"].apply(lambda value: parse_artist_year(value, "d"))
    if any(col in df.columns for col in ["url", "title", "year"]):
        df["artwork_year"] = df.apply(parse_artwork_year, axis=1)
    if "gross_appreciation_period" in df.columns:
        df["holding_period_years"] = df["gross_appreciation_period"].apply(parse_first_number)
    return df


def image_url_works(url):
    if pd.isna(url) or not str(url).startswith("http"):
        return False
    headers = {"User-Agent": "Mozilla/5.0"}
    for method in ["HEAD", "GET"]:
        try:
            request = urllib.request.Request(str(url), headers=headers, method=method)
            with urllib.request.urlopen(request, timeout=6) as response:
                content_type = response.headers.get("content-type", "").lower()
                if response.status < 400 and content_type.startswith("image/"):
                    return True
        except Exception:
            continue
    return False


def filter_valid_image_urls(df):
    if "image_url" not in df.columns:
        return df
    DATA_DIR.mkdir(exist_ok=True)
    if VALID_IMAGE_URLS_PATH.exists():
        valid_urls = set(pd.read_csv(VALID_IMAGE_URLS_PATH)["image_url"].dropna().astype(str))
        return df[df["image_url"].astype(str).isin(valid_urls)].reset_index(drop=True)
    urls = sorted(set(df["image_url"].dropna().astype(str)))
    valid_urls = []
    with ThreadPoolExecutor(max_workers=16) as executor:
        future_to_url = {executor.submit(image_url_works, url): url for url in urls}
        for future in as_completed(future_to_url):
            if future.result():
                valid_urls.append(future_to_url[future])
    pd.DataFrame({"image_url": sorted(valid_urls)}).to_csv(VALID_IMAGE_URLS_PATH, index=False)
    return df[df["image_url"].astype(str).isin(valid_urls)].reset_index(drop=True)


def filter_image_records(df):
    if "image_url" not in df.columns:
        return df
    has_url = df["image_url"].notna() & df["image_url"].astype(str).str.startswith("http")
    filtered = df[has_url]
    if "has_image" in df.columns:
        has_image = df["has_image"].astype(str).str.lower().isin(["true", "1", "yes"])
        filtered = df[has_url & has_image]
    if len(filtered) >= 20:
        valid = filter_valid_image_urls(filtered)
        return valid if len(valid) >= 20 else filtered.reset_index(drop=True)
    return df.reset_index(drop=True)


def copy_csvs_from_directory(source_dir, prefix):
    copied = 0
    for csv_file in Path(source_dir).rglob("*.csv"):
        target = DATA_DIR / f"{prefix}_{csv_file.name}"
        shutil.copy2(csv_file, target)
        copied += 1
    return copied


def try_download_kaggle_dataset():
    DATA_DIR.mkdir(exist_ok=True)
    slugs = [os.getenv("KAGGLE_DATASET_SLUG")] if os.getenv("KAGGLE_DATASET_SLUG") else []
    slugs.extend(DEFAULT_KAGGLE_SLUGS)
    errors = []

    for slug in slugs:
        try:
            import kagglehub

            downloaded_dir = Path(kagglehub.dataset_download(slug))
            copied = copy_csvs_from_directory(downloaded_dir, "kaggle")
            if copied:
                write_source_info("kaggle", f"Automatically downloaded {copied} CSV file(s) from Kaggle dataset {slug}.", slug)
                return True, f"Downloaded Kaggle dataset with kagglehub: {slug}"
            errors.append(f"{slug}: kagglehub did not return any CSV files")
        except Exception as error:
            errors.append(f"{slug}: kagglehub failed: {error}")

        try:
            zip_path = DATA_DIR / f"{slug.replace('/', '_')}.zip"
            extract_dir = DATA_DIR / f"kaggle_{slug.replace('/', '_')}"
            urllib.request.urlretrieve(f"https://www.kaggle.com/api/v1/datasets/download/{slug}", zip_path)
            if extract_dir.exists():
                shutil.rmtree(extract_dir)
            extract_dir.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(zip_path) as archive:
                archive.extractall(extract_dir)
            copied = copy_csvs_from_directory(extract_dir, "kaggle")
            if copied:
                write_source_info("kaggle", f"Automatically downloaded {copied} CSV file(s) from Kaggle dataset {slug}.", slug)
                return True, f"Downloaded Kaggle dataset through Kaggle API URL: {slug}"
            errors.append(f"{slug}: direct download did not contain CSV files")
        except Exception as error:
            errors.append(f"{slug}: direct download failed: {error}")

    return False, "; ".join(errors)


def ensure_data_available():
    if artwork_csv_files():
        if not SOURCE_INFO_PATH.exists():
            write_source_info("local_csv", "Using artwork CSV file(s) found in the data/ folder.")
        return

    downloaded, message = try_download_kaggle_dataset()
    if downloaded:
        return

    write_source_info(
        "kaggle_download_failed",
        f"Kaggle 数据集必须下载成功才能启动游戏。下载失败详情：{message}",
    )
    raise RuntimeError(
        "Kaggle 数据集必须下载成功才能启动游戏。"
        "请确认 Streamlit Cloud 可以访问 Kaggle，并且 requirements.txt 已安装 kagglehub。"
        f"下载失败详情：{message}"
    )


def load_art_data():
    ensure_data_available()
    frames = []
    for csv_file in artwork_csv_files():
        frame = pd.read_csv(csv_file)
        frame.columns = [normalize_column_name(col) for col in frame.columns]
        frame["source_file"] = csv_file.name
        frames.append(frame)
    if not frames:
        raise FileNotFoundError("No usable public artwork CSV file found in data/ after Kaggle download.")
    return merge_artist_public_info(enrich_artwork_features(pd.concat(frames, ignore_index=True, sort=False)))


def prepare_training_data(df):
    price_col = find_first_existing_column(df.columns, PRICE_CANDIDATES)
    if price_col is None:
        raise ValueError("Could not find a sale price column.")
    df = df.copy()
    df["target_price"] = df[price_col].apply(clean_price)
    df = df.dropna(subset=["target_price"])
    df = df[df["target_price"] > 0]
    if "purchase_price" in df.columns:
        df["purchase_price"] = df["purchase_price"].apply(clean_price)
    if len(df) < 20:
        raise ValueError("Not enough usable rows after cleaning.")
    feature_cols = [col for col in FEATURE_CANDIDATES if col in df.columns and col != price_col]
    return df[feature_cols], df["target_price"], feature_cols, price_col


def split_feature_types(X):
    numeric = [col for col in X.columns if pd.api.types.is_numeric_dtype(X[col])]
    return numeric, [col for col in X.columns if col not in numeric]


def onehot_preprocessor(X):
    numeric, categorical = split_feature_types(X)
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]), categorical),
    ])


def ordinal_preprocessor(X):
    numeric, categorical = split_feature_types(X)
    return ColumnTransformer([
        ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical),
    ], sparse_threshold=0)


def log_target(regressor):
    return TransformedTargetRegressor(regressor=regressor, func=np.log1p, inverse_func=np.expm1)


def candidate_models(X):
    return {
        "random_forest_log_target": log_target(Pipeline([("preprocessor", onehot_preprocessor(X)), ("model", RandomForestRegressor(n_estimators=250, min_samples_leaf=2, random_state=42, n_jobs=-1))])),
        "extra_trees_log_target": log_target(Pipeline([("preprocessor", onehot_preprocessor(X)), ("model", ExtraTreesRegressor(n_estimators=300, min_samples_leaf=2, random_state=42, n_jobs=-1))])),
        "hist_gradient_boosting_log_target": log_target(Pipeline([("preprocessor", ordinal_preprocessor(X)), ("model", HistGradientBoostingRegressor(max_iter=250, learning_rate=0.04, l2_regularization=0.03, min_samples_leaf=8, random_state=42))])),
    }


def evaluate(model, X_test, y_test):
    pred = np.maximum(0, model.predict(X_test))
    return {"mae": mean_absolute_error(y_test, pred), "median_absolute_error": median_absolute_error(y_test, pred), "r2": r2_score(y_test, pred)}


def train_and_save_model():
    df = load_art_data()
    X, y, features, price_col = prepare_training_data(df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    results = {}
    best_name, best_model, best_mae = None, None, float("inf")
    for name, model in candidate_models(X_train).items():
        model.fit(X_train, y_train)
        metrics = evaluate(model, X_test, y_test)
        results[name] = metrics
        if metrics["mae"] < best_mae:
            best_name, best_model, best_mae = name, model, metrics["mae"]
    artifact = {"model": best_model, "model_name": best_name, "model_results": results, "feature_columns": features, "price_column": price_col, "training_rows": len(X), "train_rows": len(X_train), "test_rows": len(X_test), **results[best_name]}
    joblib.dump(artifact, MODEL_PATH)
    return artifact


def main():
    artifact = train_and_save_model()
    print("Model trained successfully.")
    print(f"Rows used: {artifact['training_rows']}")
    print(f"Train rows: {artifact['train_rows']}")
    print(f"Test rows: {artifact['test_rows']}")
    print(f"Best model: {artifact['model_name']}")
    print(f"Mean absolute error: {artifact['mae']:,.2f}")
    print(f"Median absolute error: {artifact['median_absolute_error']:,.2f}")
    print(f"R2 score: {artifact['r2']:.3f}")


if __name__ == "__main__":
    main()
