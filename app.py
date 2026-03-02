import os
import traceback
import torch
import torch.nn as nn
import pandas as pd
import numpy as np

from fastapi import FastAPI, UploadFile, File, Query
from fastapi.responses import FileResponse, JSONResponse
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

# =====================================================
# CONFIG
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "fraud_cnn_model.pth")
TRAIN_PROCESSED_PATH = os.path.join(BASE_DIR, "transactions_train_processed.csv")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
app = FastAPI(title="Fraud Detection API", version="4.0")

# =====================================================
# MODEL
# =====================================================

class FraudCNN(nn.Module):
    def __init__(self, num_features):
        super(FraudCNN, self).__init__()
        self.conv1 = nn.Conv1d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm1d(32)

        self.conv2 = nn.Conv1d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(64)

        self.conv3 = nn.Conv1d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(128)

        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(128, 1)

    def forward(self, x):
        x = x.unsqueeze(1)
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.relu(self.bn2(self.conv2(x)))
        x = self.relu(self.bn3(self.conv3(x)))
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        x = self.fc(x)
        return x

# =====================================================
# LOAD MODEL
# =====================================================

if not os.path.exists(MODEL_PATH):
    raise RuntimeError(f"Model not found at {MODEL_PATH}")

checkpoint = torch.load(MODEL_PATH, map_location=device)
num_features = checkpoint["num_features"]

model = FraudCNN(num_features).to(device)
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# =====================================================
# RECOVER TRAINING COLUMNS
# =====================================================

train_df = pd.read_csv(TRAIN_PROCESSED_PATH)
if "is_fraud" in train_df.columns:
    train_df.drop(columns=["is_fraud"], inplace=True)

training_columns = train_df.columns.tolist()

# =====================================================
# PREPROCESSING
# =====================================================

def preprocess_input(df: pd.DataFrame):
    df = df.copy()

    if "is_fraud" in df.columns:
        df.drop(columns=["is_fraud"], inplace=True)

    if "transaction_time" in df.columns:
        df["transaction_time"] = pd.to_datetime(df["transaction_time"], errors="coerce")
        df["txn_hour"] = df["transaction_time"].dt.hour
        df["txn_day"] = df["transaction_time"].dt.day
        df["txn_month"] = df["transaction_time"].dt.month
        df["txn_dayofweek"] = df["transaction_time"].dt.dayofweek
        df.drop(columns=["transaction_time"], inplace=True)

    drop_cols = ["transaction_id", "customer_id", "merchant_id"]
    df.drop(columns=[c for c in drop_cols if c in df.columns],
            errors="ignore",
            inplace=True)

    categorical_cols = ["payment_channel", "device_type"]
    df = pd.get_dummies(
        df,
        columns=[c for c in categorical_cols if c in df.columns],
        drop_first=True
    )

    df = df.reindex(columns=training_columns, fill_value=0)

    return df

# =====================================================
# FILE PREDICTION WITH FULL METRICS + THRESHOLD
# =====================================================

@app.post("/predict-file")
async def predict_file(
    file: UploadFile = File(...),
    threshold: float = Query(0.5, ge=0.0, le=1.0)
):
    try:
        if file.filename.endswith(".csv"):
            df = pd.read_csv(file.file)
        elif file.filename.endswith(".xlsx"):
            df = pd.read_excel(file.file)
        else:
            return JSONResponse(
                {"error": "Only CSV or Excel files supported"},
                status_code=400
            )

        original_df = df.copy()

        y_true = None
        if "is_fraud" in df.columns:
            y_true = df["is_fraud"].values

        df_processed = preprocess_input(df)

        if df_processed.shape[1] != num_features:
            return JSONResponse(
                {"error": f"Feature mismatch. Expected {num_features}, got {df_processed.shape[1]}"},
                status_code=400
            )

        X = torch.tensor(df_processed.values.astype(np.float32)).to(device)

        with torch.no_grad():
            outputs = model(X).squeeze()
            probs = torch.sigmoid(outputs).cpu().numpy()

        preds = (probs > threshold).astype(int)

        original_df["fraud_probability"] = probs
        original_df["predicted_fraud"] = preds

        fraud_only = original_df[original_df["predicted_fraud"] == 1]

        output_path = os.path.join(BASE_DIR, "detected_fraud.csv")
        fraud_only.to_csv(output_path, index=False)

        response_data = {
            "threshold_used": threshold,
            "total_transactions": len(original_df),
            "fraud_detected": int(preds.sum()),
            "download_endpoint": "/download-fraud-csv"
        }

        if y_true is not None:
            roc = roc_auc_score(y_true, probs)
            f1 = f1_score(y_true, preds)
            precision = precision_score(y_true, preds)
            recall = recall_score(y_true, preds)

            tn, fp, fn, tp = confusion_matrix(y_true, preds).ravel()

            response_data.update({
                "roc_auc": float(roc),
                "f1_score": float(f1),
                "precision": float(precision),
                "recall": float(recall),
                "confusion_matrix": {
                    "true_negative": int(tn),
                    "false_positive": int(fp),
                    "false_negative": int(fn),
                    "true_positive": int(tp),
                }
            })

        return response_data

    except Exception as e:
        traceback.print_exc()
        return JSONResponse({"error": str(e)}, status_code=500)

# =====================================================
# DOWNLOAD CSV
# =====================================================

@app.get("/download-fraud-csv")
def download_fraud_csv():
    output_path = os.path.join(BASE_DIR, "detected_fraud.csv")
    return FileResponse(
        output_path,
        media_type="text/csv",
        filename="detected_fraud.csv"
    )

# =====================================================
# HEALTH CHECK
# =====================================================

@app.get("/")
def health():
    return {
        "status": "Running",
        "expected_features": num_features
    }