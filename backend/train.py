from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

# ============================================================
# CONFIGURATION & PATHS (Root-Aligned)
# ============================================================
BASE_DIR = Path(__file__).resolve().parent  # Points to SignBridge-AI/
RAW_DIR = BASE_DIR / "dataset" / "raw"
PROC_DIR = BASE_DIR / "backend" / "dataset" / "processed"
PROC_DIR.mkdir(parents=True, exist_ok=True)