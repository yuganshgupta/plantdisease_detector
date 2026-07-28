"""
Plant Disease AI — Streamlit Application
==========================================
Production-quality web application for diagnosing plant diseases using
trained Deep Learning models. Supports 38 conditions across 14 crop species.
"""

import sys
import time
import json
import traceback
from pathlib import Path

import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image

# ── Project paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
MODELS_DIR = PROJECT_ROOT / 'models'
DATASET_DIR = PROJECT_ROOT / 'dataset'
TEST_DIR = DATASET_DIR / 'test' / 'test'

st.set_page_config(
    page_title="Plant Disease AI",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Optional imports ───────────────────────────────────────────────────────────
try:
    import torch
    import torch.nn as nn
    from torch.amp import autocast
    from torchvision import models as tv_models
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

try:
    import albumentations as A
    from albumentations.pytorch import ToTensorV2
    ALBUM_AVAILABLE = True
except ImportError:
    ALBUM_AVAILABLE = False

try:
    import joblib
    JOBLIB_AVAILABLE = True
except ImportError:
    JOBLIB_AVAILABLE = False

try:
    from disease_info import get_disease_info
except ImportError:
    def get_disease_info(name):
        clean = name.replace('___', ' — ').replace('_', ' ')
        return {
            'name': clean, 'scientific_name': 'Unknown',
            'description': f"Information for '{clean}' is not available.",
            'symptoms': '', 'causes': '',
            'treatment': 'Consult a local agricultural expert.',
            'prevention': 'Practice crop rotation and good hygiene.',
            'severity': 'Medium',
            'recommended_actions': 'Remove affected leaves and scout regularly.',
        }

# ── Class names ────────────────────────────────────────────────────────────────
CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust',
    'Apple___healthy', 'Blueberry___healthy',
    'Cherry_(including_sour)___Powdery_mildew',
    'Cherry_(including_sour)___healthy',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
    'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight',
    'Corn_(maize)___healthy', 'Grape___Black_rot',
    'Grape___Esca_(Black_Measles)',
    'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
    'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
    'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
    'Pepper,_bell___healthy', 'Potato___Early_blight',
    'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy',
    'Soybean___healthy', 'Squash___Powdery_mildew',
    'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight',
    'Tomato___Late_blight', 'Tomato___Leaf_Mold',
    'Tomato___Septoria_leaf_spot',
    'Tomato___Spider_mites Two-spotted_spider_mite',
    'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus',
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy',
]

_cn_path = MODELS_DIR / 'class_names.json'
if _cn_path.exists():
    try:
        with open(_cn_path, encoding='utf-8') as f:
            CLASS_NAMES = json.load(f)
    except Exception:
        pass

NUM_CLASSES = len(CLASS_NAMES)
IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def format_class_name(raw: str) -> str:
    """Human-readable class name from dataset directory name."""
    if '___' in raw:
        plant, disease = raw.split('___', 1)
        plant = plant.replace('_', ' ').replace('(including sour)', '').strip()
        disease = disease.replace('_', ' ').strip()
        return f"{plant} (Healthy)" if disease.lower() == 'healthy' else f"{plant} — {disease}"
    return raw.replace('_', ' ')


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL ARCHITECTURES
# ══════════════════════════════════════════════════════════════════════════════

class CustomCNN(nn.Module):
    def __init__(self, num_classes=38):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, padding=1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(256, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, padding=1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(), nn.Dropout(0.5),
            nn.Linear(512, 256), nn.ReLU(True),
            nn.Dropout(0.3), nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def build_model(name: str, num_classes: int = 38) -> nn.Module:
    if name == 'custom_cnn':
        return CustomCNN(num_classes)
    elif name == 'efficientnet_b0':
        m = tv_models.efficientnet_b0(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    elif name == 'efficientnet_v2_s':
        m = tv_models.efficientnet_v2_s(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    elif name == 'resnet50':
        m = tv_models.resnet50(weights=None)
        m.fc = nn.Sequential(nn.Dropout(0.5), nn.Linear(m.fc.in_features, num_classes))
        return m
    elif name == 'convnext_tiny':
        m = tv_models.convnext_tiny(weights=None)
        m.classifier[-1] = nn.Linear(m.classifier[-1].in_features, num_classes)
        return m
    else:
        raise ValueError(f"Unknown model: {name}")


# ══════════════════════════════════════════════════════════════════════════════
#  DATA LOADING & MODEL DISCOVERY
# ══════════════════════════════════════════════════════════════════════════════

def load_metrics():
    p = MODELS_DIR / 'metrics.json'
    if p.exists():
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            pass
    return None


def load_comparison():
    p = MODELS_DIR / 'comparison.csv'
    if p.exists():
        try:
            return pd.read_csv(p)
        except Exception:
            pass
    return None


def discover_models() -> dict:
    """Discover available models, ordered by accuracy (best first)."""
    metrics = load_metrics()
    available = {}

    if not MODELS_DIR.exists():
        return available

    # Build a lookup of accuracy from metrics.json
    acc_lookup = {}
    if metrics and 'models' in metrics:
        for m in metrics['models']:
            acc_lookup[m['model']] = m.get('accuracy', 0)

    # DL models
    if TORCH_AVAILABLE:
        dl_models = []
        for f in MODELS_DIR.glob('*_best.pth'):
            key = f.stem.replace('_best', '')
            acc = acc_lookup.get(key, 0)
            display = key.replace('_', ' ').title()
            dl_models.append((acc, key, display, f))

        # Sort by accuracy descending — best model first
        dl_models.sort(key=lambda x: x[0], reverse=True)

        for acc, key, display, f in dl_models:
            tag = "★ Recommended" if dl_models and f == dl_models[0][3] else "DL"
            available[f"{display} ({tag})"] = f

    # ML models
    if JOBLIB_AVAILABLE:
        svm_path = MODELS_DIR / 'svm_model.pkl'
        if svm_path.exists():
            available["SVM (Classical ML)"] = svm_path

    return available


def get_device():
    if TORCH_AVAILABLE and torch.cuda.is_available():
        return torch.device('cuda')
    return torch.device('cpu') if TORCH_AVAILABLE else None


@st.cache_resource
def load_pytorch_model(path_str: str):
    if not TORCH_AVAILABLE:
        return None, None, None
    device = get_device()
    try:
        ckpt = torch.load(path_str, map_location=device, weights_only=False)
        name = ckpt.get('model_name', Path(path_str).stem.replace('_best', ''))
        nc = len(ckpt.get('class_names', CLASS_NAMES))
        model = build_model(name, nc)
        model.load_state_dict(ckpt['model_state_dict'])
        return model.to(device).eval(), name, ckpt
    except Exception as e:
        st.error(f"Model load error: {e}")
        return None, None, None


@st.cache_resource
def load_ml_model(path_str: str):
    if not JOBLIB_AVAILABLE:
        return None
    try:
        return joblib.load(path_str)
    except Exception as e:
        st.error(f"ML model error: {e}")
        return None


@st.cache_resource
def get_ml_feature_extractor():
    if not TORCH_AVAILABLE:
        return None
    metrics = load_metrics()
    if not metrics or 'best_model' not in metrics:
        return None

    best_model_name = metrics['best_model']
    best_dl_path = MODELS_DIR / f"{best_model_name}_best.pth"
    if not best_dl_path.exists():
        return None

    device = get_device()
    try:
        model, mname, _ = load_pytorch_model(str(best_dl_path))
        if model is None:
            return None

        if mname == 'custom_cnn':
            ext = nn.Sequential(model.features, nn.Flatten())
        elif mname in ('efficientnet_b0', 'efficientnet_v2_s', 'convnext_tiny'):
            ext = nn.Sequential(model.features, nn.AdaptiveAvgPool2d(1), nn.Flatten())
        elif mname == 'resnet50':
            ext = nn.Sequential(*list(model.children())[:-1], nn.Flatten())
        else:
            ext = nn.Sequential(*list(model.children())[:-1], nn.Flatten())

        return ext.to(device).eval()
    except Exception as e:
        st.error(f"Failed to load ML feature extractor: {e}")
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def preprocess(pil_img):
    img = np.array(pil_img.convert('RGB'))
    if ALBUM_AVAILABLE:
        t = A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ToTensorV2(),
        ])
        return t(image=img)['image']
    arr = np.array(pil_img.convert('RGB').resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1)).float()


def predict_dl(model, pil_img, device):
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    t0 = time.time()
    with torch.no_grad():
        with autocast(device_type=str(device).split(':')[0], enabled=(device.type == 'cuda')):
            out = model(tensor)
    return torch.softmax(out, 1).cpu().numpy()[0], time.time() - t0


def predict_ml(ml_model, feat_ext, pil_img, device):
    tensor = preprocess(pil_img).unsqueeze(0).to(device)
    with torch.no_grad():
        features = feat_ext(tensor).cpu().numpy()
    t0 = time.time()
    if hasattr(ml_model, 'predict_proba'):
        probs = ml_model.predict_proba(features)[0]
    else:
        pred = ml_model.predict(features)[0]
        probs = np.zeros(NUM_CLASSES)
        probs[int(pred)] = 1.0
    return probs, time.time() - t0


def generate_gradcam(model, model_name, pil_img, device):
    try:
        from pytorch_grad_cam import GradCAM
        from pytorch_grad_cam.utils.image import show_cam_on_image
        from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    except ImportError:
        return None

    layers = {
        'custom_cnn': lambda m: [m.features[-3]],
        'efficientnet_b0': lambda m: [m.features[-1]],
        'efficientnet_v2_s': lambda m: [m.features[-1]],
        'resnet50': lambda m: [m.layer4[-1]],
        'convnext_tiny': lambda m: [m.features[-1]],
    }
    getter = layers.get(model_name)
    if not getter:
        return None
    target_layers = getter(model)

    tensor = preprocess(pil_img)
    inp = tensor.unsqueeze(0).to(device)
    with torch.no_grad():
        pred_class = model(inp).argmax(1).item()

    cam = GradCAM(model=model, target_layers=target_layers)
    grayscale = cam(input_tensor=inp, targets=[ClassifierOutputTarget(pred_class)])[0]

    img_np = tensor.permute(1, 2, 0).numpy()
    img_np = np.clip(img_np * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1).astype(np.float32)
    return show_cam_on_image(img_np, grayscale, use_rgb=True)


def severity_color(s):
    return {'None': '#22c55e', 'Low': '#84cc16', 'Medium': '#eab308',
            'High': '#ef4444', 'Critical': '#dc2626'}.get(s, '#6b7280')


def severity_bg(s):
    return {'None': 'rgba(34,197,94,0.12)', 'Low': 'rgba(132,204,22,0.12)',
            'Medium': 'rgba(234,179,8,0.12)', 'High': 'rgba(239,68,68,0.12)',
            'Critical': 'rgba(220,38,38,0.12)'}.get(s, 'rgba(107,114,128,0.12)')


# ══════════════════════════════════════════════════════════════════════════════
#  STYLING
# ══════════════════════════════════════════════════════════════════════════════

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ── Reset ────────────────────────────────────────────────────────── */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    /* ── App background ───────────────────────────────────────────────── */
    .stApp {
        background: #0c0f1a;
    }

    /* ── Sidebar ──────────────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0f1629 0%, #131b30 100%);
        border-right: 1px solid rgba(255,255,255,0.04);
    }
    section[data-testid="stSidebar"] .stRadio label {
        color: #94a3b8 !important;
        font-weight: 500 !important;
        padding: 8px 12px !important;
        border-radius: 10px !important;
        transition: all 0.2s ease !important;
    }
    section[data-testid="stSidebar"] .stRadio label:hover {
        background: rgba(16, 185, 129, 0.08) !important;
        color: #e2e8f0 !important;
    }

    /* ── Typography ───────────────────────────────────────────────────── */
    .hero-title {
        font-size: 3.2rem; font-weight: 900; text-align: center;
        background: linear-gradient(135deg, #34d399 0%, #10b981 50%, #059669 100%);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
        margin: 0 0 0.4rem 0; letter-spacing: -0.04em; line-height: 1.1;
    }
    .hero-sub {
        font-size: 1.15rem; color: #64748b; text-align: center;
        margin-bottom: 2.5rem; font-weight: 400; line-height: 1.6;
        max-width: 600px; margin-left: auto; margin-right: auto;
    }
    .page-title {
        font-size: 1.8rem; font-weight: 800; letter-spacing: -0.03em;
        color: #f1f5f9; margin-bottom: 0.2rem;
    }
    .page-subtitle {
        font-size: 0.95rem; color: #64748b; font-weight: 400;
        margin-bottom: 1.8rem;
    }
    .section-label {
        font-size: 0.7rem; font-weight: 700; letter-spacing: 0.12em;
        text-transform: uppercase; color: #10b981; margin-bottom: 0.6rem;
    }

    /* ── Cards ─────────────────────────────────────────────────────────── */
    .card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 16px; padding: 24px;
        transition: border-color 0.2s ease;
    }
    .card:hover { border-color: rgba(16,185,129,0.2); }
    .card h4 { color: #e2e8f0; font-weight: 700; margin: 0 0 8px 0; font-size: 1rem; }
    .card p { color: #94a3b8; font-weight: 400; line-height: 1.55; margin: 0; font-size: 0.88rem; }
    .card-icon { font-size: 1.8rem; margin-bottom: 12px; display: block; }

    /* ── Step cards (home workflow) ────────────────────────────────────── */
    .step-card {
        background: rgba(15, 23, 42, 0.5);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 16px; padding: 28px 24px; text-align: center;
        position: relative;
    }
    .step-num {
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 50%;
        background: linear-gradient(135deg, #10b981, #059669);
        color: white; font-weight: 800; font-size: 0.9rem;
        margin-bottom: 14px;
    }
    .step-card h4 { color: #f1f5f9; font-weight: 700; margin: 0 0 8px 0; }
    .step-card p { color: #94a3b8; font-size: 0.85rem; line-height: 1.5; margin: 0; }

    /* ── Stat pill ─────────────────────────────────────────────────────── */
    .stat-pill {
        background: rgba(16, 185, 129, 0.08);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 12px; padding: 16px 20px; text-align: center;
    }
    .stat-pill .value {
        font-size: 1.8rem; font-weight: 800; color: #10b981;
        letter-spacing: -0.03em; line-height: 1;
    }
    .stat-pill .label {
        font-size: 0.72rem; font-weight: 600; color: #64748b;
        text-transform: uppercase; letter-spacing: 0.08em; margin-top: 6px;
    }

    /* ── Buttons ───────────────────────────────────────────────────────── */
    .stButton>button {
        border-radius: 12px !important; font-weight: 700 !important;
        font-size: 0.95rem !important; padding: 0.65rem 1.6rem !important;
        background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
        color: white !important; border: none !important;
        transition: all 0.15s ease !important;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.25) !important;
    }
    .stButton>button:hover {
        transform: translateY(-1px) !important;
        box-shadow: 0 4px 16px rgba(16, 185, 129, 0.4) !important;
    }

    /* ── File Uploader ────────────────────────────────────────────────── */
    div[data-testid="stFileUploader"] > section {
        border: 2px dashed rgba(16, 185, 129, 0.25) !important;
        border-radius: 14px !important;
        background: rgba(16, 185, 129, 0.03) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFileUploader"] > section:hover {
        border-color: rgba(16, 185, 129, 0.5) !important;
        background: rgba(16, 185, 129, 0.06) !important;
    }

    /* ── Severity badge ───────────────────────────────────────────────── */
    .severity-badge {
        display: inline-flex; align-items: center; gap: 6px;
        padding: 5px 14px; border-radius: 20px;
        font-weight: 700; font-size: 0.75rem; color: white;
        text-transform: uppercase; letter-spacing: 0.06em;
    }

    /* ── Diagnosis result card ────────────────────────────────────────── */
    .result-card {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(16, 185, 129, 0.15);
        border-radius: 16px; padding: 24px;
    }
    .result-title {
        font-size: 1.4rem; font-weight: 800; color: #f1f5f9;
        margin-bottom: 4px;
    }
    .result-subtitle {
        font-size: 0.85rem; color: #64748b; font-style: italic;
    }

    /* ── Metric tiles ─────────────────────────────────────────────────── */
    div[data-testid="stMetricValue"] {
        font-size: 1.6rem !important; font-weight: 800 !important;
        color: #10b981 !important; letter-spacing: -0.02em !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.75rem !important; color: #64748b !important;
        font-weight: 600 !important; text-transform: uppercase !important;
        letter-spacing: 0.06em !important;
    }

    /* ── Plotly bg fix ─────────────────────────────────────────────────── */
    .js-plotly-plot .plotly .main-svg { background: transparent !important; }

    /* ── Hide Streamlit branding ──────────────────────────────────────── */
    #MainMenu { visibility: hidden; }
    footer { visibility: hidden; }
    header { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════

PAGES = [
    "🏠 Home",
    "🔬 Diagnose",
    "📦 Batch Analysis",
    "📊 Models",
    "❓ Help",
]

with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 16px 0 8px 0;">
        <span style="font-size:2.2rem;">🌿</span>
        <div style="font-size:1.1rem; font-weight:800; color:#f1f5f9; margin-top:4px; letter-spacing:-0.02em;">
            Plant Disease AI
        </div>
        <div style="font-size:0.7rem; color:#475569; font-weight:500; margin-top:2px;">
            Deep Learning Diagnostics
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
    app_mode = st.radio("Navigate", PAGES, label_visibility="collapsed")

available_models = discover_models()
metrics_data = load_metrics()


# ══════════════════════════════════════════════════════════════════════════════
#  HOME
# ══════════════════════════════════════════════════════════════════════════════

if app_mode == "🏠 Home":
    st.markdown("")
    st.markdown('<div class="hero-title">Plant Disease AI</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload a photo of a plant leaf and get an instant diagnosis powered by deep learning — including disease identification, severity assessment, and treatment recommendations.</div>', unsafe_allow_html=True)

    # ── Stats row ──────────────────────────────────────────────────────
    best_acc = f"{metrics_data['best_accuracy']:.1%}" if metrics_data else "—"
    best_name = ""
    if metrics_data and 'models' in metrics_data and metrics_data['models']:
        best_name = metrics_data['models'][0].get('display_name', '')
    s1, s2, s3, s4 = st.columns(4)
    with s1:
        st.markdown(f'<div class="stat-pill"><div class="value">{best_acc}</div><div class="label">Best Accuracy</div></div>', unsafe_allow_html=True)
    with s2:
        st.markdown(f'<div class="stat-pill"><div class="value">{NUM_CLASSES}</div><div class="label">Conditions</div></div>', unsafe_allow_html=True)
    with s3:
        st.markdown(f'<div class="stat-pill"><div class="value">{len(available_models)}</div><div class="label">Models Ready</div></div>', unsafe_allow_html=True)
    with s4:
        st.markdown(f'<div class="stat-pill"><div class="value">14</div><div class="label">Crop Species</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── How it works (workflow steps) ──────────────────────────────────
    st.markdown('<div class="section-label">How It Works</div>', unsafe_allow_html=True)
    w1, w2, w3, w4 = st.columns(4)
    steps = [
        ("1", "📷", "Upload", "Take a clear photo of the affected leaf and upload it."),
        ("2", "🤖", "Analyze", "Our best AI model runs inference in milliseconds."),
        ("3", "🔬", "Diagnose", "Get the disease name, confidence, and visual attention map."),
        ("4", "💊", "Treat", "Receive actionable treatment and prevention advice."),
    ]
    for col, (num, icon, title, desc) in zip([w1, w2, w3, w4], steps):
        with col:
            st.markdown(f"""
            <div class="step-card">
                <div class="step-num">{num}</div>
                <div style="font-size:1.6rem; margin-bottom:8px;">{icon}</div>
                <h4>{title}</h4>
                <p>{desc}</p>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Supported crops ────────────────────────────────────────────────
    st.markdown('<div class="section-label">Supported Crops</div>', unsafe_allow_html=True)
    plants = sorted({c.split('___')[0].replace('_', ' ').replace('(including sour)', '').strip() for c in CLASS_NAMES})
    cols = st.columns(7)
    for i, plant in enumerate(plants):
        with cols[i % 7]:
            st.markdown(f"<div style='background:rgba(15,23,42,0.5); border:1px solid rgba(255,255,255,0.04); border-radius:10px; padding:10px; text-align:center; margin-bottom:8px;'><span style='font-size:0.82rem; color:#cbd5e1; font-weight:500;'>{plant}</span></div>", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.info("👈 Select **Diagnose** in the sidebar to get started.")


# ══════════════════════════════════════════════════════════════════════════════
#  DISEASE DETECTION
# ══════════════════════════════════════════════════════════════════════════════

elif app_mode == "🔬 Diagnose":
    st.markdown('<div class="page-title">Disease Diagnosis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload a leaf image to identify diseases instantly</div>', unsafe_allow_html=True)

    if not available_models:
        st.warning("No trained models found. Run `plant_disease.ipynb` first.")
        st.stop()

    # ── Input panel ────────────────────────────────────────────────────
    col_in, spacer, col_out = st.columns([3, 0.3, 6.7])

    with col_in:
        st.markdown('<div class="section-label">Image Input</div>', unsafe_allow_html=True)
        input_mode = st.radio("Source", ["Upload", "Sample"], horizontal=True, label_visibility="collapsed")

        pil_image = None
        if input_mode == "Upload":
            f = st.file_uploader("Upload leaf image", type=["jpg", "jpeg", "png"], label_visibility="collapsed")
            if f:
                pil_image = Image.open(f)
        else:
            samples = []
            if TEST_DIR.exists():
                for ext in ['*.jpg', '*.JPG', '*.png', '*.PNG', '*.jpeg']:
                    samples.extend(TEST_DIR.glob(ext))
            if samples:
                sel = st.selectbox("Choose sample", samples, format_func=lambda p: p.name, label_visibility="collapsed")
                pil_image = Image.open(sel)
            else:
                st.caption("No sample images available.")

        if pil_image:
            st.image(pil_image, use_container_width=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Model selector — default to recommended (first = best accuracy)
        st.markdown('<div class="section-label">Model</div>', unsafe_allow_html=True)
        model_keys = list(available_models.keys())
        model_choice = st.selectbox(
            "Model", model_keys,
            index=0,  # Best model is first
            label_visibility="collapsed",
        )
        model_path = available_models[model_choice]
        is_dl = model_path.suffix == '.pth'

        # Show model info
        if metrics_data and 'models' in metrics_data:
            mkey = model_path.stem.replace('_best', '')
            info_match = [m for m in metrics_data['models'] if m['model'] == mkey]
            if info_match:
                mi = info_match[0]
                st.caption(f"Accuracy: **{mi['accuracy']:.2%}** · Size: {mi.get('model_size_mb', 0):.0f} MB · ~{mi.get('inference_ms', 0)} ms")

        st.markdown("<br>", unsafe_allow_html=True)
        btn = st.button("🔍 Run Diagnosis", use_container_width=True, type="primary", disabled=pil_image is None)

    # ── Results panel ──────────────────────────────────────────────────
    with col_out:
        if btn and pil_image:
            device = get_device()
            with st.spinner("Analyzing leaf..."):
                try:
                    if is_dl:
                        model, mname, _ = load_pytorch_model(str(model_path))
                        if model is None:
                            st.stop()
                        preds, t = predict_dl(model, pil_image, device)
                    else:
                        ml = load_ml_model(str(model_path))
                        fe = get_ml_feature_extractor()
                        if ml is None or fe is None:
                            st.stop()
                        preds, t = predict_ml(ml, fe, pil_image, device)

                    top5_idx = np.argsort(preds)[-5:][::-1]
                    top5_probs = preds[top5_idx] * 100
                    top5_raw = [CLASS_NAMES[i] for i in top5_idx]
                    pred_raw = CLASS_NAMES[top5_idx[0]]
                    pred_fmt = format_class_name(pred_raw)
                    conf = top5_probs[0]

                    st.session_state['result'] = dict(
                        pred_raw=pred_raw, pred_fmt=pred_fmt, conf=conf,
                        inf_time=t, top5_probs=top5_probs, top5_raw=top5_raw,
                        model_choice=model_choice, is_dl=is_dl,
                        mname=mname if is_dl else None, pil_image=pil_image,
                    )
                except Exception as e:
                    st.error(f"Diagnosis failed: {e}")
                    st.code(traceback.format_exc())

        if 'result' in st.session_state:
            r = st.session_state['result']
            info = get_disease_info(r['pred_raw'])
            sev = info.get('severity', 'Medium')
            is_healthy = 'healthy' in r['pred_raw'].lower()

            # ── Diagnosis header ───────────────────────────────────────
            st.markdown(f"""
            <div class="result-card">
                <div style="display:flex; justify-content:space-between; align-items:flex-start;">
                    <div>
                        <div class="result-title">{"✅" if is_healthy else "⚠️"} {r['pred_fmt']}</div>
                        <div class="result-subtitle">{info.get('scientific_name', '')}</div>
                    </div>
                    <span class="severity-badge" style="background:{severity_color(sev)}">
                        {sev}
                    </span>
                </div>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("")

            # ── Metrics row ────────────────────────────────────────────
            m1, m2, m3 = st.columns(3)
            m1.metric("Confidence", f"{r['conf']:.1f}%")
            m2.metric("Inference", f"{r['inf_time']*1000:.0f} ms")
            m3.metric("Model", model_choice.split(' (')[0])

            st.markdown("")

            # ── Top-5 confidence chart ─────────────────────────────────
            fig = go.Figure()
            colors = ['#10b981' if i == 0 else '#1e293b' for i in range(len(r['top5_probs'][::-1]))]
            fig.add_trace(go.Bar(
                x=r['top5_probs'][::-1],
                y=[format_class_name(c) for c in r['top5_raw'][::-1]],
                orientation='h',
                marker_color=colors[::-1],
                text=[f"{v:.1f}%" for v in r['top5_probs'][::-1]],
                textposition='outside',
                textfont=dict(color='#94a3b8', size=11),
            ))
            fig.update_layout(
                height=220, margin=dict(l=0, r=40, t=0, b=0),
                plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
                xaxis=dict(visible=False, range=[0, max(r['top5_probs']) * 1.3]),
                yaxis=dict(tickfont=dict(color='#94a3b8', size=12)),
                showlegend=False,
            )
            st.plotly_chart(fig, use_container_width=True)

            # ── Grad-CAM ───────────────────────────────────────────────
            if r['is_dl'] and TORCH_AVAILABLE:
                st.markdown('<div class="section-label">Visual Explanation (Grad-CAM)</div>', unsafe_allow_html=True)
                cam = generate_gradcam(
                    load_pytorch_model(str(available_models[r['model_choice']]))[0],
                    r['mname'], r['pil_image'], get_device(),
                )
                if cam is not None:
                    gc1, gc2 = st.columns(2)
                    gc1.image(r['pil_image'], caption="Original", use_container_width=True)
                    gc2.image(cam, caption="AI Attention Heatmap", use_container_width=True)
                    st.caption("The heatmap highlights the regions the model focused on to make its diagnosis.")

            # ── Disease information tabs ───────────────────────────────
            st.markdown("")
            st.markdown('<div class="section-label">Disease Information</div>', unsafe_allow_html=True)
            t1, t2, t3 = st.tabs(["📋 Overview", "💊 Treatment", "🛡️ Prevention"])
            with t1:
                if info.get('description'):
                    st.markdown(info['description'])
                if info.get('symptoms'):
                    st.markdown(f"**Symptoms:** {info['symptoms']}")
                if info.get('causes'):
                    st.markdown(f"**Causes:** {info['causes']}")
            with t2:
                st.markdown(info.get('treatment', 'No treatment information available.'))
                if info.get('recommended_actions'):
                    st.markdown(f"**Recommended actions:** {info['recommended_actions']}")
            with t3:
                st.markdown(info.get('prevention', 'No prevention information available.'))

        elif not btn:
            st.markdown("""
            <div style="display:flex; align-items:center; justify-content:center; height:400px; text-align:center;">
                <div>
                    <div style="font-size:3rem; margin-bottom:16px; opacity:0.3;">🍃</div>
                    <div style="color:#475569; font-size:1rem; font-weight:500;">
                        Upload or select a leaf image, then click <strong>Run Diagnosis</strong>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
#  BATCH PREDICTION
# ══════════════════════════════════════════════════════════════════════════════

elif app_mode == "📦 Batch Analysis":
    st.markdown('<div class="page-title">Batch Analysis</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Upload multiple images for bulk diagnosis</div>', unsafe_allow_html=True)

    if not available_models:
        st.warning("No models found.")
        st.stop()

    mc = st.selectbox("Model", list(available_models.keys()), label_visibility="collapsed")
    mp = available_models[mc]
    is_dl = mp.suffix == '.pth'

    files = st.file_uploader("Upload images", type=["jpg", "jpeg", "png"], accept_multiple_files=True, label_visibility="collapsed")

    if files:
        st.caption(f"{len(files)} image{'s' if len(files) != 1 else ''} selected")

    if files and st.button("🚀 Analyze All", type="primary"):
        device = get_device()
        if is_dl:
            model, _, _ = load_pytorch_model(str(mp))
        else:
            ml = load_ml_model(str(mp))
            fe = get_ml_feature_extractor()

        results = []
        bar = st.progress(0, text="Analyzing images...")
        for i, f in enumerate(files):
            try:
                img = Image.open(f)
                if is_dl:
                    p, t = predict_dl(model, img, device)
                else:
                    p, t = predict_ml(ml, fe, img, device)
                idx = np.argmax(p)
                results.append({
                    'File': f.name,
                    'Diagnosis': format_class_name(CLASS_NAMES[idx]),
                    'Confidence': f"{p[idx]*100:.1f}%",
                    'Time (ms)': f"{t*1000:.0f}",
                })
            except Exception as e:
                results.append({'File': f.name, 'Diagnosis': f'Error: {e}', 'Confidence': '—', 'Time (ms)': '—'})
            bar.progress((i + 1) / len(files), text=f"Analyzing {i+1}/{len(files)}...")
        bar.empty()

        st.markdown('<div class="section-label">Results</div>', unsafe_allow_html=True)
        df = pd.DataFrame(results)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.download_button("📥 Download CSV", df.to_csv(index=False), "diagnosis_report.csv", "text/csv")


# ══════════════════════════════════════════════════════════════════════════════
#  MODEL COMPARISON
# ══════════════════════════════════════════════════════════════════════════════

elif app_mode == "📊 Models":
    st.markdown('<div class="page-title">Model Dashboard</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Compare accuracy, speed, and size across all trained models</div>', unsafe_allow_html=True)

    comp = load_comparison()

    if comp is not None and not comp.empty:
        # ── Best model highlight ───────────────────────────────────────
        best_row = comp.iloc[0]  # Already sorted by accuracy desc
        st.markdown(f"""
        <div class="card" style="border-color: rgba(16,185,129,0.2); margin-bottom: 24px;">
            <div style="display:flex; align-items:center; gap:16px;">
                <div style="font-size:2rem;">🏆</div>
                <div>
                    <h4 style="margin:0; color:#10b981;">Recommended: {best_row.get('display_name', best_row['model'])}</h4>
                    <p style="margin:4px 0 0 0;">{best_row['accuracy']:.4%} accuracy · {best_row.get('model_size_mb', 0):.0f} MB · ~{best_row.get('inference_ms', 0)} ms inference</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ── Accuracy chart ─────────────────────────────────────────────
        st.markdown('<div class="section-label">Validation Accuracy</div>', unsafe_allow_html=True)
        comp_sorted = comp.sort_values('accuracy', ascending=True)
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=comp_sorted['accuracy'] * 100,
            y=comp_sorted.get('display_name', comp_sorted['model']),
            orientation='h',
            marker_color=['#10b981' if a == comp['accuracy'].max() else '#1e3a5f' for a in comp_sorted['accuracy']],
            text=[f"{a:.2f}%" for a in comp_sorted['accuracy'] * 100],
            textposition='outside',
            textfont=dict(color='#94a3b8', size=12),
        ))
        fig.update_layout(
            height=280, margin=dict(l=0, r=60, t=0, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(visible=False, range=[99, 100.2]),
            yaxis=dict(tickfont=dict(color='#cbd5e1', size=13)),
            showlegend=False,
        )
        st.plotly_chart(fig, use_container_width=True)

        # ── Comparison table ───────────────────────────────────────────
        st.markdown('<div class="section-label">Detailed Comparison</div>', unsafe_allow_html=True)
        display_df = comp.copy()
        display_df['accuracy'] = display_df['accuracy'].apply(lambda x: f"{x:.4%}")
        display_df['total_params'] = display_df['total_params'].apply(lambda x: f"{x/1e6:.1f}M")
        display_df['model_size_mb'] = display_df['model_size_mb'].apply(lambda x: f"{x:.0f} MB")
        display_df['inference_ms'] = display_df['inference_ms'].apply(lambda x: f"~{x} ms")
        rename = {
            'display_name': 'Model', 'accuracy': 'Accuracy',
            'total_params': 'Parameters', 'model_size_mb': 'Size',
            'inference_ms': 'Speed', 'notes': 'Notes',
        }
        show_cols = [c for c in rename.keys() if c in display_df.columns]
        st.dataframe(
            display_df[show_cols].rename(columns=rename),
            use_container_width=True, hide_index=True,
        )

        # ── Size vs Speed scatter ──────────────────────────────────────
        st.markdown('<div class="section-label">Size vs Speed Tradeoff</div>', unsafe_allow_html=True)
        fig2 = px.scatter(
            comp,
            x='model_size_mb', y='inference_ms',
            size='accuracy', text=comp.get('display_name', comp['model']),
            labels={'model_size_mb': 'Model Size (MB)', 'inference_ms': 'Inference Time (ms)'},
        )
        fig2.update_traces(
            marker=dict(color='#10b981', line=dict(width=1, color='#059669')),
            textposition='top center', textfont=dict(color='#94a3b8', size=11),
        )
        fig2.update_layout(
            height=320, margin=dict(l=0, r=0, t=20, b=0),
            plot_bgcolor='rgba(0,0,0,0)', paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#64748b')),
            yaxis=dict(gridcolor='rgba(255,255,255,0.03)', tickfont=dict(color='#64748b')),
            showlegend=False,
        )
        st.plotly_chart(fig2, use_container_width=True)

    else:
        st.info("Run the training notebook to generate model comparison data.")


# ══════════════════════════════════════════════════════════════════════════════
#  HELP
# ══════════════════════════════════════════════════════════════════════════════

elif app_mode == "❓ Help":
    st.markdown('<div class="page-title">Help & Documentation</div>', unsafe_allow_html=True)
    st.markdown('<div class="page-subtitle">Everything you need to know about using this application</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-label">Quick Start</div>', unsafe_allow_html=True)
    st.markdown("""
    1. Go to **Diagnose** in the sidebar
    2. Upload a clear photo of a plant leaf (JPG, PNG)
    3. The best model is selected automatically — click **Run Diagnosis**
    4. Review the diagnosis, attention heatmap, and treatment recommendations
    """)

    st.markdown("")
    st.markdown('<div class="section-label">Available Models</div>', unsafe_allow_html=True)

    model_data = {
        'Model': ['EfficientNet-V2-S', 'EfficientNet-B0', 'ConvNeXt-Tiny', 'ResNet-50', 'Custom CNN', 'SVM'],
        'Type': ['Deep Learning', 'Deep Learning', 'Deep Learning', 'Deep Learning', 'Deep Learning', 'Classical ML'],
        'Best For': [
            'Highest accuracy',
            'Best speed/accuracy balance',
            'Modern architecture research',
            'Established benchmark',
            'Lightweight baseline',
            'Non-GPU environments',
        ],
    }
    st.dataframe(pd.DataFrame(model_data), use_container_width=True, hide_index=True)

    st.markdown("")
    st.markdown('<div class="section-label">Frequently Asked Questions</div>', unsafe_allow_html=True)

    with st.expander("Which model should I use?"):
        st.markdown("The app automatically selects the most accurate model (marked ★ Recommended). For faster inference on CPU, try EfficientNet-B0 or Custom CNN.")

    with st.expander("What image quality is needed?"):
        st.markdown("A clear, well-lit photo of a single leaf works best. Avoid blurry images, photos with multiple leaves, or images where the leaf is very small in the frame.")

    with st.expander("How was the model trained?"):
        st.markdown("All deep learning models were trained end-to-end on the PlantVillage dataset (~87,000 images, 38 classes) using PyTorch with mixed-precision training on an NVIDIA RTX 3060 Ti GPU.")

    with st.expander("Can I retrain the models?"):
        st.markdown("Yes. Open `plant_disease.ipynb` in Jupyter with the `plant-ai` kernel and run all cells. Training takes approximately 2-3 hours on a modern GPU.")