#!/usr/bin/env python3
import io, sys, zipfile, random, argparse
from pathlib import Path
import numpy as np, pandas as pd
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision
from torchvision import transforms, models

parser = argparse.ArgumentParser()
parser.add_argument("--zip", default="./FashionD.zip")
parser.add_argument("--weights", default="./resnet18_fitcheck_from_zip.pth")
parser.add_argument("--max-samples", type=int, default=5000)
parser.add_argument("--batch-size", type=int, default=32)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument("--train", action="store_true")
parser.add_argument("--extract-sampled", action="store_true")
parser.add_argument("--image", default=None)
parser.add_argument("--zip-image", default=None)
parser.add_argument("--diagnose", action="store_true")
args = parser.parse_args()

ARCHIVE_PATH = Path(args.zip)
WEIGHTS_PATH = Path(args.weights)
MAX_SAMPLES = args.max_samples
BATCH_SIZE = args.batch_size
NUM_WORKERS = args.num_workers
EXTRACT_SAMPLED = args.extract_sampled
EXTRACT_DIR = Path("./_fitcheck_cache/images")

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

def log(*a, **k):
    print(*a, **k, flush=True)

if not ARCHIVE_PATH.exists():
    log("Archive not found:", ARCHIVE_PATH.resolve()); sys.exit(1)

with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
    members = zf.namelist()
    styles_paths = [m for m in members if m.lower().endswith("styles.csv")]
    if not styles_paths:
        log("styles.csv missing in zip"); sys.exit(1)
    styles_csv = min(styles_paths, key=len)
    with zf.open(styles_csv) as f:
        txt = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        styles_df = pd.read_csv(txt, on_bad_lines="skip", low_memory=False)
    images_prefixes = set()
    for n in members:
        low = n.lower()
        if "/images/" in low:
            i = low.index("/images/")
            images_prefixes.add(n[: i + len("/images/")])
    if not images_prefixes:
        parent_counts = {}
        for n in members:
            if n.lower().endswith((".jpg", ".jpeg", ".png")):
                parent = n.rsplit("/", 1)[0] + "/"
                parent_counts[parent] = parent_counts.get(parent, 0) + 1
        if parent_counts:
            images_prefixes.add(max(parent_counts.items(), key=lambda x: x[1])[0])
    if not images_prefixes:
        log("No images prefix detected"); sys.exit(1)
    IMAGES_DIR_ZIP_PREFIX = min(images_prefixes, key=len)
    image_members = [m for m in members if m.startswith(IMAGES_DIR_ZIP_PREFIX) and m.lower().endswith(('.jpg', '.jpeg', '.png'))]
    image_member_set = set(image_members)

styles_df['image_filename'] = styles_df['id'].astype(str) + ".jpg"
with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
    all_members = set(zf.namelist())

def find_member_for_filename(fn):
    candidates = [
        IMAGES_DIR_ZIP_PREFIX + fn,
        IMAGES_DIR_ZIP_PREFIX + fn.lower(),
        IMAGES_DIR_ZIP_PREFIX + fn.upper(),
        fn, "images/" + fn,
        "fashion-dataset/images/" + fn,
        "fashion_dataset/images/" + fn,
    ]
    for c in candidates:
        if c in all_members: return c
    suffix = "/" + fn
    for mem in all_members:
        if mem.endswith(suffix): return mem
    return None

styles_df['zip_path'] = styles_df['image_filename'].apply(find_member_for_filename)
styles_df['exists_in_zip'] = styles_df['zip_path'].notnull()
df = styles_df[styles_df['exists_in_zip']].reset_index(drop=True)
if len(df) == 0:
    log("No images matched styles.csv"); sys.exit(1)

df['fit_rating'] = np.random.randint(1, 101, size=len(df))
if len(df) > MAX_SAMPLES:
    df = df.sample(MAX_SAMPLES, random_state=RANDOM_SEED).reset_index(drop=True)
m = int(0.9 * len(df))
train_df = df.iloc[:m].reset_index(drop=True)
val_df = df.iloc[m:].reset_index(drop=True)

if EXTRACT_SAMPLED:
    EXTRACT_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(ARCHIVE_PATH, "r") as zf:
        for split in (train_df, val_df):
            for _, row in split.iterrows():
                out = EXTRACT_DIR / row['image_filename']
                if out.exists(): continue
                with zf.open(row['zip_path']) as src, open(out, "wb") as dst:
                    dst.write(src.read())

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

train_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
val_tfms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

class ZipImageDataset(Dataset):
    def __init__(self, df, archive_path, tfm, meta_cols=('articleType',)):
        self.df = df.reset_index(drop=True)
        self.archive_path = str(archive_path)
        self.tfm = tfm
        self.meta_cols = meta_cols
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        zpath = row['zip_path']
        with zipfile.ZipFile(self.archive_path, 'r') as zf:
            with zf.open(zpath) as f:
                img_bytes = f.read()
        img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor([float(row['fit_rating'])], dtype=torch.float32)
        meta = {c: row[c] if c in row else "" for c in self.meta_cols}
        meta['image_filename'] = row['image_filename']
        return x, y, meta

class FolderImageDataset(Dataset):
    def __init__(self, df, folder, tfm, meta_cols=('articleType',)):
        self.df = df.reset_index(drop=True)
        self.folder = Path(folder)
        self.tfm = tfm
        self.meta_cols = meta_cols
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self.folder / row['image_filename']
        img = Image.open(img_path).convert("RGB")
        x = self.tfm(img)
        y = torch.tensor([float(row['fit_rating'])], dtype=torch.float32)
        meta = {c: row[c] if c in row else "" for c in self.meta_cols}
        meta['image_filename'] = row['image_filename']
        return x, y, meta

if EXTRACT_SAMPLED:
    train_ds = FolderImageDataset(train_df, EXTRACT_DIR, train_tfms)
    val_ds = FolderImageDataset(val_df, EXTRACT_DIR, val_tfms)
else:
    train_ds = ZipImageDataset(train_df, ARCHIVE_PATH, train_tfms)
    val_ds = ZipImageDataset(val_df, ARCHIVE_PATH, val_tfms)

train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS, pin_memory=True)
val_dl = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
log("Device:", device)
model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
model.fc = nn.Linear(model.fc.in_features, 1)
model = model.to(device)

if WEIGHTS_PATH.exists() and not args.train:
    try:
        model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=device))
        log("Loaded weights:", WEIGHTS_PATH)
    except Exception as e:
        log("Failed to load weights:", e); sys.exit(1)
elif not WEIGHTS_PATH.exists() and not args.train:
    log("Weights missing — run with --train to create them"); sys.exit(0)

model.eval()

a, b = 1.0, 0.0
preds_list, labels_list, filenames = [], [], []
for xb, yb, meta in val_dl:
    with torch.no_grad():
        out = model(xb.to(device)).cpu().squeeze(-1).numpy()
    preds_list.append(out)
    labels_list.append(yb.cpu().numpy().squeeze(-1))
    bs = xb.shape[0]
    batch_fnames = []
    if isinstance(meta, list):
        if len(meta) > 0 and isinstance(meta[0], dict):
            batch_fnames = [m.get("image_filename","") for m in meta]
        else:
            batch_fnames = [str(m) for m in meta]
    elif isinstance(meta, tuple):
        for e in meta:
            if isinstance(e, dict): batch_fnames.append(e.get("image_filename",""))
            else: batch_fnames.append(str(e))
    elif isinstance(meta, dict):
        batch_fnames = [meta.get("image_filename","")] * bs
    else:
        batch_fnames = [str(meta)] * bs
    if len(batch_fnames) != bs:
        if len(batch_fnames) < bs and len(batch_fnames) > 0:
            batch_fnames = (batch_fnames * ((bs // len(batch_fnames)) + 1))[:bs]
        else:
            batch_fnames = batch_fnames[:bs] + [""] * max(0, bs - len(batch_fnames))
    filenames += batch_fnames

preds = np.concatenate(preds_list) if len(preds_list) else np.array([])
labels = np.concatenate(labels_list) if len(labels_list) else np.array([])

log("val n:", len(labels))
log("label mean/std/min/max: {:.3f}/{:.3f}/{:.1f}/{:.1f}".format(labels.mean(), labels.std(), labels.min(), labels.max()))
log("pred mean/std/min/max: {:.3f}/{:.3f}/{:.3f}/{:.3f}".format(preds.mean(), preds.std(), preds.min(), preds.max()))

if len(preds) > 5:
    a, b = np.polyfit(preds, labels, 1)
    preds_rescaled = a * preds + b
    log("Calibration: label ≈ {:.6f}*pred + {:.6f}".format(a, b))
    abs_err = np.abs(preds_rescaled - labels)
    idx = np.argsort(abs_err)[::-1]
    log("Top mismatches:")
    for i in idx[:10]:
        fname = filenames[i] if i < len(filenames) else "<no-file>"
        log(f"{i}: pred_rescaled={preds_rescaled[i]:6.1f}, label={labels[i]:6.1f}, err={abs_err[i]:6.1f}, file={fname}")
    mae = np.mean(np.abs(preds_rescaled - labels))
    rmse = np.sqrt(np.mean((preds_rescaled - labels)**2))
    log("MAE:", mae, "RMSE:", rmse)

def predict_raw(img_t):
    with torch.no_grad():
        return model(img_t.unsqueeze(0).to(device)).item()

def predict_score(img_t):
    raw = predict_raw(img_t)
    calibrated = a * raw + b
    return float(max(1.0, min(100.0, calibrated)))

def load_image_from_path(path, tfm=val_tfms):
    p = Path(path)
    if not p.exists(): raise FileNotFoundError(path)
    img = Image.open(p).convert("RGB")
    return tfm(img)

def load_image_from_zip(name, archive_path=ARCHIVE_PATH, tfm=val_tfms):
    name = str(name)
    if not name.lower().endswith(('.jpg','.jpeg','.png')): name = name + ".jpg"
    candidate = []
    with zipfile.ZipFile(archive_path, "r") as zf:
        for mem in zf.namelist():
            low = mem.lower()
            if low.endswith('/' + name.lower()) or low.endswith('/' + name):
                candidate.append(mem)
        if not candidate:
            suffix = '/' + name
            for mem in zf.namelist():
                if mem.lower().endswith(suffix.lower()): candidate.append(mem)
        if not candidate: raise FileNotFoundError(name)
        member = candidate[0]
        with zf.open(member) as f:
            img_bytes = f.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    return tfm(img)

if args.image or args.zip_image:
    try:
        if args.image:
            t = load_image_from_path(args.image)
            src = args.image
        else:
            t = load_image_from_zip(args.zip_image)
            src = f"zip:{args.zip_image}"
    except Exception as e:
        log("Failed to load image:", e); sys.exit(1)
    raw = predict_raw(t)
    score = predict_score(t)
    log("Source:", src)
    log("Raw:", f"{raw:.4f}", "Calibrated score:", f"{score:.1f}/100")
    def short_verdict(s):
        if s >= 85: return "🔥 Runway Ready"
        if s >= 70: return "😎 Clean Fit"
        if s >= 50: return "🙂 Decent"
        if s >= 35: return "😬 Mid"
        return "🆘 Fashion Emergency"
    log("Verdict:", short_verdict(score))
    sys.exit(0)

def show_preview(ds, n=6):
    n = min(n, len(ds))
    if n == 0: return
    idxs = np.random.choice(len(ds), size=n, replace=False)
    for i in idxs:
        x, y, meta = ds[i]
        pred = predict_score(x)
        if isinstance(meta, dict):
            name = meta.get('articleType','') or 'item'
            fname = meta.get('image_filename','')
        elif isinstance(meta, list):
            first = meta[0] if len(meta) > 0 else {}
            if isinstance(first, dict):
                name = first.get('articleType','') or 'item'
                fname = first.get('image_filename','')
            else:
                name = 'item'; fname = str(first)
        else:
            name = 'item'; fname = str(meta)
        log(f" - {name[:12]:12s} | Pred: {pred:6.1f} / 100 | file: {fname}")

show_preview(val_ds, n=6)

# Ratings
def verdict(score):
    if score >= 85: return "🔥 Runway Ready — Maximum drip!"
    if score >= 70: return "😎 Clean Fit — Campus slay."
    if score >= 50: return "🙂 Decent — Solid casual look."
    if score >= 35: return "😬 Mid — Needs a lil tweak."
    return "🆘 Fashion Emergency — Call a stylist!"

pool = val_ds if len(val_ds) > 0 else train_ds
k = min(3, len(pool))
if k > 0:
    idxs = np.random.choice(len(pool), size=k, replace=False)
    parts, scores = [], []
    for i in idxs:
        x, y, meta = pool[i]
        s = predict_score(x)
        if isinstance(meta, dict):
            cat = meta.get('articleType','item'); fname = meta.get('image_filename','')
        elif isinstance(meta, list):
            first = meta[0] if len(meta) > 0 else {}
            if isinstance(first, dict):
                cat = first.get('articleType','item'); fname = first.get('image_filename','')
            else:
                cat = 'item'; fname = str(first)
        else:
            cat = 'item'; fname = str(meta)
        parts.append((cat, s, fname)); scores.append(s)
    final = float(np.mean(scores))
    log("\nFit-o-Meter Parts & Scores:")
    for cat, s, fname in parts:
        log(f" - {cat[:12]:12s}: {s:5.1f}/100 | {fname}")
    log("\nFinal Fit Rating:", f"{final:.1f}/100")
    log("Verdict:", verdict(final))

log("Done.")
