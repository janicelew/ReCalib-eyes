import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import clip

torch.manual_seed(42)
np.random.seed(42)

# ==================== 2. Config ====================
class Config:
    CSV_PATH = "/kaggle/input/datasets/nadaol0/messidor2/messidor_2.csv"
    IMAGE_DIR = "/kaggle/input/datasets/nadaol0/messidor2/images"
    OUTPUT_DIR = "/kaggle/working/results"
    CSV_OUTPUT = os.path.join(OUTPUT_DIR, "messidor2_linear_probe_predictions.csv")
    METRICS_OUTPUT = os.path.join(OUTPUT_DIR, "messidor2_linear_probe_metrics.txt")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    CLIP_MODEL_TYPE = "ViT-B/32"
    EYECLIP_WEIGHTS = "/kaggle/input/datasets/selen917/eyeclip/eyeclip (1).pt"
    CV_FOLDS = 5
    RANDOM_SEED = 42
    C = 1.0
    CLASS_WEIGHT = "balanced"


# ==================== 3. Dataset ====================
class Messidor2Dataset(Dataset):
    def __init__(self, csv_path, img_dir, transform=None):
        self.img_dir = img_dir
        self.transform = transform
        try:
            self.df = pd.read_csv(csv_path, sep=None, engine='python')
        except Exception:
            self.df = pd.read_csv(csv_path, sep=',')
        self.img_col = 'image_path' if 'image_path' in self.df.columns else self.df.columns[1]
        self.label_col = 'diagnosis' if 'diagnosis' in self.df.columns else self.df.columns[2]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_filename = os.path.basename(str(row[self.img_col]))
        img_path = os.path.join(self.img_dir, img_filename)
        label = int(float(row[self.label_col]))
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, label, img_filename


# ==================== 4. Load Model ====================
def load_eyeclip_model(config):
    model, preprocess = clip.load(config.CLIP_MODEL_TYPE, device=config.DEVICE)
    if config.EYECLIP_WEIGHTS and os.path.exists(config.EYECLIP_WEIGHTS):
        checkpoint = torch.load(config.EYECLIP_WEIGHTS, map_location=config.DEVICE)
        state_dict = None
        for key in ["model_state_dict", "state_dict", "model"]:
            if isinstance(checkpoint, dict) and key in checkpoint:
                state_dict = checkpoint[key]
                break
        if state_dict is None:
            state_dict = checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            name = k
            if name.startswith("module."):
                name = name[7:]
            if name.startswith("model."):
                name = name[6:]
            new_state_dict[name] = v
        has_visual_prefix = any(k.startswith("visual.") for k in new_state_dict.keys())
        if has_visual_prefix:
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.visual.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, preprocess


# ==================== 5. Feature extraction ====================
def extract_image_features(model, dataloader, config):
    all_features, all_grades, all_filenames = [], [], []
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader):
            images = images.to(config.DEVICE)
            image_features = model.encode_image(images)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            all_features.append(image_features.cpu().numpy())
            all_grades.append(labels.numpy())
            all_filenames.extend(filenames)
    return (
        np.concatenate(all_features, axis=0),
        np.concatenate(all_grades, axis=0),
        all_filenames,
    )


# ==================== 6. Linear probe (source-only, 5-fold OOF) ====================
def make_probe(config):
    class_weight = None if config.CLASS_WEIGHT == "none" else config.CLASS_WEIGHT
    return make_pipeline(
        StandardScaler(),
        LogisticRegression(
            C=config.C,
            class_weight=class_weight,
            max_iter=2000,
            solver="lbfgs",
            random_state=config.RANDOM_SEED,
        ),
    )


def fit_oof_linear_probe(features, labels, config):
    labels = np.asarray(labels, dtype=int)
    min_class_count = int(np.bincount(labels).min())
    n_folds = max(2, min(config.CV_FOLDS, min_class_count))
    splitter = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=config.RANDOM_SEED)

    oof_probs = np.zeros(len(labels), dtype=float)
    for train_idx, valid_idx in splitter.split(features, labels):
        probe = make_probe(config)
        probe.fit(features[train_idx], labels[train_idx])
        oof_probs[valid_idx] = probe.predict_proba(features[valid_idx])[:, 1]
    return oof_probs, n_folds


# ==================== 7. Metrics ====================
def calculate_metrics(y_true_grade, prob_any_dr, prob_rdr):
    metrics = {}

    y_any_dr = (y_true_grade >= 1).astype(int)
    pred_any_dr = (prob_any_dr >= 0.5).astype(int)
    metrics['any_dr_auroc'] = roc_auc_score(y_any_dr, prob_any_dr)
    metrics['any_dr_aupr'] = average_precision_score(y_any_dr, prob_any_dr)
    metrics['any_dr_f1'] = f1_score(y_any_dr, pred_any_dr)
    metrics['any_dr_accuracy'] = accuracy_score(y_any_dr, pred_any_dr)

    y_rdr = (y_true_grade >= 2).astype(int)
    pred_rdr = (prob_rdr >= 0.5).astype(int)
    metrics['rdr_auroc'] = roc_auc_score(y_rdr, prob_rdr)
    metrics['rdr_aupr'] = average_precision_score(y_rdr, prob_rdr)
    metrics['rdr_f1'] = f1_score(y_rdr, pred_rdr)
    metrics['rdr_accuracy'] = accuracy_score(y_rdr, pred_rdr)

    return metrics, y_any_dr, y_rdr


# ==================== 8. Save ====================
def save_evaluation_results(filenames, y_true_grade, y_any_dr, prob_any_dr, y_rdr, prob_rdr,
                             metrics, n_folds_any, n_folds_rdr, config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df_results = pd.DataFrame({
        'image_id': filenames,
        'true_grade': y_true_grade,
        'true_any_dr': y_any_dr,
        'prob_any_dr_linear_probe': prob_any_dr,
        'true_rdr': y_rdr,
        'prob_rdr_linear_probe': prob_rdr,
    })
    df_results.to_csv(config.CSV_OUTPUT, index=False)

    with open(config.METRICS_OUTPUT, 'w') as f:
        f.write("=== MESSIDOR2 Linear Probe Baseline Metrics (EyeCLIP features + Logistic Regression) ===\n")
        f.write(f"cv_folds_any_dr: {n_folds_any}\n")
        f.write(f"cv_folds_rdr: {n_folds_rdr}\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n" if not np.isnan(v) else f"{k}: NaN\n")


def main():
    config = Config()
    model, preprocess = load_eyeclip_model(config)
    dataset = Messidor2Dataset(csv_path=config.CSV_PATH, img_dir=config.IMAGE_DIR, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)

    features, grades, filenames = extract_image_features(model, dataloader, config)

    prob_any_dr, n_folds_any = fit_oof_linear_probe(features, (grades >= 1).astype(int), config)
    prob_rdr, n_folds_rdr = fit_oof_linear_probe(features, (grades >= 2).astype(int), config)

    metrics, y_any_dr, y_rdr = calculate_metrics(grades, prob_any_dr, prob_rdr)
    save_evaluation_results(
        filenames, grades, y_any_dr, prob_any_dr, y_rdr, prob_rdr,
        metrics, n_folds_any, n_folds_rdr, config,
    )
    print(metrics)


if __name__ == '__main__':
    main()
