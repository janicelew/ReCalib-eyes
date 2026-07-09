import os
import torch
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score, accuracy_score
import clip

# 设置随机种子
torch.manual_seed(42)
np.random.seed(42)

# ==================== 2. Config ====================
class Config:
    CSV_PATH = "/kaggle/input/datasets/nadaol0/messidor2/messidor_2.csv"
    IMAGE_DIR = "/kaggle/input/datasets/nadaol0/messidor2/images"
    OUTPUT_DIR = "/kaggle/working/results"
    CSV_OUTPUT = os.path.join(OUTPUT_DIR, "messidor2_predictions.csv")
    METRICS_OUTPUT = os.path.join(OUTPUT_DIR, "messidor2_metrics.txt")
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    BATCH_SIZE = 32
    NUM_WORKERS = 4
    CLIP_MODEL_TYPE = "ViT-B/32" 
    EYECLIP_WEIGHTS = "/kaggle/input/datasets/selen917/eyeclip/eyeclip (1).pt" 
    PROMPTS = ["normal retina", "diabetic retinopathy"]

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
            if name.startswith("module."): name = name[7:]
            if name.startswith("model."): name = name[6:]
            new_state_dict[name] = v
        has_visual_prefix = any(k.startswith("visual.") for k in new_state_dict.keys())
        if has_visual_prefix:
            model.load_state_dict(new_state_dict, strict=False)
        else:
            model.visual.load_state_dict(new_state_dict, strict=False)
    model.eval()
    return model, preprocess

# ==================== 5. Inference ====================
def run_zero_shot_inference(model, dataloader, config):
    text_inputs = clip.tokenize(config.PROMPTS).to(config.DEVICE)
    with torch.no_grad():
        text_features = model.encode_text(text_inputs)
        text_features /= text_features.norm(dim=-1, keepdim=True)
    all_predictions, all_targets, all_filenames = [], [], []
    with torch.no_grad():
        for images, labels, filenames in tqdm(dataloader):
            images = images.to(config.DEVICE)
            image_features = model.encode_image(images)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            all_predictions.append(similarity.cpu().numpy())
            all_targets.append(labels.numpy())
            all_filenames.extend(filenames)
    return all_filenames, np.concatenate(all_predictions, axis=0), np.concatenate(all_targets, axis=0)

# ==================== 6. Metrics ====================
def calculate_metrics(y_true, y_probs):
    metrics = {}
    y_probs = y_probs / np.sum(y_probs, axis=1, keepdims=True)
    prob_dr = y_probs[:, 1]
    pred_binary = np.argmax(y_probs, axis=1)
    
    y_true_any_dr = (y_true >= 1).astype(int)
    metrics['any_dr_auroc'] = roc_auc_score(y_true_any_dr, prob_dr)
    metrics['any_dr_aupr'] = average_precision_score(y_true_any_dr, prob_dr)
    metrics['any_dr_f1'] = f1_score(y_true_any_dr, pred_binary)
    metrics['any_dr_accuracy'] = accuracy_score(y_true_any_dr, pred_binary)
    
    y_true_rdr = (y_true >= 2).astype(int)
    metrics['rdr_auroc'] = roc_auc_score(y_true_rdr, prob_dr)
    metrics['rdr_aupr'] = average_precision_score(y_true_rdr, prob_dr)
    return metrics, y_true_any_dr, prob_dr

# ==================== 7. Save ====================
def save_evaluation_results(filenames, y_true, y_probs, y_true_binary, y_probs_binary, metrics, config):
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    df_results = pd.DataFrame({
        'image_id': filenames, 'true_grade': y_true,
        'prob_normal': y_probs[:, 0], 'prob_dr': y_probs[:, 1], 'true_any_dr': y_true_binary
    })
    df_results.to_csv(config.CSV_OUTPUT, index=False)
    with open(config.METRICS_OUTPUT, 'w') as f:
        f.write("=== MESSIDOR2 Zero-Shot Baseline Metrics (Binary) ===\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v:.4f}\n" if not np.isnan(v) else f"{k}: NaN\n")

def main():
    config = Config()
    model, preprocess = load_eyeclip_model(config)
    dataset = Messidor2Dataset(csv_path=config.CSV_PATH, img_dir=config.IMAGE_DIR, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=config.NUM_WORKERS)
    filenames, y_probs, y_true = run_zero_shot_inference(model, dataloader, config)
    metrics, y_true_binary, prob_dr = calculate_metrics(y_true, y_probs)
    save_evaluation_results(filenames, y_true, y_probs, y_true_binary, prob_dr, metrics, config)

if __name__ == '__main__':
    main()
