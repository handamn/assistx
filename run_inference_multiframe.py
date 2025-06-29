import os
import json
import torch
import clip
from PIL import Image
from tqdm import tqdm
from utils import load_classnames, extract_multiple_frames, preprocess_image, compute_topk_accuracy

DATASET_DIR = "dataset"
FRAME_DIR = "frames"
CLASSNAMES_FILE = "classnames.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

print("Loading CLIP model...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

classnames = load_classnames(CLASSNAMES_FILE)
prompts = [f"a photo of a person doing {name.replace('_', ' ')}" for name in classnames]
text_tokens = clip.tokenize(prompts).to(DEVICE)
text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)

predictions = []
y_true, y_pred = [], []

print("Running multi-frame inference...")
for cls_name in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls_name)
    if not os.path.isdir(cls_path): continue
    for vid_name in os.listdir(cls_path):
        vid_path = os.path.join(cls_path, vid_name)
        frame_paths = extract_multiple_frames(vid_path, FRAME_DIR)

        # Dapatkan embedding untuk setiap frame, lalu rata-rata
        image_features_list = []
        for frame_path in frame_paths:
            image = preprocess_image(frame_path, preprocess, DEVICE)
            with torch.no_grad():
                image_feat = model.encode_image(image)
                image_feat /= image_feat.norm(dim=-1, keepdim=True)
                image_features_list.append(image_feat)

        image_features = torch.stack(image_features_list).mean(dim=0)
        similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        topk = similarity.topk(TOP_K, dim=-1)

        pred_idx = topk.indices[0, 0].item()
        predictions.append({
            "video": vid_path,
            "true_label": cls_name,
            "top1_pred": classnames[pred_idx],
            "top5_pred": [classnames[i] for i in topk.indices[0].tolist()],
            "top5_score": [round(score, 3) for score in topk.values[0].tolist()]
        })

        y_true.append(cls_name)
        y_pred.append(classnames[pred_idx])

# Save results
os.makedirs("results", exist_ok=True)
with open("results/predictions_multiframe.json", "w") as f:
    json.dump(predictions, f, indent=2)

acc1, acc5 = compute_topk_accuracy(y_true, predictions)
print(f"[Multi-Frame] Top-1 Accuracy: {acc1:.2f}%")
print(f"[Multi-Frame] Top-5 Accuracy: {acc5:.2f}%")
