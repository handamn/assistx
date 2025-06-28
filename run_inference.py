import os, json, torch, clip
from PIL import Image
from tqdm import tqdm
from utils import load_classnames, extract_frame, preprocess_image, compute_topk_accuracy

DATASET_DIR = "dataset"
FRAME_DIR = "frames"
CLASSNAMES_FILE = "classnames.txt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
TOP_K = 5

print("Loading CLIP...")
model, preprocess = clip.load("ViT-B/32", device=DEVICE)

classnames = load_classnames(CLASSNAMES_FILE)
prompts = [f"a photo of a person doing {name.replace('_', ' ')}" for name in classnames]
text_tokens = clip.tokenize(prompts).to(DEVICE)
text_features = model.encode_text(text_tokens)
text_features /= text_features.norm(dim=-1, keepdim=True)

predictions, y_true, y_pred = [], [], []

print("Running inference...")
for cls_name in os.listdir(DATASET_DIR):
    cls_path = os.path.join(DATASET_DIR, cls_name)
    if not os.path.isdir(cls_path): continue
    for vid_name in os.listdir(cls_path):
        vid_path = os.path.join(cls_path, vid_name)
        frame_path = extract_frame(vid_path, FRAME_DIR)
        image = preprocess_image(frame_path, preprocess, DEVICE)

        with torch.no_grad():
            image_features = model.encode_image(image)
            image_features /= image_features.norm(dim=-1, keepdim=True)
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

os.makedirs("results", exist_ok=True)
with open("results/predictions.json", "w") as f:
    json.dump(predictions, f, indent=2)

acc1, acc5 = compute_topk_accuracy(y_true, predictions)
print(f"Top-1 Accuracy: {acc1:.2f}%")
print(f"Top-5 Accuracy: {acc5:.2f}%")
