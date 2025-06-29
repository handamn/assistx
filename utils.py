import os, cv2
import torch
from PIL import Image
from sklearn.metrics import accuracy_score

def load_classnames(filename):
    with open(filename, 'r') as f:
        return [line.strip() for line in f.readlines() if line.strip()]

def extract_frame(video_path, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_path = os.path.join(out_dir, f"{video_id}.jpg")
    if os.path.exists(out_path): return out_path

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    mid_frame = frame_count // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, mid_frame)
    ret, frame = cap.read()
    if ret: cv2.imwrite(out_path, frame)
    cap.release()
    return out_path

def preprocess_image(image_path, preprocess_fn, device):
    image = Image.open(image_path).convert("RGB")
    image = preprocess_fn(image).unsqueeze(0).to(device)
    return image

def compute_topk_accuracy(y_true, predictions, k=5):
    y_pred_top1 = [pred['top1_pred'] for pred in predictions]
    acc1 = accuracy_score(y_true, y_pred_top1) * 100
    acc5_count = sum([true in pred['top5_pred'] for true, pred in zip(y_true, predictions)])
    acc5 = (acc5_count / len(y_true)) * 100
    return acc1, acc5

def extract_multiple_frames(video_path, out_dir, positions=[0.1, 0.5, 0.9]):
    os.makedirs(out_dir, exist_ok=True)
    video_id = os.path.splitext(os.path.basename(video_path))[0]
    out_paths = []

    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    for i, p in enumerate(positions):
        frame_idx = int(p * frame_count)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        
        frame_name = f"{video_id}_{i}.jpg" 
        frame_path = os.path.join(out_dir, frame_name)

        if ret:
            cv2.imwrite(frame_path, frame)

        out_paths.append(frame_path)

    cap.release()
    return out_paths

