
import json
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import os

# Load embedding dan label dari predictions.json
with open("results/predictions.json", "r") as f:
    data = json.load(f)

# Dummy vector sebagai placeholder (karena kita tidak simpan embedding sebelumnya)
# Untuk proyek nyata, ini sebaiknya dikumpulkan saat inference
np.random.seed(42)
image_embs = []
labels = []
label_to_idx = {}

for item in data:
    label = item["true_label"]
    if label not in label_to_idx:
        label_to_idx[label] = len(label_to_idx)
    labels.append(label_to_idx[label])
    image_embs.append(np.random.normal(0, 1, size=(512,)))  # <- ganti dengan embedding asli jika tersedia

image_embs = np.array(image_embs)
labels = np.array(labels)

# t-SNE projection
print("Running t-SNE...")
tsne = TSNE(n_components=2, random_state=42, perplexity=15)
proj = tsne.fit_transform(image_embs)

# Plot
plt.figure(figsize=(8,6))
classes = list(label_to_idx.keys())
for idx, class_name in enumerate(classes):
    class_proj = proj[labels == idx]
    plt.scatter(class_proj[:, 0], class_proj[:, 1], label=class_name)

plt.legend()
plt.title("CLIP Image Embedding Space (t-SNE)")
os.makedirs("results", exist_ok=True)
plt.savefig("results/tsne_plot.png")
print("Saved to results/tsne_plot.png")
