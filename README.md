# AssistX - Zero-Shot Action Recognition using CLIP

## 📌 Overview
This project implements **zero-shot action recognition** using the [UCF101 dataset](https://www.crcv.ucf.edu/data/UCF101.php) and the **CLIP Vision-Language Model**. It leverages the CLIP model to classify video actions by matching visual frame embeddings to natural language prompts, without training any model from scratch.

---

## 🧠 Task Summary
- Use CLIP (ViT-B/32) for action classification on video clips from UCF101
- Perform zero-shot inference using prompts: `"a photo of a person doing {action}"`
- Evaluate Top-1 and Top-5 Accuracy
- Compare single-frame vs multi-frame inference
- Visualize embedding space using t-SNE

---

## 📁 Folder Structure
```
assistx-clip-zero-shot/
├── dataset/                  
├── frames/                  
├── results/                  
│   ├── predictions.json
│   ├── predictions_multiframe.json
│   └── tsne_plot.png
├── run_inference.py          
├── run_inference_multiframe.py 
├── tsne_visualize.py         
├── utils.py                  
├── classnames.txt            
├── requirements.txt          
├── README.md                 
└── clip_ucf101_report.pdf    
```

---

## 🛠️ Setup Instructions
### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset
- Download UCF101 videos: [UCF101](https://www.crcv.ucf.edu/data/UCF101.php)
- Create structure:
```
dataset/<ClassName>/<video>.avi
```
- Only a subset of 2 classes is required for this experiment (ApplyLipstick, Archery).

---

## 🚀 How to Run
### 1. Single-Frame Inference
```bash
python run_inference.py
```

### 2. Multi-Frame (Temporal) Inference
```bash
python run_inference_multiframe.py
```

### 3. Visualize Embeddings
```bash
python tsne_visualize.py
```
Output is saved to `results/tsne_plot.png`

---

## 📈 Sample Output
```
Top-1 Accuracy: 100.00%
Top-5 Accuracy: 100.00%
```

Both single-frame and multi-frame inference achieved perfect accuracy on the chosen classes.

---

## 📄 Report
A detailed report explaining methodology, results, analysis, and bonus tasks is available:
📎 [clip_ucf101_report.pdf](./clip_ucf101_report.pdf)

---

## 🏁 Submission Guide
- Create a GitHub repository (public or private)
- Push all files except raw videos
- Include this README and PDF report
- Share the GitHub repo link as your submission

**Alternative:** zip the entire folder (excluding dataset) and upload via Google Drive.

---

## ✅ Requirements
- Python 3.8+
- torch
- torchvision
- clip-by-openai
- opencv-python
- numpy
- scikit-learn
- matplotlib
- markdown2 (for PDF conversion, optional)
- weasyprint (for PDF conversion, optional)

---

## 👤 Author
Created for AssistX technical assessment, 2025.

---
MIT License
