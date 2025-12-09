# visible.py
import os
import random
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from PIL import Image
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['Microsoft YaHei']
matplotlib.rcParams['axes.unicode_minus'] = False



import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T

from sklearn.metrics import confusion_matrix, classification_report

# Import your project code
from celeba_utils import load_celeba_full_df, load_image, align_face_by_eyes
from train import CelebAFaceDataset, SimpleCNN


# ============================================================
# 0. Device
# ============================================================
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Using device:", device)


# ============================================================
# 1. Load dataset
# ============================================================
full_df = load_celeba_full_df()
target_attrs = ["Male", "Eyeglasses"]

for col in target_attrs:
    full_df[col] = full_df[col].replace({-1: 0, 1: 1})

test_df = full_df[full_df["split"] == 2].reset_index(drop=True)
print("Test size:", len(test_df))

IMAGE_SIZE = (128, 128)
test_dataset = CelebAFaceDataset(test_df, target_attrs, IMAGE_SIZE, augment=False)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)


# ============================================================
# 2. Load model
# ============================================================
model = SimpleCNN(num_classes=2).to(device)
state_dict = torch.load("best_model.pth", map_location=device)
model.load_state_dict(state_dict)
model.eval()
print("Loaded best_model.pth")


# ============================================================
# 3. Run inference
# ============================================================
all_labels = []
all_preds = []
all_probs = []

with torch.no_grad():
    for xb, yb in test_loader:
        xb = xb.to(device)

        logits = model(xb)
        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs > 0.5).astype(int)
        yb = yb.cpu().numpy()

        all_labels.append(yb)
        all_preds.append(preds)
        all_probs.append(probs)

all_labels = np.vstack(all_labels)
all_preds = np.vstack(all_preds)
all_probs = np.vstack(all_probs)

print("Inference complete.")


# ============================================================
# 4. Classification report
# ============================================================
print("\n================ Classification Report ================")
for i, attr in enumerate(target_attrs):
    print(f"\n=== {attr} ===")
    print(classification_report(all_labels[:, i], all_preds[:, i], digits=4))


# ============================================================
# 5. Confusion matrix (per attribute)
# ============================================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for i, attr in enumerate(target_attrs):
    cm = confusion_matrix(all_labels[:, i], all_preds[:, i])
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["0","1"], yticklabels=["0","1"], ax=axes[i])
    axes[i].set_title(f"{attr} Confusion Matrix")
    axes[i].set_xlabel("Predicted")
    axes[i].set_ylabel("True")

plt.tight_layout()
plt.savefig("confusion_matrix.png", dpi=200)
plt.show()
print("Confusion matrix saved as confusion_matrix.png")


# ============================================================
# 6. Visualize correct predictions
# ============================================================
correct_idx = np.where((all_labels == all_preds).all(axis=1))[0]
sample_correct = random.sample(correct_idx.tolist(), min(12, len(correct_idx)))

tf = T.Compose([
    T.ToTensor(),
    T.Normalize([0.5]*3, [0.5]*3)
])

plt.figure(figsize=(12, 6))
for i, idx in enumerate(sample_correct):
    row = test_df.iloc[idx]
    img_raw = load_image(row)
    face = align_face_by_eyes(img_raw, row, output_size=(128,128))
    rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    gt = all_labels[idx]
    pred = all_preds[idx]

    plt.subplot(3, 4, i+1)
    plt.imshow(rgb)
    plt.axis("off")
    plt.title(f"✓ Correct\nGT: {gt}\nPred: {pred}", fontsize=9)

plt.tight_layout()
plt.savefig("correct_samples.png", dpi=200)
plt.show()
print("Correct samples saved as correct_samples.png")


# ============================================================
# 7. Visualize wrong predictions
# ============================================================
wrong_idx = np.where((all_labels != all_preds).any(axis=1))[0]

if len(wrong_idx) == 0:
    print("🎉 测试集中没有错误预测！模型极其优秀！")
else:
    sample_wrong = random.sample(wrong_idx.tolist(), min(12, len(wrong_idx)))

    plt.figure(figsize=(12, 6))
    for i, idx in enumerate(sample_wrong):
        row = test_df.iloc[idx]
        img_raw = load_image(row)
        face = align_face_by_eyes(img_raw, row, output_size=(128,128))
        rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

        gt = all_labels[idx]
        pred = all_preds[idx]

        plt.subplot(3, 4, i+1)
        plt.imshow(rgb)
        plt.axis("off")
        plt.title(f"❌ Wrong\nGT: {gt}\nPred: {pred}", fontsize=9)

    plt.tight_layout()
    plt.savefig("wrong_samples.png", dpi=200)
    plt.show()
    print("Wrong samples saved as wrong_samples.png")
