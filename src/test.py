# test.py
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from celeba_utils import load_celeba_full_df, load_image, align_face_by_eyes
from train import CelebAFaceDataset, SimpleCNN   # 直接复用 Dataset 和 模型结构


def evaluate(model, loader, device, criterion):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    num_classes = model.classifier[-1].out_features
    correct_per_attr = torch.zeros(num_classes, device=device)

    with torch.no_grad():
        for xb, yb in tqdm(loader, desc="Testing"):
            xb, yb = xb.to(device), yb.to(device)

            logits = model(xb)
            loss = criterion(logits, yb)

            total_loss += loss.item() * xb.size(0)
            total_samples += xb.size(0)

            probs = torch.sigmoid(logits)
            preds = (probs > 0.5).float()
            correct_per_attr += (preds == yb).sum(dim=0)

    avg_loss = total_loss / total_samples
    acc_per_attr = (correct_per_attr / total_samples).cpu().numpy()
    return avg_loss, acc_per_attr


def visualize_predictions(model, device, test_df, target_attrs, num_samples=12):
    """
    从 test_df 抽 num_samples 张图，可视化 GT & 预测
    """
    model.eval()

    tf = T.Compose([
        T.ToTensor(),
        T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    samples = test_df.sample(num_samples, random_state=42)

    rows = 3
    cols = int(np.ceil(num_samples / rows))

    plt.figure(figsize=(4 * cols, 4 * rows))

    with torch.no_grad():
        for i, (_, row) in enumerate(samples.iterrows()):
            img_raw = load_image(row)  # BGR
            face = align_face_by_eyes(img_raw, row, output_size=(128, 128))
            face_rgb = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            x = tf(Image.fromarray(face_rgb)).unsqueeze(0).to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()[0]

            # 真值（0/1）
            gt = row[target_attrs].values.astype(int)

            plt.subplot(rows, cols, i + 1)
            plt.imshow(face_rgb)
            plt.axis("off")
            title = (
                f"GT: M={gt[0]}, G={gt[1]}\n"
                f"P : M={probs[0]:.2f}, G={probs[1]:.2f}"
            )
            plt.title(title, fontsize=10)

    plt.tight_layout()
    plt.savefig("test_visualization.png", dpi=200)
    plt.show()
    print("可视化结果已保存到 test_visualization.png")


def main():
    # 设备选择
    assert torch.cuda.is_available(), "没有检测到 CUDA，请检查 GPU 环境"
    device = torch.device("cuda:0")
    print("当前 GPU:", torch.cuda.get_device_name(device))

    # 1. 读取 full_df，并准备 test_df（split == 2）
    full_df = load_celeba_full_df()
    print("CelebA 总样本数:", len(full_df))

    target_attrs = ["Male", "Eyeglasses"]

    for col in target_attrs:
        full_df[col] = full_df[col].replace({-1: 0, 1: 1})

    test_df = full_df[full_df["split"] == 2].reset_index(drop=True)
    print("Test size:", len(test_df))

    # 2. 构建 Dataset / DataLoader
    IMAGE_SIZE = (128, 128)
    BATCH_SIZE = 512
    NUM_WORKERS = 0  # Windows 稳定起见

    test_dataset = CelebAFaceDataset(test_df, target_attrs, IMAGE_SIZE, augment=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE,
                             shuffle=False, num_workers=NUM_WORKERS, pin_memory=True)

    # 3. 构建模型并加载最优权重
    num_classes = len(target_attrs)
    model = SimpleCNN(num_classes).to(device)

    state_dict = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(state_dict)
    print("已加载 best_model.pth")

    criterion = torch.nn.BCEWithLogitsLoss()

    # 4. 在 test 集上评估
    test_loss, test_acc_attr = evaluate(model, test_loader, device, criterion)

    print("\n=== Test Result ===")
    print(f"Test Loss: {test_loss:.4f}")
    for name, acc in zip(target_attrs, test_acc_attr):
        print(f"{name} accuracy: {acc:.4f}")
    print(f"Mean accuracy: {float(test_acc_attr.mean()):.4f}")

    # 5. 抽样可视化若干张图片
    visualize_predictions(model, device, test_df, target_attrs, num_samples=12)


if __name__ == "__main__":
    import torch.multiprocessing as mp
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

    main()
