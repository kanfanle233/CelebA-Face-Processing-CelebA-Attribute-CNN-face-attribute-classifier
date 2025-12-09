# celeba_utils.py
"""
CelebA 数据加载 + 对齐工具函数
从 all.ipynb 的 1–3 步抽出来的公共代码
"""

import numpy as np
import pandas as pd
import cv2
from pathlib import Path

# ========== 1. 路径配置（相对 qimo 目录） ==========
# 本文件在 qimo/daima/ 下，所以 parents[1] 是 qimo/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "archive"

IMG_DIR   = DATA_ROOT / "img_align_celeba"
ATTR_PATH = DATA_ROOT / "list_attr_celeba.csv"
BBOX_PATH = DATA_ROOT / "list_bbox_celeba.csv"
LAND_PATH = DATA_ROOT / "list_landmarks_align_celeba.csv"
PART_PATH = DATA_ROOT / "list_eval_partition.csv"


# ========== 2. 读 CSV 并合成 full_df ==========

def load_celeba_full_df():
    """
    读取 CelebA 的四个 csv，并合并成一个 full_df：
    包含 image_id + 40 属性 + bbox + 关键点 + split
    """
    attr_df = pd.read_csv(ATTR_PATH)
    bbox_df = pd.read_csv(BBOX_PATH)
    land_df = pd.read_csv(LAND_PATH)
    part_df = pd.read_csv(PART_PATH)

    # list_eval_partition 里的 partition 列改名为 split
    if "partition" in part_df.columns:
        part_df = part_df.rename(columns={"partition": "split"})

    full_df = (
        attr_df
        .merge(bbox_df, on="image_id")
        .merge(land_df, on="image_id")
        .merge(part_df, on="image_id")
    )
    return full_df


# ========== 3. 读图函数（BGR） ==========

def load_image(row):
    """
    根据 full_df 的一行记录读取原始图像 (BGR)
    """
    img_path = IMG_DIR / row["image_id"]
    img = cv2.imread(str(img_path))
    if img is None:
        raise FileNotFoundError(f"图像读取失败: {img_path}")
    return img


# ========== 4. 人脸 bbox & 对齐（照着你 notebook 的代码搬） ==========

def compute_bbox_from_landmarks(row, img_shape, expand_ratio=0.3):
    """
    用 5 个关键点生成一个人脸 bbox，并适当扩边
    expand_ratio: 在关键点包围盒基础上再放大一点
    """
    h, w = img_shape[:2]

    xs = np.array([
        row["lefteye_x"],
        row["righteye_x"],
        row["nose_x"],
        row["leftmouth_x"],
        row["rightmouth_x"],
    ], dtype=np.float32)

    ys = np.array([
        row["lefteye_y"],
        row["righteye_y"],
        row["nose_y"],
        row["leftmouth_y"],
        row["rightmouth_y"],
    ], dtype=np.float32)

    x_min, x_max = xs.min(), xs.max()
    y_min, y_max = ys.min(), ys.max()

    cx = (x_min + x_max) / 2
    cy = (y_min + y_max) / 2
    bw = (x_max - x_min) * (1 + expand_ratio)
    bh = (y_max - y_min) * (1 + expand_ratio * 1.2)  # 竖直方向多放一点

    x1 = int(np.clip(cx - bw / 2, 0, w - 1))
    y1 = int(np.clip(cy - bh / 2, 0, h - 1))
    x2 = int(np.clip(cx + bw / 2, x1 + 1, w))
    y2 = int(np.clip(cy + bh / 2, y1 + 1, h))

    return x1, y1, x2, y2


def crop_face_auto(img, row, expand_ratio=0.3):
    """先根据关键点算 bbox，再裁剪人脸"""
    x1, y1, x2, y2 = compute_bbox_from_landmarks(row, img.shape, expand_ratio)
    face = img[y1:y2, x1:x2].copy()
    return face, (x1, y1, x2, y2)


def align_face_by_eyes(img, row, output_size=(128, 128), expand_ratio=0.3):
    """
    1. 用关键点自动生成 bbox 并裁剪
    2. 根据双眼连线旋转对齐
    3. resize 到统一尺寸
    """
    face, (x1, y1, x2, y2) = crop_face_auto(img, row, expand_ratio)
    fh, fw = face.shape[:2]

    # 裁剪后关键点坐标（相对 face）
    lx = row["lefteye_x"] - x1
    ly = row["lefteye_y"] - y1
    rx = row["righteye_x"] - x1
    ry = row["righteye_y"] - y1

    # 计算旋转角度
    dy = ry - ly
    dx = rx - lx
    angle = np.degrees(np.arctan2(dy, dx))
    center = ((lx + rx) / 2.0, (ly + ry) / 2.0)

    M = cv2.getRotationMatrix2D(center, -angle, 1.0)
    face_aligned = cv2.warpAffine(face, M, (fw, fh), flags=cv2.INTER_LINEAR)

    face_resized = cv2.resize(face_aligned, output_size, interpolation=cv2.INTER_LINEAR)
    return face_resized
