import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
import random

def plot_image(image, title=None, figsize=(10, 10)):
    """画像を表示する"""
    plt.figure(figsize=figsize)
    plt.imshow(image)
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()

def plot_batch(images, titles=None, figsize=(15, 5)):
    """バッチの画像を表示する"""
    n = len(images)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    
    for i, (ax, img) in enumerate(zip(axes, images)):
        ax.imshow(img)
        if titles and i < len(titles):
            ax.set_title(titles[i])
        ax.axis('off')
    plt.tight_layout()
    plt.show()

def plot_bbox(image, boxes, labels, scores=None, class_names=None, figsize=(10, 10)):
    """バウンディングボックスを描画する"""
    image = image.copy()
    colors = [(0, 255, 0), (255, 0, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]
    
    for i, (box, label) in enumerate(zip(boxes, labels)):
        color = colors[label % len(colors)]
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        label_text = f"{class_names[label]}" if class_names else f"Class {label}"
        if scores is not None:
            label_text += f": {scores[i]:.2f}"
        
        cv2.putText(image, label_text, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.axis('off')
    plt.show()

def plot_heatmap(image, heatmap, alpha=0.6, figsize=(10, 10)):
    """ヒートマップを重ねて表示する"""
    plt.figure(figsize=figsize)
    plt.imshow(image)
    plt.imshow(heatmap, alpha=alpha, cmap='jet')
    plt.axis('off')
    plt.show()

def plot_augmentations(image, augmentations, figsize=(15, 5)):
    """データ拡張の結果を表示する"""
    n = len(augmentations)
    fig, axes = plt.subplots(1, n+1, figsize=figsize)
    
    # 元画像
    axes[0].imshow(image)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    # 拡張画像
    for i, aug in enumerate(augmentations, 1):
        aug_img = aug(image=image)['image']
        axes[i].imshow(aug_img)
        axes[i].set_title(f'Augmentation {i}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()

def plot_class_distribution(labels, class_names=None, figsize=(10, 5)):
    """クラス分布を表示する"""
    plt.figure(figsize=figsize)
    sns.countplot(x=labels)
    if class_names:
        plt.xticks(range(len(class_names)), class_names, rotation=45)
    plt.title('Class Distribution')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.tight_layout()
    plt.show()

def plot_bbox_size_distribution(boxes, figsize=(10, 5)):
    """バウンディングボックスのサイズ分布を表示する"""
    widths = boxes[:, 2] - boxes[:, 0]
    heights = boxes[:, 3] - boxes[:, 1]
    areas = widths * heights
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    sns.histplot(widths, ax=axes[0])
    axes[0].set_title('Width Distribution')
    
    sns.histplot(heights, ax=axes[1])
    axes[1].set_title('Height Distribution')
    
    sns.histplot(areas, ax=axes[2])
    axes[2].set_title('Area Distribution')
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, class_names=None, figsize=(10, 8)):
    """混同行列を表示する"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names if class_names else 'auto',
                yticklabels=class_names if class_names else 'auto')
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.show()

def plot_learning_curve(train_losses, val_losses, figsize=(10, 5)):
    """学習曲線を表示する"""
    plt.figure(figsize=figsize)
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_pr_curve(precision, recall, figsize=(10, 5)):
    """PR曲線を表示する"""
    plt.figure(figsize=figsize)
    plt.plot(recall, precision)
    plt.title('Precision-Recall Curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.grid(True)
    plt.show()

def plot_roc_curve(fpr, tpr, figsize=(10, 5)):
    """ROC曲線を表示する"""
    plt.figure(figsize=figsize)
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.title('ROC Curve')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.grid(True)
    plt.show()

def create_augmentation_pipeline():
    """データ拡張のパイプラインを作成する"""
    return A.Compose([
        A.RandomRotate90(p=0.5),
        A.Flip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.1, rotate_limit=45, p=0.5),
        A.OneOf([
            A.ElasticTransform(alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03, p=0.5),
            A.GridDistortion(p=0.5),
            A.OpticalDistortion(distort_limit=1, shift_limit=0.5, p=0.5),
        ], p=0.3),
        A.OneOf([
            A.GaussNoise(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.RandomGamma(p=0.5),
        ], p=0.3),
    ]) 