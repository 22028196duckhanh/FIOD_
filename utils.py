from typing import Tuple

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np


def overlapScore(rects1, rects2):
    avgScore = 0
    scores = []

    for i, _ in enumerate(rects1):
        rect1 = rects1[i]
        rect2 = rects2[i]

        left = np.max((rect1[0], rect2[0]))
        right = np.min((rect1[0] + rect1[2], rect2[0] + rect2[2]))

        top = np.max((rect1[1], rect2[1]))
        bottom = np.min((rect1[1] + rect1[3], rect2[1] + rect2[3]))

        # area of intersection
        i = np.max((0, right - left)) * np.max((0, bottom - top))

        # combined area of two rectangles
        u = rect1[2] * rect1[3] + rect2[2] * rect2[3] - i

        # return the overlap ratio
        # value is always between 0 and 1
        score = np.clip(i / u, 0, 1)
        avgScore += score
        scores.append(score)

    return avgScore, scores


def draw_box(ax, box: np.ndarray, color: str = 'r', label: str = None):
    rect = patches.Rectangle(
        (box[0], box[1]), box[2], box[3],
        linewidth=2, edgecolor=color, facecolor='none'
    )
    ax.add_patch(rect)
    if label:
        ax.text(box[0], box[1] - 2, label, color=color)


def visualize_batch(images: np.ndarray,
                    pred_boxes: np.ndarray,
                    gt_boxes: np.ndarray,
                    img_size: Tuple[int, int] = (100, 100),
                    num_samples: int = 4,
                    save_dir: str = None):
    num_samples = min(num_samples, len(images))
    fig, axes = plt.subplots(2, num_samples // 2, figsize=(15, 8))
    axes = axes.ravel()

    for i in range(num_samples):
        # Get image and convert to proper format
        img = images[i].squeeze()
        if len(img.shape) == 2:
            img = img.reshape(img_size)
        else:
            img = img.transpose(1, 2, 0)

        # Show image
        axes[i].imshow(img, cmap='gray')

        # Draw boxes
        draw_box(axes[i], pred_boxes[i], 'r', 'Pred')
        draw_box(axes[i], gt_boxes[i], 'g', 'GT')

        axes[i].set_title(f'Sample {i + 1}')
        axes[i].axis('off')

    plt.tight_layout()

    if save_dir:
        plt.savefig(f'{save_dir}/batch_visualization.png')
        plt.close()
    else:
        plt.show()

