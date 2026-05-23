"""Classical image segmentation using K-Means or Gaussian Mixture Models."""

import argparse

import cv2
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture


def parse_args():
    parser = argparse.ArgumentParser(description="Classical image segmentation")
    parser.add_argument("--image", default="road.png", help="Path to input image")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters")
    parser.add_argument(
        "--method",
        choices=["kmeans", "gmm"],
        default="gmm",
        help="Segmentation algorithm: kmeans or gmm",
    )
    return parser.parse_args()


def segment(image_path, n_clusters, method):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    pixels = img_rgb.reshape(-1, 3)

    if method == "kmeans":
        print("Running K-Means clustering...")
        model = KMeans(
            n_clusters=n_clusters,
            init="k-means++",
            max_iter=250,
            n_init=10,
            random_state=35,
        )
        labels = model.fit_predict(pixels)
    else:
        print("Running Gaussian Mixture Model...")
        model = GaussianMixture(n_components=n_clusters, covariance_type="tied")
        labels = model.fit_predict(pixels)

    mask = labels.reshape(img_rgb.shape[0], img_rgb.shape[1])

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(img_rgb)
    axes[0].set_title("Original")
    axes[0].axis("off")

    axes[1].imshow(mask, cmap="viridis")
    axes[1].set_title(f"{method.upper()} mask")
    axes[1].axis("off")

    foreground = img_rgb * np.expand_dims(mask == mask.max(), axis=-1)
    axes[2].imshow(foreground)
    axes[2].set_title("Foreground")
    axes[2].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    args = parse_args()
    segment(args.image, args.clusters, args.method)
