from typing import Generator, Iterable, List, TypeVar

import cv2
import numpy as np
import supervision as sv
import torch
import umap
from sklearn.cluster import KMeans
from tqdm import tqdm
from transformers import AutoProcessor, SiglipVisionModel

V = TypeVar("V")

SIGLIP_MODEL_PATH = 'google/siglip-base-patch16-224'


def create_batches(
    sequence: Iterable[V], batch_size: int
) -> Generator[List[V], None, None]:
    """
    Generate batches from a sequence with a specified batch size.

    Args:
        sequence (Iterable[V]): The input sequence to be batched.
        batch_size (int): The size of each batch.

    Yields:
        Generator[List[V], None, None]: A generator yielding batches of the input
            sequence.
    """
    batch_size = max(batch_size, 1)
    current_batch = []
    for element in sequence:
        if len(current_batch) == batch_size:
            yield current_batch
            current_batch = []
        current_batch.append(element)
    if current_batch:
        yield current_batch


class TeamClassifier:
    """
    A classifier that uses a pre-trained SiglipVisionModel for feature extraction,
    UMAP for dimensionality reduction, and KMeans for clustering.
    """
    def __init__(self, device: str = 'cpu', batch_size: int = 32):
        """
       Initialize the TeamClassifier with device and batch size.

       Args:
           device (str): The device to run the model on ('cpu' or 'cuda').
           batch_size (int): The batch size for processing images.
       """
        self.device = device
        self.batch_size = batch_size
        self.features_model = SiglipVisionModel.from_pretrained(
            SIGLIP_MODEL_PATH).to(device)
        self.processor = AutoProcessor.from_pretrained(SIGLIP_MODEL_PATH)
        self.reducer = umap.UMAP(n_components=3)
        self.cluster_model = KMeans(n_clusters=2)

    def extract_features(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Extract features from a list of image crops using the pre-trained
            SiglipVisionModel.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Extracted features as a numpy array.
        """
        crops = [sv.cv2_to_pillow(crop) for crop in crops]
        batches = create_batches(crops, self.batch_size)
        data = []
        with torch.no_grad():
            for batch in tqdm(batches, desc='Embedding extraction'):
                inputs = self.processor(
                    images=batch, return_tensors="pt").to(self.device)
                outputs = self.features_model(**inputs)
                embeddings = torch.mean(outputs.last_hidden_state, dim=1).cpu().numpy()
                data.append(embeddings)

        return np.concatenate(data)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit the classifier model on a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.
        """
        data = self.extract_features(crops)
        projections = self.reducer.fit_transform(data)
        self.cluster_model.fit(projections)

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict the cluster labels for a list of image crops.

        Args:
            crops (List[np.ndarray]): List of image crops.

        Returns:
            np.ndarray: Predicted cluster labels.
        """
        if len(crops) == 0:
            return np.array([])

        data = self.extract_features(crops)
        projections = self.reducer.transform(data)
        return self.cluster_model.predict(projections)


def _extract_cap_color_features(crops: List[np.ndarray]) -> np.ndarray:
    """
    Extract color features from cap crops for dark-vs-light clustering.
    Uses the center 50% of each crop (to reduce water/edges) and mean HSV
    saturation and value so KMeans(2) separates dark caps from light caps.

    Args:
        crops (List[np.ndarray]): List of BGR image crops (e.g. cap regions).

    Returns:
        np.ndarray: Shape (N, 2), each row is (mean_S, mean_V) in [0, 1].
    """
    features = []
    for crop in crops:
        if crop.size == 0:
            features.append([0.0, 0.0])
            continue
        h, w = crop.shape[:2]
        cx, cy = w // 2, h // 2
        h2, w2 = max(1, h // 2), max(1, w // 2)
        center = crop[
            max(0, cy - h2) : cy + h2,
            max(0, cx - w2) : cx + w2,
        ]
        hsv = cv2.cvtColor(center, cv2.COLOR_BGR2HSV)
        mean_s = np.float64(hsv[:, :, 1].mean()) / 255.0
        mean_v = np.float64(hsv[:, :, 2].mean()) / 255.0
        features.append([mean_s, mean_v])
    return np.array(features, dtype=np.float64)


class TeamClassifierByColor:
    """
    Team classifier that clusters cap crops by color (dark vs light) using
    mean HSV saturation and value in the center of each crop. Use this when
    team identity is determined by cap color (e.g. black vs white) and
    SigLIP-based clustering gives mixed results.
    """

    def __init__(self) -> None:
        self.cluster_model = KMeans(n_clusters=2, n_init=10, random_state=0)

    def fit(self, crops: List[np.ndarray]) -> None:
        """
        Fit KMeans on color features (mean S, V) from each crop.
        Cluster 0 will be the darker group, cluster 1 the lighter (by mean V).
        """
        data = _extract_cap_color_features(crops)
        self.cluster_model.fit(data)
        # Optional: ensure consistent labeling (cluster 0 = darker)
        centers = self.cluster_model.cluster_centers_
        if centers[1][1] < centers[0][1]:
            self._swap_labels = True
        else:
            self._swap_labels = False

    def predict(self, crops: List[np.ndarray]) -> np.ndarray:
        """
        Predict cluster (0 or 1) for each crop. Returns same shape as
        number of crops; 0 = darker caps, 1 = lighter caps (if fit used
        consistent labeling).
        """
        if len(crops) == 0:
            return np.array([])
        data = _extract_cap_color_features(crops)
        labels = self.cluster_model.predict(data)
        if getattr(self, "_swap_labels", False):
            labels = 1 - labels
        return labels
