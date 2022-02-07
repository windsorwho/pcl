import numpy as np


def normalize_features(features):
    norms = 1.0 / np.linalg.norm(features, axis=1)
    norms = np.tile(np.expand_dims(norms, 1), (1, features.shape[1]))
    features = np.multiply(features, norms).astype(np.float32)
    return features



