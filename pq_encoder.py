import faiss
import os
import numpy as np
import json

DATA_FILENAME = '/data/pcl/train/dense_data.npz'
ALIGNED_FEATURE_DIM = 464


def normalize_features(features):
    norms = 1.0 / np.linalg.norm(features, axis=1)
    norms = np.tile(np.expand_dims(norms, 1), (1, features.shape[1]))
    features = np.multiply(features, norms).astype(np.float32)
    return features


def align_features(data_dict):
    feature_names = ['queries', 'galleries', 'features']
    new_order = np.random.permutation(ALIGNED_FEATURE_DIM)
    for name in feature_names:
        feature = data_dict[name]
        feature = normalize_features(feature)
        feature = np.pad(feature,
                         ((0, 0), (0, ALIGNED_FEATURE_DIM - feature.shape[1])),
                         constant_values=0)
        feature = ((feature.T)[new_order]).T.astype(np.float32)
        data_dict[name] = feature.copy(order='C')
    return data_dict


np_dict = np.load(DATA_FILENAME)
data_dict = dict()
for key in np_dict.keys():
    data_dict[key] = np_dict[key]
data_dict = align_features(data_dict)

queries = data_dict['queries']
galleries = data_dict['galleries']
gallery_names = data_dict['gallery_names']
query_names = data_dict['query_names']
features = data_dict['features']

n_dim = ALIGNED_FEATURE_DIM  # Input feature dimension.
n_list = 32  # Number of flat index.
m = 232  # number of segments for each feature
n_bits = 8  # Number of bits for each segment

quantizer = faiss.IndexFlatL2(n_dim)  # this remains the same
index = faiss.IndexIVFPQ(quantizer, n_dim, n_list, m, n_bits)
index.nprobe = n_list
index.train(features)
index.add(galleries)

distance, top_idx = index.search(queries, 100)
result_dict = dict()
for i, name in enumerate(query_names):
    key = os.path.basename(name)
    top_names = [os.path.basename(gallery_names[idx]) for idx in top_idx[i]]
    result_dict[key] = top_names
json.dump(result_dict, open('result_pq.json', 'w'))
