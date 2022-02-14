import scipy
import scipy.spatial
import numpy as np
import os
import tqdm
import json
import sklearn.neighbors

DATA_DIR = '/users/wenzehu/data/pcl'
TEST_A_DIR = '/Users/wenzehu/data/pcl/test_A'


def top_k_dot_product(queries, galleries, k=100):

    dist_mat = 1 - np.matmul(queries, galleries.T)
    partition = np.argpartition(dist_mat, k, axis=1)
    order = partition[:, :k]
    for i in range(order.shape[0]):
        order[i] = order[i][np.argsort(dist_mat[i, order[i]])]


#    order_2 = np.argsort(dist_mat, axis=1)[:, :k]
#    if np.array_equal(order, order_2):
#        print('logic equal.')
    return order


def top_k(queries, galleries, k=100):
    dist_mat = scipy.spatial.distance.cdist(queries,
                                            galleries,
                                            metric='cosine')
    partition = np.argpartition(dist_mat, k, axis=1)
    order = partition[:, :k]
    for i in range(order.shape[0]):
        order[i] = order[i][np.argsort(dist_mat[i, order[i]])]


#    order_2 = np.argsort(dist_mat, axis=1)[:, :k]
#    if np.array_equal(order, order_2):
#        print('logic equal.')
    return order

data_dict = np.load(
    open(os.path.join(DATA_DIR, 'train', 'dense_data_b.npz'), 'rb'))
all_queries = data_dict['queries']
galleries = data_dict['galleries']
gallery_names = data_dict['gallery_names']
all_query_names = data_dict['query_names']
# Normalize the features.
query_norms = 1.0 / np.linalg.norm(all_queries, axis=1)
query_norms = np.tile(np.expand_dims(query_norms, 1), (1, 463))
all_queries = np.multiply(query_norms, all_queries)
gallery_norms = 1.0 / np.linalg.norm(galleries, axis=1)
gallery_norms = np.tile(np.expand_dims(gallery_norms, 1), (1, 463))
galleries = np.multiply(galleries, gallery_norms)

result_dict = dict()
batch_size = 100
index_engine = sklearn.neighbors.NearestNeighbors(n_neighbors=100,
                                                  algorithm='brute',
                                                  n_jobs=4)
index_engine.fit(galleries)
for i in tqdm.tqdm(range(0, len(all_queries), batch_size)):
    queries = all_queries[i:i + batch_size]
    query_names = all_query_names[i:i + batch_size]
    top_idx = index_engine.kneighbors(queries, return_distance=False)
    for i, name in enumerate(query_names):
        key = os.path.basename(name)
        top_names = [
            os.path.basename(gallery_names[idx]) for idx in top_idx[i]
        ]
        result_dict[key] = top_names
json.dump(result_dict, open('result_b_naive.json', 'w'))
