import os
import glob
import numpy as np
import pickle
import sys

import project.pq as pq
#import project.constants as constants
PQ_64_CODEC = './codec_64.pkl'
NONE_NULL_COLUMNS_FILE = './non_null_index.pkl'
sys.modules['pq'] = pq


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def decompress_feature(path: str, codec: pq.PQ, non_null_index) -> np.ndarray:
    with open(path, 'rb') as f:
        code = np.frombuffer(f.read(), dtype=np.ubyte)
        feature = codec.decode(np.expand_dims(code, axis=0))
        full_feature = np.zeros(2048, dtype=np.float32)
        full_feature[non_null_index] = feature
    return full_feature


def write_feature_file(fea: np.ndarray, path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    with open(path, 'wb') as f:
        f.write(fea.astype('<f4').tostring())
    return True


def reconstruct(byte_rate: str):
    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(byte_rate)
    reconstructed_query_fea_dir = 'reconstructed_query_feature/{}'.format(
        byte_rate)
    os.makedirs(reconstructed_query_fea_dir, exist_ok=True)
    pq_codec = pickle.load(open(PQ_64_CODEC, 'rb'))
    non_null_index = pickle.load(open(NONE_NULL_COLUMNS_FILE, 'rb'))

    compressed_query_fea_paths = glob.glob(
        os.path.join(compressed_query_fea_dir, '*.*'))
    for compressed_query_fea_path in compressed_query_fea_paths:
        query_basename = get_file_basename(compressed_query_fea_path)
        reconstructed_fea = decompress_feature(compressed_query_fea_path,
                                               pq_codec, non_null_index)
        reconstructed_fea_path = os.path.join(reconstructed_query_fea_dir,
                                              query_basename + '.dat')
        write_feature_file(reconstructed_fea, reconstructed_fea_path)

    print('Decode Done' + byte_rate)
