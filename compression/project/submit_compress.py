import os
import glob
import zipfile
import numpy as np
import shutil
import pickle
import sys

import project.pq as pq
#import project.constants as constants

sys.modules['pq'] = pq

PQ_64_CODEC = './codec_64.pkl'
NONE_NULL_COLUMNS_FILE = './non_null_index.pkl'


def get_file_basename(path: str) -> str:
    return os.path.splitext(os.path.basename(path))[0]


def read_feature_file(path: str) -> np.ndarray:
    with open(path, 'rb') as f:
        fea = np.frombuffer(f.read(), dtype='<f4')
    return fea


def extract_zipfile(dir_input: str, dir_dest: str):
    files = zipfile.ZipFile(dir_input, "r")
    for file in files.namelist():
        if file.find("__MACOSX") >= 0 or file.startswith('.'): continue
        else:
            files.extract(file, dir_dest)
    files.close()
    return 1


def compress_feature(fea: np.ndarray, encoder: pq.PQ, non_null_index: np.array,
                     path: str):
    assert fea.ndim == 1 and fea.dtype == np.float32
    feature = np.expand_dims(fea[non_null_index], axis=0)

    code = encoder.encode(feature).astype(np.ubyte)
    with open(path, 'wb') as f:
        f.write(code.tobytes())
    return True


def compress_all(input_path: str, bytes_rate: int):
    'Load the pq encoder for the current byte length.'
    pq_codec = pickle.load(open(PQ_64_CODEC, 'rb'))
    non_null_index = pickle.load(open(NONE_NULL_COLUMNS_FILE, 'rb'))

    compressed_query_fea_dir = 'compressed_query_feature/{}'.format(bytes_rate)
    os.makedirs(compressed_query_fea_dir, exist_ok=True)

    query_fea_paths = glob.glob(os.path.join(input_path, '*.*'))
    for query_fea_path in query_fea_paths:
        query_basename = get_file_basename(query_fea_path)
        fea = read_feature_file(query_fea_path)
        compressed_fea_path = os.path.join(compressed_query_fea_dir,
                                           query_basename + '.dat')
        compress_feature(fea, pq_codec, non_null_index, compressed_fea_path)

    print('Encode Done for bytes_rate' + str(bytes_rate))


def compress(test_path: str, byte: str):
    query_fea_dir = 'query_feature'
    extract_zipfile(test_path, query_fea_dir)
    compress_all(query_fea_dir, int(byte))
    shutil.rmtree(query_fea_dir)
    return 1
