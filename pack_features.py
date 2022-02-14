import numpy as np
import matplotlib.pyplot as plt
import tqdm
import logging
import sys
import os
import pickle
import glob
from sklearn.manifold import TSNE

import metric_learn

TRAIN_DATA_DIR = '/Users/wenzehu/data/pcl/train'
TEST_A_DIR = '/Users/wenzehu/data/pcl/test_A'
TEST_B_DIR = '/Users/wenzehu/data/pcl/test_B'
logFormatter = logging.Formatter(
    "%(asctime)s [%(threadName)-12.12s] [%(levelname)-5.5s]  %(message)s")
rootLogger = logging.getLogger()
consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(logFormatter)
rootLogger.addHandler(consoleHandler)
rootLogger.setLevel(logging.INFO)
"Read from dir and dump files in to a pickle file"


def read_features(label_filename, feature_path):
    lines = open(label_filename, 'r').readlines()
    features = []
    labels = []
    for line in tqdm.tqdm(lines):
        segs = line.strip().split(' ')
        label = int(segs[1])
        filename = os.path.join(feature_path, segs[0])
        feature = np.fromfile(filename, dtype=np.dtype('<f'))
        features.append(feature)
        labels.append(label)
    labels = np.array(labels)
    features = np.array(features)
    return labels, features


def pack_test_features(dataset_folder, query_folder_name, gallery_folder_name):
    gallery_files = glob.glob(
        os.path.join(dataset_folder, gallery_folder_name, '*.dat'))
    gallery_features = [
        np.fromfile(name, dtype=np.dtype('<f'))
        for name in tqdm.tqdm(gallery_files)
    ]
    query_files = glob.glob(
        os.path.join(dataset_folder, query_folder_name, '*.dat'))
    query_features = [
        np.fromfile(name, dtype=np.dtype('<f'))
        for name in tqdm.tqdm(query_files)
    ]
    gallery_features = np.array(gallery_features)
    query_features = np.array(query_features)
    pickle.dump(query_features,
                open(os.path.join(dataset_folder, 'query_features.pkl'), 'wb'))
    pickle.dump(
        gallery_features,
        open(os.path.join(dataset_folder, 'gallery_features.pkl'), 'wb'))
    query_files = [os.path.basename(name) for name in query_files]
    gallery_files = [os.path.basename(name) for name in gallery_files]
    pickle.dump(query_files,
                open(os.path.join(dataset_folder, 'query_names.pkl'), 'wb'))
    pickle.dump(gallery_files,
                open(os.path.join(dataset_folder, 'gallery_names.pkl'), 'wb'))
    return query_features, gallery_features, query_files, gallery_files


def pack_test_a_features():
    return pack_test_features(TEST_A_DIR, 'query_feature_A',
                              'gallery_feature_A')


def pack_test_b_features():
    return pack_test_features(TEST_B_DIR, 'query_feature_B',
                              'gallery_feature_B')


def pack_train_features():
    labels, features = read_features(
        os.path.join(TRAIN_DATA_DIR, 'train_list.txt'),
        os.path.join(TRAIN_DATA_DIR, 'train_feature'))
    pickle.dump(features,
                open(os.path.join(TRAIN_DATA_DIR, 'features.pkl'), 'wb'))
    pickle.dump(labels, open(os.path.join(TRAIN_DATA_DIR, 'labels.pkl'), 'wb'))


def read_packed_train_data():
    features = pickle.load(
        open(os.path.join(TRAIN_DATA_DIR, 'features.pkl'), 'rb'))
    labels = pickle.load(open(os.path.join(TRAIN_DATA_DIR, 'labels.pkl'),
                              'rb'))
    return features, labels


def read_packed_test_data(foldername):
    gallery_features = pickle.load(
        open(os.path.join(foldername, 'gallery_features.pkl'), 'rb'))
    query_features = pickle.load(
        open(os.path.join(foldername, 'query_features.pkl'), 'rb'))
    query_names = pickle.load(
        open(os.path.join(foldername, 'query_names.pkl'), 'rb'))
    gallery_names = pickle.load(
        open(os.path.join(foldername, 'gallery_names.pkl'), 'rb'))
    return query_features, gallery_features, query_names, gallery_names


def read_packed_test_a_data():
    return read_packed_test_data(TEST_A_DIR)


def remove_single_class(features, labels):
    " Input should be two np arrays of numbers"
    single_labels = np.where(np.bincount(labels) == 1)[0]
    single_idx = [np.where(labels == label)[0][0] for label in single_labels]
    logging.debug('Single exmpale indexes: {0}'.format(single_idx))
    features = np.delete(features, single_idx, axis=0)
    labels = np.delete(labels, single_idx)
    " make sure no more single class label exist"
    if len(np.where(np.bincount(labels) == 1)[0]) != 0:
        logging.error("There are still single example classes.")
    return features, labels


def plot_in_plan(X, y, colormap=plt.cm.Paired):
    plt.figure(figsize=(8, 6))

    # clean the figure
    plt.clf()

    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=colormap)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def plot_tsne(X, y, colormap=plt.cm.Paired):
    plt.figure(figsize=(8, 6))

    # clean the figure
    plt.clf()

    tsne = TSNE()
    X_embedded = tsne.fit_transform(X)
    plt.scatter(X_embedded[:, 0], X_embedded[:, 1], c=y, cmap=colormap)
    plt.xticks(())
    plt.yticks(())
    plt.show()


def remove_null_columns(train, query, gallery):
    train_idx = np.var(train, axis=0).nonzero()[0]
    query_idx = np.var(query, axis=0).nonzero()[0]
    gallery_idx = np.var(gallery, axis=0).nonzero()[0]
    if (not np.array_equal(gallery_idx, query_idx)) or (not np.array_equal(
            train_idx, query_idx)):
        logging.error(
            'Inputs do not share same zero colums, will return original input.'
        )
        assert ValueError
        return train, query, gallery
    train = train[:, train_idx]
    query = query[:, train_idx]
    gallery = gallery[:, train_idx]
    pickle.dump(train_idx, open('non_null_index.pkl', 'wb'))
    return train, query, gallery


def dump_data_for_test_b():
    features, labels = read_packed_train_data()
    queries, galleries, query_names, gallery_names = read_packed_test_data(
        TEST_B_DIR)
    features, labels = remove_single_class(features, labels)
    features, queries, galleries = remove_null_columns(features, queries,
                                                       galleries)
    np.savez(open(os.path.join(TRAIN_DATA_DIR, 'dense_data_b.npz'), 'wb'),
             features=features,
             labels=labels,
             queries=queries,
             galleries=galleries,
             query_names=query_names,
             gallery_names=gallery_names)


def try_metric_learn():
    features, labels = read_packed_train_data()
    queries, galleries, query_names, gallery_names = read_packed_test_a_data()
    features, labels = remove_single_class(features, labels)
    features, queries, galleries = remove_null_columns(features, queries,
                                                       galleries)
    np.savez(open(os.path.join(TRAIN_DATA_DIR, 'dense_data.npz'), 'wb'),
             features=features,
             labels=labels,
             queries=queries,
             galleries=galleries,
             query_names=query_names,
             gallery_names=gallery_names)
    logging.info('Start doing NCA.')
    nca = metric_learn.NCA(verbose=True)
    nca.fit(features, labels)
    X_lmnn = nca.transform(features)
    logging.info('Finished embedding.')
    plot_tsne(X_lmnn, labels)
