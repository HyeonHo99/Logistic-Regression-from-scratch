import os
import numpy as np
from optim.Optimizer import *


def load_class_data(path, filename, target_at_front, to_binary=False, normalize=False, exclude_label=None, exclude_feature=None, shuffle=False):
    if exclude_feature is None:
        exclude_feature = []
    if exclude_label is None:
        exclude_label = []

    fullpath = os.path.join(path, filename)

    with open(fullpath, 'r') as f:
        lines = f.readlines()
    lines = [s.strip().split(',') for s in lines]

    header = lines[0]
    raw_data = lines[1:]
    num_feat = len(raw_data[0])
    feat_to_idx = [{} for _ in range(num_feat)]
    data = []
    for d in raw_data:
        line = []

        for i, f in enumerate(d):
            if i in exclude_feature:
                continue
            try:
                line.append(float(f))
            except:
                if f in feat_to_idx[i]:
                    f_idx = feat_to_idx[i][f]
                else:
                    f_idx = len(feat_to_idx[i])
                    feat_to_idx[i][f] = f_idx
                line.append(f_idx)
        data.append(line)

    data = np.array(data, dtype=np.float32)
    if target_at_front:
        x, y = data[:, 1:], data[:, 0].astype(np.int32)
    else:
        x, y = data[:, :-1], data[:, -1].astype(np.int32)

    num_data = x.shape[0]
    if normalize:
        mins = np.expand_dims(np.min(x, axis=0), 0).repeat(num_data, 0)
        maxs = np.expand_dims(np.max(x, axis=0), 0).repeat(num_data, 0)
        x = (x - mins) / maxs

    # Add 1 column for bias
    bias = np.ones((x.shape[0], 1), dtype=np.float32)
    x = np.concatenate((bias, x), axis=1)

    if to_binary:
        y[y > 1] = 1

    if shuffle:
        perm = np.random.permutation(num_data)
        x = x[perm]
        y = y[perm]

    return x, y


def BreastCancerData(path, filename):
    x, y = load_class_data(path, filename, target_at_front=False, normalize=False, shuffle=True)
    return (x, y)


def EMNISTData(path):
    train_x = np.load(os.path.join(path, 'train_images_full.npy'))
    train_y = np.load(os.path.join(path, 'train_labels_full.npy'))
    test_x = np.load(os.path.join(path, 'test_images_full.npy'))
    test_y = np.load(os.path.join(path, 'test_labels_full.npy'))

    bias = np.ones((train_x.shape[0], 1), dtype=np.float32)
    train_x = np.concatenate((bias, train_x), axis=1)

    bias = np.ones((test_x.shape[0], 1), dtype=np.float32)
    test_x = np.concatenate((bias, test_x), axis=1)
    
    return train_x, train_y, test_x, test_y


def accuracy(h, y):
    """
    h : (N, ), predicted label
    y : (N, ), correct label
    """

    total = h.shape[0]
    correct = len(np.where(h==y)[0])

    acc = correct / total

    return acc


data_dir = {
    'Breast_cancer': 'breast_cancer',
    'EMNIST': 'EMNIST'
}


def load_data(data_name):
    dir_name = data_dir[data_name]
    path = os.path.join('./data', dir_name)

    if data_name == 'Breast_cancer':
        train_x, train_y = BreastCancerData(path, 'train.csv')
        test_x, test_y = BreastCancerData(path, 'test.csv')
    elif data_name == 'EMNIST':
        train_x, train_y, test_x, test_y = EMNISTData(path)
    else:
        raise NotImplementedError

    return (train_x, train_y), (test_x, test_y)
