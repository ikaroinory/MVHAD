import numpy as np
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler


def downsample(data_np: ndarray, labels_np: ndarray, sample_len: int) -> tuple[ndarray, ndarray]:
    sequence_len, num_nodes = data_np.shape

    new_len = (sequence_len // sample_len) * sample_len
    data_np = data_np[:new_len]
    labels_np = labels_np[:new_len]

    data_np = data_np.reshape(-1, sample_len, num_nodes)
    downsampled_data_np = np.median(data_np, axis=1)

    labels_np = labels_np.reshape(-1, sample_len)
    downsampled_labels_np = np.max(labels_np, axis=1).round()

    return downsampled_data_np, downsampled_labels_np


def normalize(train_data_df: DataFrame, test_data_df: DataFrame = None, *, node_indices: dict[NodeType, list[int]]) -> ndarray:
    train_data_np = train_data_df.to_numpy()

    normalizer = MinMaxScaler()
    normalizer.fit(train_data_np[:, node_indices['sensor']])

    if test_data_df is None:
        sensor_train_data_np = normalizer.transform(train_data_np[:, node_indices['sensor']])

        train_data_np[:, node_indices['sensor']] = sensor_train_data_np

        return train_data_np
    else:
        test_data_np = test_data_df.to_numpy()

        sensor_test_data_np = normalizer.transform(test_data_np[:, node_indices['sensor']])

        test_data_np[:, node_indices['sensor']] = sensor_test_data_np

        return test_data_np
