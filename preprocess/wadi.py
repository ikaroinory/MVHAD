import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from data_types import NodeConfig, NodeType
from utils import Logger


def __downsample(data_np: ndarray, labels_np: ndarray, sample_len: int) -> tuple[ndarray, ndarray]:
    sequence_len, num_nodes = data_np.shape

    new_len = (sequence_len // sample_len) * sample_len
    data_np = data_np[:new_len]
    labels_np = labels_np[:new_len]

    data_np = data_np.reshape(-1, sample_len, num_nodes)
    downsampled_data_np = np.median(data_np, axis=1)

    labels_np = labels_np.reshape(-1, sample_len)
    downsampled_labels_np = np.max(labels_np, axis=1).round()

    return downsampled_data_np, downsampled_labels_np


def __normalize(train_data_df: DataFrame, test_data_df: DataFrame = None, *, node_indices: dict[NodeType, list]) -> ndarray:
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


def __preprocess(data_path: str, processed_data_path: str, sample_len: int, train_df: DataFrame = None) -> DataFrame:
    mode: str = 'train' if train_df is None else 'test'

    _processed_data_path = Path(processed_data_path)
    _processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    Logger.init()

    # Load data
    Logger.info(f'Loading {data_path}...')
    data_df = pd.read_csv(data_path, skiprows=0 if mode == 'train' else 1, index_col=0)
    data_df.rename(columns=lambda x: re.sub(r'\s+', '', x), inplace=True)
    data_df.rename(columns=lambda x: 'Attack' if 'AttackLABLE' in x else x, inplace=True)
    data_df = data_df.drop(columns=['Date', 'Time'])
    data_df = data_df.drop(columns=['2_LS_001_AL', '2_LS_002_AL', '2_P_001_STATUS', '2_P_002_STATUS'])
    Logger.info(f'Loaded.')

    # Replace 'Normal' and 'Attack' with 0 and 1
    Logger.info(f'Replacing Normal and Attack with 0 and 1...')
    if mode == 'train':
        data_df['Attack'] = 0
    data_df['Attack'] = data_df['Attack'].astype(str).str.replace(r'\s+', '', regex=True).map({'1': 0, '-1': 1})
    Logger.info(f'Replaced.')

    # Fill missing values
    Logger.info(f'Fill missing values...')
    data_df.fillna(data_df.mean(), inplace=True)
    data_df.fillna(0, inplace=True)
    Logger.info(f'Filled.')

    # Generate node name list
    Logger.info(f'Generating node indices...')
    node_names = [col for col in data_df.columns if col != 'Attack']
    actuator_names = [name for name in node_names if name.endswith('STATUS') or 'LS' in name or name == 'PLANT_START_STOP_LOG']
    sensor_names = [name for name in node_names if name not in actuator_names]

    node_config_path = _processed_data_path.parent / 'node_config.json'
    with open(node_config_path, 'w') as file:
        node_config: dict[NodeType, NodeConfig] = {
            'sensor': {
                'value_type': 'float',
                'index': [data_df.columns.get_loc(sensor) for sensor in sensor_names]
            },
            'actuator': {
                'value_type': 'enum',
                'index': [data_df.columns.get_loc(actuator) for actuator in actuator_names]
            }
        }
        file.write(json.dumps(node_config, indent=2))
    Logger.info(f'Save to {node_config_path} .')

    # Scale data using MinMaxScaler
    Logger.info(f'Scaling data...')
    data_labels = data_df['Attack']
    data_df.drop(columns=['Attack'], inplace=True)
    original_data_df = data_df.copy()
    node_indices = {'sensor': node_config['sensor']['index'], 'actuator': node_config['actuator']['index']}
    if mode == 'train':
        data_np = __normalize(data_df, node_indices=node_indices)
    else:
        data_np = __normalize(train_df, data_df, node_indices=node_indices)

    Logger.info(f'Scaled.')

    # Down-sample
    Logger.info('Down-sampling...')
    downsampled_data_np, downsampled_labels_np = __downsample(data_np, data_labels.to_numpy(), sample_len)
    data_df = pd.DataFrame(downsampled_data_np, columns=data_df.columns)
    data_df['Attack'] = downsampled_labels_np
    Logger.info('Down-sampled.')

    # Drop the first 2160 rows
    if mode == 'train':
        Logger.info(f'Dropping the first 2160 rows...')
        data_df = data_df.iloc[2160:]
        Logger.info(f'Dropped.')

    # Save data
    Logger.info('Saving data...')
    data_df.to_csv(processed_data_path, index=False)
    Logger.info(f'Saved to {processed_data_path} .')

    # Save edge types
    Logger.info('Saving edge types...')
    with open(_processed_data_path.parent / 'edge_types.json', 'w') as file:
        edge_types = [
            ['sensor', 'ss', 'sensor'],
            ['sensor', 'sa', 'actuator'],
            ['actuator', 'as', 'sensor'],
            ['actuator', 'aa', 'actuator']
        ]
        file.write(json.dumps(edge_types, indent=2))
    Logger.info('Saved edge types.')

    return original_data_df


def preprocess_wadi(original_data_path: tuple[str, str], processed_data_path: tuple[str, str], sample_len: int = 11):
    original_train_data_path, original_test_data_path = original_data_path
    processed_train_data_path, processed_test_data_path = processed_data_path

    original_train_data_df = __preprocess(original_train_data_path, processed_train_data_path, sample_len)
    __preprocess(original_test_data_path, processed_test_data_path, sample_len, original_train_data_df)
