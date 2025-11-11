import json
import re
from pathlib import Path

import pandas as pd
from numpy import ndarray
from pandas import DataFrame
from sklearn.preprocessing import MinMaxScaler

from data_types import NodeInformation, NodeType
from utils import Logger


def __normalize(train_data_df: DataFrame, test_data_df: DataFrame = None, *, node_indices: dict[NodeType, list[int]]) -> ndarray:
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


def __preprocess(data_path: str, processed_data_path: str, sample_len: int = 10, train_df: DataFrame = None) -> DataFrame:
    mode: str = 'train' if train_df is None else 'test'

    _processed_data_path = Path(processed_data_path)
    _processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    Logger.init()

    # Load data
    Logger.info(f'Loading {data_path}...')
    data_df = pd.read_csv(data_path, index_col=0)
    data_df.rename(columns=lambda x: re.sub(r'\s+', '', x), inplace=True)
    Logger.info(f'Loaded.')

    # Replace 'Normal' and 'Attack' with 0 and 1
    Logger.info(f'Replacing Normal and Attack with 0 and 1...')
    data_df['ATT_FLAG'] = data_df['ATT_FLAG'].astype(str).str.replace(r'\s+', '', regex=True).map({'0': 0, '-999': 0, '1': 1})
    Logger.info(f'Replaced.')

    # Fill missing values
    Logger.info(f'Fill missing values...')
    data_df.fillna(data_df.mean(), inplace=True)
    data_df.fillna(0, inplace=True)
    Logger.info(f'Filled.')

    # Generate node name list
    Logger.info(f'Generating node indices...')
    node_names = [col for col in data_df.columns if col != 'ATT_FLAG']
    actuator_names = [
        'S_PU1',
        'S_PU2',
        'F_PU3', 'S_PU3',
        'S_PU4',
        'F_PU5', 'S_PU5',
        'S_PU6',
        'S_PU7',
        'S_PU8',
        'F_PU9', 'S_PU9',
        'S_PU10',
        'S_PU11',
        'S_V2'
    ]
    sensor_names = [name for name in node_names if name not in actuator_names]

    node_config_path = _processed_data_path.parent / 'node_config.json'
    with open(node_config_path, 'w') as file:
        node_config: dict[NodeType, NodeInformation] = {
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
    data_labels = data_df['ATT_FLAG'].values
    data_df.drop(columns=['ATT_FLAG'], inplace=True)
    original_data_df = data_df.copy()
    node_indices: dict[NodeType, list[int]] = {'sensor': node_config['sensor']['index'], 'actuator': node_config['actuator']['index']}
    if mode == 'train':
        data_np = __normalize(data_df, node_indices=node_indices)
    else:
        data_np = __normalize(train_df, data_df, node_indices=node_indices)
    Logger.info(f'Scaled.')

    data_df = pd.DataFrame(data_np, columns=data_df.columns)
    data_df['Attack'] = data_labels

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


def preprocess_BATADAL(original_data_path: tuple[str, str], processed_data_path: tuple[str, str], sample_len: int = 11):
    original_train_data_path, original_test_data_path = original_data_path
    processed_train_data_path, processed_test_data_path = processed_data_path

    original_train_data_df = __preprocess(original_train_data_path, processed_train_data_path, sample_len)
    __preprocess(original_test_data_path, processed_test_data_path, sample_len, original_train_data_df)
