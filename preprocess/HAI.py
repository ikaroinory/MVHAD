import json
import re
from pathlib import Path

import pandas as pd
from pandas import DataFrame

from data_types import NodeInformation, NodeType
from utils import Logger
from .core import downsample, normalize


def __preprocess(data_path: str, processed_data_path: str, sample_len: int = 10, train_df: DataFrame = None) -> DataFrame:
    mode: str = 'train' if train_df is None else 'test'

    _data_path = Path(data_path) / 'haiend-23.05' / ('end-train1.csv' if mode == 'train' else 'end-test1.csv')
    _processed_data_path = Path(processed_data_path) / ('train.csv' if mode == 'train' else 'test.csv')
    _processed_data_path.parent.mkdir(parents=True, exist_ok=True)

    Logger.init()

    # Load data
    Logger.info(f'Loading {_data_path}...')
    data_df = pd.read_csv(_data_path, index_col=0)
    data_df.rename(columns=lambda x: re.sub(r'\s+', '', x), inplace=True)
    if mode == 'train':
        data_df['Attack'] = 0
    else:
        data_df['Attack'] = pd.read_csv(_data_path.parent / 'label-test1.csv', index_col=0)['label']

    # Fill missing values
    Logger.info(f'Fill missing values...')
    data_df.fillna(data_df.mean(), inplace=True)
    data_df.fillna(0, inplace=True)

    # Generate node name list
    Logger.info(f'Generating node indices...')
    node_names = [col for col in data_df.columns if col != 'Attack']
    sensor_names = [
        'DM-FT01Z', 'DM-FT02Z', 'DM-FT03Z', '1001.5-OUT', '1001.13-OUT', '1001.14-OUT', '1001.15-OUT', '1001.16-OUT', '1001.17-OUT', '1001.20-OUT',
        '1002.7-OUT', '1002.8-OUT', '1002.9-OUT', '1002.20-OUT', '1002.21-OUT', '1002.30-OUT', '1002.31-OUT', '1003.5-OUT', '1003.10-OUT',
        '1003.11-OUT', '1003.17-OUT', '1003.18-OUT', '1003.23-OUT', '1003.24-OUT', '1003.25-OUT', '1003.26-OUT', '1003.29-OUT', '1003.30-OUT',
        '1020.13-OUT', '1020.14-OUT', '1020.15-OUT', 'DM-AIT-DO', 'DM-AIT-PH', 'DM-PP04-AO', 'DM-TWIT-04', 'DM-TWIT-05', 'GATEOPEN', 'PP04-SP-OUT',
        'DM-FCV01-D', 'DM-FCV01-Z', 'DM-FCV02-D', 'DM-FCV02-Z', 'DM-FCV03-D', 'DM-FCV03-Z', 'DM-FT01', 'DM-FT02', 'DM-FT03', 'DM-LCV01-D',
        'DM-LCV01-Z', 'DM-LIT01', 'DM-PCV01-D', 'DM-PCV01-Z', 'DM-PCV02-D', 'DM-PCV02-Z', 'DM-PIT01', 'DM-PIT02', 'DM-PWIT-03', 'DM-TIT01',
        'DM-TIT02', 'DM-TWIT-03'
    ]
    actuator_names = [name for name in node_names if name not in sensor_names]

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
    data_labels = data_df['Attack']
    data_df.drop(columns=['Attack'], inplace=True)
    original_data_df = data_df.copy()
    node_indices: dict[NodeType, list[int]] = {'sensor': node_config['sensor']['index'], 'actuator': node_config['actuator']['index']}
    if mode == 'train':
        data_np = normalize(data_df, node_indices=node_indices)
    else:
        data_np = normalize(train_df, data_df, node_indices=node_indices)

    Logger.info(f'Scaled.')
    data_df = pd.DataFrame(data_np, columns=data_df.columns)

    # Down-sample
    Logger.info('Down-sampling...')
    downsampled_data_np, downsampled_labels_np = downsample(data_np, data_labels.to_numpy(), sample_len)
    data_df = pd.DataFrame(downsampled_data_np, columns=data_df.columns)
    data_df['Attack'] = downsampled_labels_np
    Logger.info('Down-sampled.')

    data_df[actuator_names] = data_df[actuator_names].replace({0: 0, 1: 1, 19: 2, 30: 3, 40: 4})

    # Save data
    Logger.info('Saving data...')
    data_df.to_csv(_processed_data_path, index=False)
    Logger.info(f'Saved to {_processed_data_path} .')

    # Save edge types
    Logger.info('Saving edge types...')
    edge_types_path = _processed_data_path.parent / 'edge_types.json'
    with open(edge_types_path, 'w') as file:
        edge_types = [
            ['sensor', 'ss', 'sensor'],
            ['sensor', 'sa', 'actuator'],
            ['actuator', 'as', 'sensor'],
            ['actuator', 'aa', 'actuator']
        ]
        file.write(json.dumps(edge_types, indent=2))
    Logger.info(f'Saved to {edge_types_path} .')

    return original_data_df


def preprocess_HAI(original_data_path: str, processed_data_path: str, sample_len: int = 11) -> None:
    original_train_data_df = __preprocess(original_data_path, processed_data_path, sample_len)
    __preprocess(original_data_path, processed_data_path, sample_len, original_train_data_df)
