import argparse

import torch

from data_types import EdgeType


class Arguments:
    def __init__(self):
        args = self.parse_args()

        self.seed: int = args.seed

        self.model_path: str | None = args.model

        self.dataset: str = args.dataset
        self.dtype = torch.float32 if args.dtype == 'float' else torch.float64
        self.device = args.device

        self.batch_size: int = args.batch_size
        self.epochs: int = args.epochs

        self.slide_window: int = args.slide_window
        self.slide_stride: int = args.slide_stride
        self.k_dict: dict[EdgeType, int] = {
            ('sensor', 'ss', 'sensor'): args.k_ss,
            ('sensor', 'sa', 'actuator'): args.k_sa,
            ('actuator', 'as', 'sensor'): args.k_as,
            ('actuator', 'aa', 'actuator'): args.k_aa
        }

        self.d_hidden: int = args.d_hidden
        self.d_output_hidden: int = args.d_output_hidden

        self.num_heads: int = args.num_heads
        self.num_output_layer: int = args.num_output_layer

        self.share_lr: float = args.share_lr
        self.sensor_lr: float = args.sensor_lr
        self.actuator_lr: float = args.actuator_lr

        self.early_stop: int = args.early_stop

        self.point_adjustment: bool = args.point_adjustment

        self.log: bool = not args.nolog

    @staticmethod
    def parse_args():
        parser = argparse.ArgumentParser()

        parser.add_argument('--seed', type=int, default=42)

        parser.add_argument('--model', type=str)

        parser.add_argument('-ds', '--dataset', type=str, default='swat')
        parser.add_argument('--dtype', choices=['float', 'double'], default='float')
        parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda')

        parser.add_argument('-b', '--batch_size', type=int, default=256)
        parser.add_argument('-e', '--epochs', type=int, default=50)

        parser.add_argument('-sw', '--slide_window', type=int, default=10)
        parser.add_argument('-ss', '--slide_stride', type=int, default=1)
        parser.add_argument('--k_ss', type=int, default=10)
        parser.add_argument('--k_sa', type=int, default=10)
        parser.add_argument('--k_as', type=int, default=10)
        parser.add_argument('--k_aa', type=int, default=10)

        parser.add_argument('--d_hidden', type=int, default=64)
        parser.add_argument('--d_output_hidden', type=int, default=205)

        parser.add_argument('--num_heads', type=int, default=8)
        parser.add_argument('--num_output_layer', type=int, default=1)

        parser.add_argument('--share_lr', type=float, default=1e-4)
        parser.add_argument('--sensor_lr', type=float, default=1e-4)
        parser.add_argument('--actuator_lr', type=float, default=1e-4)

        parser.add_argument('--early_stop', type=int, default=100)

        parser.add_argument('-pa', '--point_adjustment', action='store_true')

        parser.add_argument('--nolog', action='store_true')

        return parser.parse_args()
