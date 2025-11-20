import copy
import json
import random
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import Tensor
from torch.autograd import Variable
from torch.nn import CrossEntropyLoss, L1Loss
from torch.optim import Adam
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from data_types import EdgeType, NodeInformation
from datasets import MVHADDataset
from models import MVHAD
from .Arguments import Arguments
from .Logger import Logger
from .MGDA import MinNormSolver
from .evaluate import get_metrics


class Runner:
    def __init__(self):
        self.args = Arguments()

        self.start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.__log_path = Path(f'logs/{self.args.dataset}/{self.start_time}.log')
        self.__model_path = Path(f'saves/{self.args.dataset}/{self.start_time}.pth')

        Logger.init(self.__log_path if self.args.log else None)

        Logger.info('Setting seed...')
        self.__set_seed()

        Logger.info('Loading data...')
        train_dataloader, valid_dataloader, test_dataloader = self.__get_dataloaders()

        self.__train_dataloader: DataLoader = train_dataloader
        self.__valid_dataloader: DataLoader = valid_dataloader
        self.__test_dataloader: DataLoader = test_dataloader

        with open(f'data/processed/{self.args.dataset}/node_config.json', 'r') as f:
            self.node_config: dict[str, NodeInformation] = json.load(f)
        with open(f'data/processed/{self.args.dataset}/edge_types.json', 'r') as f:
            edge_types = json.load(f)
            edge_types: list[EdgeType] = [tuple(edge_type) for edge_type in edge_types]

        Logger.info('Building model...')
        self.__model = MVHAD(
            sequence_len=self.args.slide_window,
            d_hidden=self.args.d_hidden,
            num_heads=self.args.num_heads,
            num_output_layer=self.args.num_output_layer,
            k_dict=self.args.k_dict,
            node_config=self.node_config,
            edge_types=edge_types,
            dtype=self.args.dtype,
            device=self.args.device
        )
        self.__parameters_dict = defaultdict(list)
        for name, parameters in self.__model.named_parameters():
            if 'sensor' in name and 'output' in name:
                self.__parameters_dict['sensor'].append(parameters)
            elif 'actuator' in name and 'output' in name:
                self.__parameters_dict['actuator'].append(parameters)
            else:
                self.__parameters_dict['share'].append(parameters)

        self.__sensor_loss = L1Loss()
        self.__actuator_loss = CrossEntropyLoss()

        self.__sensor_optimizer = Adam(self.__parameters_dict['sensor'], lr=self.args.sensor_lr)
        self.__actuator_optimizer = Adam(self.__parameters_dict['actuator'], lr=self.args.actuator_lr)
        self.__share_optimizer = Adam(self.__parameters_dict['share'], lr=self.args.share_lr)

    def __set_seed(self) -> None:
        random.seed(self.args.seed)

        np.random.seed(self.args.seed)

        torch.manual_seed(self.args.seed)
        torch.cuda.manual_seed(self.args.seed)
        torch.cuda.manual_seed_all(self.args.seed)

        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    def __get_train_and_valid_dataloader(self, train_dataset: MVHADDataset, valid_size: float) -> tuple[DataLoader, DataLoader]:
        dataset_size = int(len(train_dataset))
        train_dataset_size = int((1 - valid_size) * dataset_size)
        valid_dataset_size = int(valid_size * dataset_size)

        valid_start_index = random.randrange(train_dataset_size)

        indices = torch.arange(dataset_size)
        train_indices = torch.cat([indices[:valid_start_index], indices[valid_start_index + valid_dataset_size:]])
        valid_indices = indices[valid_start_index:valid_start_index + valid_dataset_size]

        train_subset = Subset(train_dataset, train_indices)
        valid_subset = Subset(train_dataset, valid_indices)

        train_dataloader = DataLoader(train_subset, batch_size=self.args.batch_size, shuffle=True, worker_init_fn=lambda _: self.__set_seed())

        valid_dataloader = DataLoader(valid_subset, batch_size=self.args.batch_size, shuffle=False, worker_init_fn=lambda _: self.__set_seed())

        return train_dataloader, valid_dataloader

    def __get_dataloaders(self) -> tuple[DataLoader, DataLoader, DataLoader]:
        train_df = pd.read_csv(f'data/processed/{self.args.dataset}/train.csv')
        train_np = train_df.to_numpy()
        test_df = pd.read_csv(f'data/processed/{self.args.dataset}/test.csv')
        test_np = test_df.to_numpy()

        train_dataset = MVHADDataset(
            train_np,
            self.args.slide_window,
            self.args.slide_stride,
            mode='train',
            dtype=self.args.dtype
        )
        test_dataset = MVHADDataset(
            test_np,
            self.args.slide_window,
            self.args.slide_stride,
            mode='test',
            dtype=self.args.dtype
        )

        train_dataloader, valid_dataloader = self.__get_train_and_valid_dataloader(train_dataset, 0.1)

        test_dataloader = DataLoader(test_dataset, batch_size=self.args.batch_size, shuffle=False, worker_init_fn=lambda _: self.__set_seed())

        return train_dataloader, valid_dataloader, test_dataloader

    def __train_epoch(self) -> float:
        self.__model.train()

        total_train_loss = 0
        for x, y, _ in tqdm(self.__train_dataloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)

            grads: dict[str, list] = {'sensor': [], 'actuator': []}

            self.__sensor_optimizer.zero_grad()
            self.__actuator_optimizer.zero_grad()
            self.__share_optimizer.zero_grad()

            sensor_output, actuator_output = self.__model(x)

            sensor_loss = self.__sensor_loss(sensor_output, y[:, self.node_config['sensor']['index']])
            sensor_loss.backward(retain_graph=True)
            for name, parameters in self.__model.named_parameters():
                if parameters.grad is not None and 'sensor' in name and 'output' in name:
                    grads['sensor'].append(Variable(parameters.grad.data.clone(), requires_grad=True))

            actuator_output = actuator_output.reshape(-1, actuator_output.shape[-1])
            actuator_loss = self.__actuator_loss(actuator_output, y[:, self.node_config['actuator']['index']].long().reshape(-1))
            actuator_loss.backward(retain_graph=True)
            for name, parameters in self.__model.named_parameters():
                if parameters.grad is not None and 'actuator' in name and 'output' in name:
                    grads['actuator'].append(Variable(parameters.grad.data.clone(), requires_grad=True))

            alpha = MinNormSolver.find_min_norm_element(list(grads.values()), sensor_loss.data, actuator_loss.data)

            loss = alpha[0] * sensor_loss + alpha[1] * actuator_loss
            loss.backward()

            self.__sensor_optimizer.step()
            self.__actuator_optimizer.step()
            self.__share_optimizer.step()

            total_train_loss += loss.item() * x.shape[0]

        return total_train_loss / len(self.__train_dataloader.dataset)

    def __valid_epoch(self, dataloader: DataLoader) -> tuple[Tensor, Tensor, Tensor]:
        self.__model.eval()

        predicted_list = []
        actual_list = []
        label_list = []

        total_valid_loss = 0
        for x, y, label in tqdm(dataloader):
            x = x.to(self.args.device)
            y = y.to(self.args.device)
            label = label.to(self.args.device)

            with torch.no_grad():
                sensor_output, actuator_output = self.__model(x)

                output = torch.zeros([x.shape[0], x.shape[1]], dtype=self.args.dtype, device=self.args.device)
                output[:, self.__model.node_indices['sensor']] = sensor_output
                output[:, self.__model.node_indices['actuator']] = actuator_output.argmax(dim=-1).to(self.args.dtype)

                loss = self.__sensor_loss(output, y)

                total_valid_loss += loss.item() * x.shape[0]

                predicted_list.append(output)
                actual_list.append(y)
                label_list.append(label)

        predicted_tensor = torch.cat(predicted_list, dim=0)
        actual_tensor = torch.cat(actual_list, dim=0)
        label_tensor = torch.cat(label_list, dim=0)

        return predicted_tensor, actual_tensor, label_tensor

    def __train(self) -> None:
        Logger.info('Training...')

        best_epoch = -1
        best_valid_loss = float('inf')
        best_model_weights = copy.deepcopy(self.__model.state_dict())
        patience_counter = 0

        for epoch in tqdm(range(self.args.epochs)):
            train_loss = self.__train_epoch()

            Logger.info(f'Epoch {epoch + 1}:')
            Logger.info(f' - Train loss: {train_loss:.8f}')

            if train_loss < best_valid_loss:
                best_epoch = epoch + 1

                best_valid_loss = train_loss

                best_model_weights = copy.deepcopy(self.__model.state_dict())
                patience_counter = 0
            else:
                patience_counter += 1

            Logger.info(f' - Current best epoch: {best_epoch}')

            if patience_counter >= self.args.early_stop:
                break

        self.__model_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best_model_weights, self.__model_path)

        Logger.info(f'Best epoch: {best_epoch}')
        Logger.info(f' - Valid loss: {best_valid_loss:.8f}')
        Logger.info(f'Model save to {self.__model_path}')

    def __evaluate(self, model_name: Path) -> tuple[float, float, float, float, float, float | None]:
        Logger.info('Evaluating...')

        self.__model.load_state_dict(torch.load(f'{model_name}', weights_only=True, map_location=torch.device(self.args.device)))

        test_result = self.__valid_epoch(self.__test_dataloader)
        precision, recall, fpr, fnr, f1, pa_f1 = get_metrics(test_result, self.args.point_adjustment)

        Logger.info(f' - F1 score: {f1:.4f}')
        if self.args.point_adjustment:
            Logger.info(f' - PA-F1 score: {pa_f1:.4f}')
        Logger.info(f' - Precision: {precision:.4f}')
        Logger.info(f' - Recall: {recall:.4f}')
        Logger.info(f' - FPR: {fpr:.4f}')
        Logger.info(f' - FNR: {fnr:.4f}')

        return precision, recall, fpr, fnr, f1, pa_f1

    def run(self) -> tuple[float, float, float, float, float, float | None]:
        if self.args.model_path is None:
            self.__train()

        return self.__evaluate(self.__model_path if self.args.model_path is None else Path(self.args.model_path))
