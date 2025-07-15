from torch import Tensor, nn


class OutputLayer(nn.Module):
    def __init__(self, d_input: int, d_hidden: int, d_output: int, num_layers: int):
        super().__init__()

        self.num_layers = num_layers

        self.mlp = nn.ModuleList()
        for i in range(num_layers - 1):
            self.mlp.append(nn.Linear(d_input if i == 0 else d_hidden, d_hidden))
            self.mlp.append(nn.BatchNorm1d(d_hidden))
            self.mlp.append(nn.ReLU())
        self.output_linear = nn.Linear(d_input if num_layers == 1 else d_hidden, d_output)

    def forward(self, x: Tensor) -> Tensor:
        batch_size, num_nodes, d_input = x.shape

        mlp_output = x.reshape(-1, d_input)

        for model in self.mlp:
            mlp_output = model(mlp_output)
        mlp_output = mlp_output.reshape(batch_size, num_nodes, -1)

        sensor_output = self.output_linear(mlp_output).squeeze()

        return sensor_output

    def __call__(self, x: Tensor) -> Tensor:
        return super().__call__(x)
