
import torch
import numpy as np 

class MlpModel(torch.nn.Module):
    """Multilayer Perceptron with last layer linear.
    Args:
        input_size (int): number of inputs
        hidden_sizes (list): can be empty list for none (linear model).
        output_size: linear layer at output, or if ``None``, the last hidden size will be the output size and will have nonlinearity applied
        nonlinearity: torch nonlinearity Module (not Functional).
    """

    def __init__(
            self,
            input_size,
            hidden_sizes,  # Can be empty list for none.
            output_size=None,  # if None, last layer has nonlinearity applied.
            nonlinearity=torch.nn.ReLU,  # Module, not Functional.
            init_level=5
            ):
        super().__init__()
        if isinstance(hidden_sizes, int):
            hidden_sizes = [hidden_sizes]
        hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
            zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
        sequence = list()
        for layer in hidden_layers:
            # torch.nn.init.kaiming_uniform_(layer.weight, nonlinearity='relu')
            torch.nn.init.orthogonal_(layer.weight)
            sequence.extend([layer, nonlinearity()])
        if output_size is not None:
            last_size = hidden_sizes[-1] if hidden_sizes else input_size
            lastlayer = torch.nn.Linear(last_size, output_size)
            torch.nn.init.uniform_(lastlayer.weight, -1e-6, 1e-6)
            torch.nn.init.uniform_(lastlayer.bias, -1e-6, 1e-6)                
            sequence.append(lastlayer)
        self.model = torch.nn.Sequential(*sequence)
        self._output_size = (hidden_sizes[-1] if output_size is None
            else output_size)

    def forward(self, input):
        """Compute the model on the input, assuming input shape [B,input_size]."""
        return self.model(input)

    @property
    def output_size(self):
        """Retuns the output size of the model."""
        return self._output_size
# import torch
# import numpy as np

# class MlpModel(torch.nn.Module):
#     """Multilayer Perceptron with last layer linear."""

#     def __init__(
#             self,
#             input_size,
#             hidden_sizes,  # Can be empty list for none.
#             output_size=None,  # if None, last layer has nonlinearity applied.
#             nonlinearity=torch.nn.ReLU,  # Module, not Functional.
#             ):
#         super().__init__()
#         if isinstance(hidden_sizes, int):
#             hidden_sizes = [hidden_sizes]
#         hidden_layers = [torch.nn.Linear(n_in, n_out) for n_in, n_out in
#             zip([input_size] + hidden_sizes[:-1], hidden_sizes)]
#         sequence = list()
#         for layer in hidden_layers:
#             torch.nn.init.kaiming_uniform_(layer.weight,a=np.sqrt(5), nonlinearity='relu')
#             sequence.extend([layer, nonlinearity()])
#         if output_size is not None:
#             last_size = hidden_sizes[-1] if hidden_sizes else input_size
#             sequence.append(torch.nn.Linear(last_size, output_size))
#             torch.nn.init.uniform_(sequence[-1].weight, -3e-4, 3e-4)

#         self.model = torch.nn.Sequential(*sequence)
#         self._output_size = (hidden_sizes[-1] if output_size is None
#             else output_size)

#     def forward(self, input):
#         return self.model(input)

#     @property
#     def output_size(self):
#         return self._output_size
