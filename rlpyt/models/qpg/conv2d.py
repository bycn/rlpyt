
import numpy as np
import torch

from rlpyt.utils.tensor import infer_leading_dims, restore_leading_dims
from rlpyt.models.conv2d import Conv2dModel
from rlpyt.models.mlp import MlpModel

class MuConv2dModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            action_size,
            # output_size,
            fc_sizes=256,
            use_maxpool=False,
            channels=None, 
            kernel_sizes=None,
            strides=None,
            paddings=None,
            output_max=1,
            ):
        super().__init__()
        self._c, self._h, self._w = image_shape
        self._output_max = output_max
        self.conv = Conv2dModel(
            in_channels=self._c,
            channels=channels or [16, 32],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 0],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(self._h, self._w)
        self.mlp = MlpModel(
            input_size=conv_out_size, 
            hidden_sizes=fc_sizes, 
            output_size=action_size
        )

    def forward(self, observation, prev_action, prev_reward):
        #ideal
        # img = observation.observation.rgb_camera.type(torch.float)
        img = observation.observation.type(torch.float)
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        lead_dim, T, B, img_shape = infer_leading_dims(img, self._c)
        conv_out = self.conv(img.view(T * B, *img_shape))

        mu = self._output_max * torch.tanh(self.mlp(conv_out.view(T * B, -1)))
        mu = restore_leading_dims(mu, lead_dim, T, B)
        return mu


class QofMuConv2dModel(torch.nn.Module):

    def __init__(
            self,
            image_shape,
            action_size,
            output_max=1,
            # output_size,
            fc_sizes=256,
            use_maxpool=False,
            channels=None, 
            kernel_sizes=None,
            strides=None,
            paddings=None,
            ):
        super().__init__()
        self._c, self._h, self._w, = image_shape
        self._output_max = output_max
        self.conv = Conv2dModel(
            in_channels=self._c,
            channels=channels or [16, 32],
            kernel_sizes=kernel_sizes or [8, 4],
            strides=strides or [4, 2],
            paddings=paddings or [0, 0],
            use_maxpool=use_maxpool,
        )
        conv_out_size = self.conv.conv_out_size(self._h, self._w)
        self.mlp = MlpModel(
            input_size=(conv_out_size + action_size), 
            hidden_sizes=fc_sizes, 
            output_size=1
        )
    

    def forward(self, observation, prev_action, prev_reward, action):
        # ideal
        # img = observation.observation.rgb_camera.type(torch.float)
        img = observation.observation.type(torch.float)
        img = img.mul_(1. / 255)  # From [0-255] to [0-1], in place.
        lead_dim, T, B, img_shape = infer_leading_dims(img, self._c )

        conv_out = self.conv(img.view(T * B, *img_shape))
        q_input = torch.cat(
            [conv_out.view(T * B, -1), action.view(T * B, -1)], dim=1)
        q = self.mlp(q_input).squeeze(-1)
        q = restore_leading_dims(q, lead_dim, T, B)
        return q