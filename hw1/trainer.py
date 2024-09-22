import os
import torch
import torch.nn.functional as F
import torchvision
from torch.utils.data.dataloader import DataLoader
import numpy as np
import logging
logger = logging.getLogger(__name__)
from torch import nn
from models.mlp_head import MLPHead
import copy
# loss fn
def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def loss_fn(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)

# exponential moving average
class EMA():
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

class Online_Encoder(nn.Module):
    def __init__(self, backbone, output_dim, hidden_dim, layer=-2):
        super().__init__()
        self.backbone   = backbone
        self.layer      = layer
        self.projector  = None
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.MLPHead    = MLPHead( 2048, self.output_dim, self.hidden_dim)
        self.hidden     = {}
        self.hook_registered = False

    def _hook(self, _, input, output):
        device = input[0].device
        self.hidden[device] = output.reshape(output.shape[0], -1)

    def _register_hook(self):
        # find layer
        children = [*self.backbone.children()]
        layer = children[self.layer]
        # Register the tensor for a backward hook
        handle = layer.register_forward_hook(self._hook)
        self.hook_registered = True

    def get_representation(self, x):
        self._register_hook()
        self.hidden.clear()
        # triggers the forward pass and also activates the hook,
        # capturing the output from the hooked layer.
        _ = self.backbone(x)
        # It retrieves the captured hidden representation stored in self.hidden
        # (from the hooked layer output) and clears self.hidden again.
        hidden = self.hidden[x.device]
        self.hidden.clear()
        return hidden

    def forward(self, x):
        representation = self.get_representation(x)
        projection = self.MLPHead(representation)
        return projection, representation

class BYOLTrainer(nn.Module):
    def __init__(   self, **input ):
        super().__init__()
        self.backbone = input['backbone']
        self.device = input['device']
        self.online_encoder     = Online_Encoder(   self.backbone,
                                                    input['output_dim'],
                                                    input['hidden_dim'],
                                                    layer=input['hidden_layer'])

        self.use_momentum       = input['use_momentum']
        self.target_encoder     = None
        self.target_ema_updater = EMA(input['moving_average_decay'])
        self.online_predictor   = MLPHead(  input['output_dim'],
                                            input['output_dim'],
                                            input['hidden_dim'])

        # send a mock image tensor to instantiate singleton parameters
        self.to(self.device)
        self.forward(torch.randn(2,
                                input['input_shape'][2],
                                input['input_shape'][0],
                                input['input_shape'][0],
                                device=self.device),
                    torch.randn(2,
                                input['input_shape'][2],
                                input['input_shape'][0],
                                input['input_shape'][0],
                                device=self.device))

    def _get_target_encoder(self):
        target_encoder = copy.deepcopy(self.online_encoder)
        set_requires_grad(target_encoder, False)
        return target_encoder

    def reset_moving_average(self):
        del self.target_encoder
        self.target_encoder = None

    def update_moving_average(self):
        assert self.use_momentum, 'you do not need to update the moving average, since you have turned off momentum for the target encoder'
        assert self.target_encoder is not None, 'target encoder has not been created yet'
        update_moving_average(self.target_ema_updater,
                              self.target_encoder, self.online_encoder)

    def forward(self, x, x2, return_embedding=False, return_projection=True):
        if return_embedding:
            return self.online_encoder(x, return_projection=return_projection)

        online_proj_one, _      = self.online_encoder(x)
        online_proj_two, _      = self.online_encoder(x2)

        online_pred_one         = self.online_predictor(online_proj_one)
        online_pred_two         = self.online_predictor(online_proj_two)

        with torch.no_grad():
            target_encoder      = self._get_target_encoder(
            ) if self.use_momentum else self.online_encoder
            target_proj_one, _  = target_encoder(x)
            target_proj_two, _  = target_encoder(x2)
            target_proj_one.detach_()
            target_proj_two.detach_()

        loss_one    = loss_fn(online_pred_one, target_proj_two.detach())
        loss_two    = loss_fn(online_pred_two, target_proj_one.detach())

        loss        = loss_one + loss_two
        return loss.mean()