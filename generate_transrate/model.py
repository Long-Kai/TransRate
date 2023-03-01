import torch.nn as nn
from resnet18 import get_resnet, get_resnet_encoder, get_resnet_w_org_head


def get_model(name, num_class, retrain_head, not_pretrained_layer):
    _name = name.lower()
    if 'resnet' in _name:
        net = get_resnet(pretrained_model_name=_name, num_class=num_class,
                         retrain_head=retrain_head,
                         not_pretrained_layers=not_pretrained_layer)

    return net


def get_init_model_encoder(name, not_pretrained_layer):
    if 'resnet' in name:
        net = get_resnet_encoder(name, not_pretrained_layers=not_pretrained_layer)
    return net


def get_init_model_w_head(name):
    if 'resnet' in name:
        net = get_resnet_w_org_head(name)
    return net


class TLmodel(nn.Module):
    def __init__(self, name, num_class, args, device):
        super(TLmodel, self).__init__()

        self.name = name.lower()
        self.num_class = num_class
        self.args = args
        self.device = device
        self.use_cuda = args.use_cuda

        self.not_pretrained_layer = args.num_not_pretrained_layer
        self.retrain_head = args.retrain_head
        self.model = get_model(name=self.name, num_class=self.num_class,
                               retrain_head=self.retrain_head,
                               not_pretrained_layer=self.not_pretrained_layer)

        self.to(device)

        print("Network parameters")
        for name, param in self.named_parameters():
            if param.requires_grad:
                print(name, param.shape, param.device, param.requires_grad)

    def get_model_w_encoder(self):
        net = get_init_model_encoder(self.name, not_pretrained_layer=self.not_pretrained_layer)
        return net

    def get_model_w_org_head(self):
        net = get_init_model_w_head(self.name)
        return net




