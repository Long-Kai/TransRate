import torch
import torch.nn as nn
import torch.nn.functional as F
# from torch.utils.model_zoo import load_url as load_state_dict_from_url


model_urls = {
    'resnet18_imagenet': 'resnet18-5c106cde.pth',
}

num_class_dataset = {
    'resnet18_imagenet': 1000,
}

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion *
                               planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.downsample = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.downsample(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, layers, num_class, retrain_head):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, self.in_planes, kernel_size=7,
                               stride=2, padding=3, bias=False)  # for imagenet images: input n * 3 * 224 * 224
        self.bn1 = nn.BatchNorm2d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], stride=1)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        if len(layers) == 4:
            self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
            self.fc = nn.Linear(512 * block.expansion, num_class)
            self.has_layer4 = True
        else:
            self.fc = nn.Linear(256 * block.expansion, num_class)
            self.has_layer4 = False
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.retrain_head = retrain_head
        if self.retrain_head:
            for name, param in self.named_parameters():
                if param.requires_grad:
                    if name not in ['fc.weight', 'fc.bias']:
                        param.requires_grad = False  # set false for filter out in optimizer
                        print(name)

    def load_pretrain_model(self, pretrained_dict, remove_head=True):
        model_dict = self.state_dict()

        if remove_head:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if
                               k in model_dict and k not in ['fc.weight', 'fc.bias']}
        else:
            pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        print("pretrained_dict_len", len(pretrained_dict))

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict)


    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.has_layer4:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def extract_feature(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        if self.has_layer4:
            x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)

        return x


def ShrinkResNet18(num_class, retrain_head, pretrained=True, delete_layers=None, pretrained_model_name='resnet18_imagenet'):
    if delete_layers is not None:
        if delete_layers <= 1:
            layers = [2, 2, 2, 2 - delete_layers]
        elif delete_layers == 2:
            layers = [2, 2, 2]
        elif delete_layers == 3:
            layers = [2, 2, 1]
        else:
            layers = [2, 2, 1]
    else:
        layers = [2, 2, 2, 2]
    net = ResNet(BasicBlock, layers, num_class=num_class, retrain_head=retrain_head)
    if pretrained:
        model_url = './pretrained_models/' + model_urls[pretrained_model_name]
        pretrained_dict = torch.load(model_url)
        net.load_pretrain_model(pretrained_dict)
    return net


def ResNet18(num_class, retrain_head, not_pretrained_layers=None, pretrained_model_name='resnet18_imagenet'):
    if not_pretrained_layers is None or not_pretrained_layers == 0:
        return ShrinkResNet18(num_class, retrain_head=retrain_head, pretrained=True,
                              pretrained_model_name=pretrained_model_name)
    else:
        net = ShrinkResNet18(num_class, retrain_head=retrain_head, pretrained=False,
                             pretrained_model_name=pretrained_model_name)
        pretrained_net = ShrinkResNet18(num_class, retrain_head=False, pretrained=True,
                                        delete_layers=not_pretrained_layers, pretrained_model_name=pretrained_model_name)
        pretrained_dict = pretrained_net.state_dict()
        net.load_pretrain_model(pretrained_dict)
        return net


def get_resnet(pretrained_model_name, num_class, retrain_head, not_pretrained_layers=None):
    # load pretrained model without head (new head for downstream tasks) (and extra "not_pretrained layers")
    print("loading " + pretrained_model_name + ", excluding head")

    net = ResNet18(num_class, retrain_head=retrain_head, not_pretrained_layers=not_pretrained_layers,
                   pretrained_model_name=pretrained_model_name)

    return net


def get_resnet_encoder(pretrained_model_name, not_pretrained_layers=None):
    # load pretrained model without head (and extra "not_pretrained layers") -- for feature extraction
    num_class = 2
    retrain_head = False
    print("loading " + pretrained_model_name + f" L-{not_pretrained_layers} layer" + " feature encoder")

    net = ShrinkResNet18(num_class, retrain_head=retrain_head, pretrained=True,
                         delete_layers=not_pretrained_layers, pretrained_model_name=pretrained_model_name)

    return net


def get_resnet_w_org_head(pretrained_model_name):
    # load pretrained model with head
    print("loading original" + pretrained_model_name)

    layers = [2, 2, 2, 2]

    net = ResNet(BasicBlock, layers, num_class=num_class_dataset[pretrained_model_name], retrain_head=False)
    model_url = './pretrained_models/' + model_urls[pretrained_model_name]
    pretrained_dict = torch.load(model_url)
    net.load_pretrain_model(pretrained_dict, remove_head=False)

    return net


def test_ResNet():
    torch.manual_seed(0)

    x = torch.randn(4, 3, 224, 224)

    net = get_resnet_w_org_head('resnet18_imagenet')
    # net = get_resnet_encoder('resnet18_imagenet')
    # net = get_resnet('resnet18_imagenet', 10, False, 2)
    y = net.extract_feature(x)
    print(y.shape)


test_ResNet()


