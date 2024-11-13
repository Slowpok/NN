import NN_init
import torch.nn as nn
import torch

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_chanels, out_chanels, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3(in_chanels, out_chanels, stride)
        self.bn1 = nn.BatchNorm1d(out_chanels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3(out_chanels, out_chanels)
        self.bn2 = nn.BatchNorm1d(out_chanels)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        #print("conv 1 ", out.shape)
        out = self.bn1(out)
        #print("bn 1 ", out.shape)
        out = self.relu(out)

        out = self.conv2(out)
        #print("conv 2 ", out.shape)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1,downsample=None):
        super().__init__()
        self.conv1 = conv1(in_channels, out_channels)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = conv3(out_channels, out_channels, stride)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.conv3 = conv1(out_channels, out_channels*self.expansion)
        self.bn3 = nn.BatchNorm1d(out_channels*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        #print("nachalo bottle")
        out = self.conv1(x)
        #print(out.shape, " bottle conv1")
        out = self.bn1(out)
        #print(out.shape, " bottle bn1")
        out = self.relu(out)
        #print(out.shape, " bottle relu")

        out = self.conv2(out)
        #print(out.shape, " bottle conv2")
        out = self.bn2(out)
        #print(out.shape, " bottle bn2")
        out = self.relu(out)
        #print(out.shape, " bottle relu")

        out = self.conv3(out)
        #print(out.shape, " bottle conv3")
        out = self.bn3(out)
        #print(out.shape, " bottle bn3")

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        #print(out.shape, " bottle +identity")
        out = self.relu(out)
        #print(out.shape, " bottle relu")
        return out


class Resnet(nn.Module):
    cfgs = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
        "resnet101": (Bottleneck, [3, 4, 23, 3]),
        "resnet152": (Bottleneck, [3, 8, 36, 3])
    }

    def __init__(self, name, nettype, unique_words, size_token, num_classes=2, num_of_layers=4):
        if num_of_layers not in [1, 2, 3, 4]:
            print("incorrect num_of_layers")

        super().__init__()
        block, layers = self.cfgs[nettype]

        self.inplanes = 64
        self.embedding_dim = 100
        self.num_classes = num_classes
        self.name = name
        self.num_of_layers = num_of_layers

        self.emb1 = nn.Embedding(unique_words, embedding_dim=self.embedding_dim, max_norm=size_token)
        # self.conv1 = nn.Conv1d(size_token, self.inplanes, 7, stride=2, padding=3, bias=False)
        self.conv1 = nn.Conv1d(self.embedding_dim, self.inplanes, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        if self.num_of_layers >= 2:
            self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        if self.num_of_layers >= 3:
            self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        if self.num_of_layers == 4:
            self.layer4 = self.make_layer(block, 512, layers[3], stride=2)

        avg_num = 13
        num_of_block = 64
        if self.num_of_layers == 2:
            avg_num = 7
            num_of_block = 128
        elif self.num_of_layers == 3:
            avg_num = 4
            num_of_block = 256
        elif self.num_of_layers == 4:
            avg_num = 2
            num_of_block = 512

        self.avgpool = nn.AvgPool1d(avg_num)
        self.drop_out = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(num_of_block * block.expansion, self.num_classes)
        self.sm = nn.Softmax(dim=1)

        # nize - atavizmi
        # # self.avgpool = nn.AvgPool1d(4)
        # # self.drop_out = nn.Dropout(0.5)
        # # self.flatten = nn.Flatten()
        # # self.fc = nn.Linear(512*block.expansion, num_classes)
        # self.avgpool = nn.AvgPool1d(2)  # 4 elsi propyshen 4 sloy
        # # self.avgpool = nn.AvgPool1d(13) # elsi propysheny 3-4 sloy
        # self.drop_out = nn.Dropout(0.5)
        # self.flatten = nn.Flatten()
        # # self.fc = nn.Linear(128*block.expansion, num_classes) # elsi propysheny 3-4 sloy
        # # self.fc = nn.Linear(256 * block.expansion, self.num_classes)  # elsi propyshen 4 sloy
        # self.fc = nn.Linear(512 * block.expansion, self.num_classes)
        # self.sm = nn.Softmax(dim=1)
    def forward(self, x):
        #print("prohod osn classa")
        x = self.emb1(x.long())
        #print(x.shape, " emb1")
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #print(x.shape, " view")
        x = self.conv1(x)
        #print(x.shape, " conv1 osn")
        x = self.bn1(x)
        #print(x.shape, " bn1")
        x = self.relu(x)
        #print(x.shape, " relu")
        x = self.maxpool(x)
        #print(x.shape, " maxpool")

        x = self.layer1(x)
        # print(x.shape, " layer1")
        if self.num_of_layers >= 2:
            x = self.layer2(x)
        # print(x.shape, " layer2")
        if self.num_of_layers >= 3:
            x = self.layer3(x)
        # print(x.shape, " layer3")
        if self.num_of_layers == 4:
            x = self.layer4(x)
        # print(x.shape, " layer4")

        x = self.avgpool(x)
        #print(x.shape, " avgpool")
        x = self.drop_out(x)
        #print(x.shape, " drop_out")
        x = self.flatten(x)
        #print(x.shape, " flatten")
        out = self.fc(x)
        #print(out.shape, " fc")
        out = self.sm(out)

        return out

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []

        downsample = None
        if stride !=1 or self.inplanes !=out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, out_channels*block.expansion, stride),
                nn.BatchNorm1d(out_channels*block.expansion)
            )
        layers.append(
            block(self.inplanes, out_channels, stride, downsample)
        )
        self.inplanes = out_channels*block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, out_channels)
            )
        return nn.Sequential(*layers)


class ClassicResnet(nn.Module):
    cfgs = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
        "resnet101": (Bottleneck, [3, 4, 23, 3]),
        "resnet152": (Bottleneck, [3, 8, 36, 3])
    }

    def __init__(self, name, nettype, unique_words, size_token, num_classes=2):
        super().__init__()
        block, layers = self.cfgs[nettype]

        self.inplanes = 64
        self.embedding_dim = 100
        self.num_classes = num_classes
        self.name = name

        self.emb1 = nn.Embedding(unique_words, embedding_dim=100, max_norm=size_token)
        self.conv1 = nn.Conv1d(size_token, self.inplanes, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        # self.layer4 = self.make_layer(block, 512, layers[3], stride=2)
        # self.avgpool = nn.AvgPool1d(4)
        # self.drop_out = nn.Dropout(0.5)
        # self.flatten = nn.Flatten()
        # self.fc = nn.Linear(512*block.expansion, num_classes)
        self.avgpool = nn.AvgPool1d(4)  # elsi propyshen 4 sloy
        # self.avgpool = nn.AvgPool1d(13) # elsi propysheny 3-4 sloy
        self.drop_out = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        # self.fc = nn.Linear(128*block.expansion, num_classes) # elsi propysheny 3-4 sloy
        self.fc = nn.Linear(256 * block.expansion, self.num_classes)  # elsi propyshen 4 sloy
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        #print("prohod osn classa")
        x = self.emb1(x.long())
        #print(x.shape, " emb1")
        # x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #print(x.shape, " view")
        x = self.conv1(x)
        #print(x.shape, " conv1 osn")
        x = self.bn1(x)
        #print(x.shape, " bn1")
        x = self.relu(x)
        #print(x.shape, " relu")
        x = self.maxpool(x)
        #print(x.shape, " maxpool")

        x = self.layer1(x)
        #print(x.shape, " layer1")
        x = self.layer2(x)
        #print(x.shape, " layer2")
        x = self.layer3(x)
        #print(x.shape, " layer3")
        # x = self.layer4(x)
        # print(x.shape, " layer4")

        x = self.avgpool(x)
        #print(x.shape, " avgpool")
        x = self.drop_out(x)
        #print(x.shape, " drop_out")
        x = self.flatten(x)
        #print(x.shape, " flatten")
        out = self.sigm(self.fc(x))
        #print(out.shape, " fc")
        out = torch.squeeze(out, 1)
        return out

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []

        downsample = None
        if stride !=1 or self.inplanes !=out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, out_channels*block.expansion, stride),
                nn.BatchNorm1d(out_channels*block.expansion)
            )
        layers.append(
            block(self.inplanes, out_channels, stride, downsample)
        )
        self.inplanes = out_channels*block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, out_channels)
            )
        return nn.Sequential(*layers)

class ResnetTest(nn.Module):
    cfgs = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
        "resnet101": (Bottleneck, [3, 4, 23, 3]),
        "resnet152": (Bottleneck, [3, 8, 36, 3])
    }

    def __init__(self, name, nettype, unique_words, size_token, num_classes=2):
        super().__init__()
        block, layers = self.cfgs[nettype]

        self.inplanes = 64
        self.embedding_dim = 100
        self.num_classes = num_classes
        self.name = name

        self.emb1 = nn.Embedding(unique_words, embedding_dim=self.embedding_dim, max_norm=size_token)
        self.conv1 = nn.Conv1d(self.embedding_dim, self.inplanes, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self.make_layer(block, 256, layers[2], stride=2)
        self.avgpool = nn.AvgPool1d(4)
        self.drop_out = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(256 * block.expansion, num_classes)  # elsi propyshen 4 sloy
        self.sigm = nn.Sigmoid()

    def forward(self, x):
        #print("prohod osn classa")
        x = self.emb1(x.long())
        #print(x.shape, " emb1")
        x = x.view(x.shape[0], x.shape[2], x.shape[1])
        #x = x.transpose(1, 2).contiguous()
        #print(x.shape, " transpose")
        x = self.conv1(x)
        #print(x.shape, " conv1 osn")
        x = self.bn1(x)
        #print(x.shape, " bn1 osn")
        x = self.relu(x)
        #print(x.shape, " relu osn")
        x = self.maxpool(x)
        #print(x.shape, " maxpool osn")

        x = self.layer1(x)
        #print(x.shape, " layer1 osn")
        x = self.layer2(x)
        #print(x.shape, " layer2 osn")
        x = self.layer3(x)
        #print(x.shape, " layer3 osn")

        x = self.avgpool(x)
        #print(x.shape, " avgpool osn")
        x = self.drop_out(x)
        #print(x.shape, " drop_out osn")
        x = self.flatten(x)
        #print(x.shape, " flatten osn")
        x = self.fc(x)
        #print(x.shape, " fc osn")
        out = self.sigm(x)
        if self.num_classes == 1:
            # out = self.flatten(out)
            out = torch.reshape(out, (-1,))
            # out = [1 if x > 0.5 else 0 for x in out]
            # out = torch.as_tensor(out).long()
        return out

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []

        downsample = None
        if stride !=1 or self.inplanes !=out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, out_channels*block.expansion, stride),
                nn.BatchNorm1d(out_channels*block.expansion)
            )
        layers.append(
            block(self.inplanes, out_channels, stride, downsample)
        )
        self.inplanes = out_channels*block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, out_channels)
            )
        return nn.Sequential(*layers)

class ResnetTwoLayer(nn.Module):
    cfgs = {
        "resnet18": (BasicBlock, [2, 2, 2, 2]),
        "resnet34": (BasicBlock, [3, 4, 6, 3]),
        "resnet50": (Bottleneck, [3, 4, 6, 3]),
        "resnet101": (Bottleneck, [3, 4, 23, 3]),
        "resnet152": (Bottleneck, [3, 8, 36, 3])
    }

    def __init__(self, name, unique_words, size_token, num_classes=2):
        super().__init__()
        block, layers = self.cfgs[name]

        self.inplanes = 64

        self.emb1 = nn.Embedding(unique_words, embedding_dim=100, max_norm=size_token)
        self.conv1 = nn.Conv1d(size_token, self.inplanes, 7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(3, stride=2, padding=1)
        self.layer1 = self.make_layer(block, 64, layers[0])
        self.layer2 = self.make_layer(block, 128, layers[1], stride=2)

        self.avgpool = nn.AvgPool1d(13) # elsi propysheny 3-4 sloy
        self.drop_out = nn.Dropout(0.5)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128*block.expansion, num_classes) # elsi propysheny 3-4 sloy


    def forward(self, x):
        #print("prohod osn classa")
        x = self.emb1(x.long())
        #print(x.shape, " emb1")
        x = self.conv1(x)
        #print(x.shape, " conv1")
        x = self.bn1(x)
        #print(x.shape, " bn1")
        x = self.relu(x)
        #print(x.shape, " relu")
        x = self.maxpool(x)
        #print(x.shape, " maxpool")

        x = self.layer1(x)
        #print(x.shape, " layer1")
        x = self.layer2(x)
        #print(x.shape, " layer2")

        x = self.avgpool(x)
        #print(x.shape, " avgpool")
        x = self.drop_out(x)
        #print(x.shape, " drop_out")
        x = self.flatten(x)
        #print(x.shape, " flatten")
        out = self.fc(x)
        # print(out.shape, " fc")

        return out

    def make_layer(self, block, out_channels, blocks, stride=1):
        layers = []

        downsample = None
        if stride !=1 or self.inplanes !=out_channels * block.expansion:
            downsample = nn.Sequential(
                conv1(self.inplanes, out_channels*block.expansion, stride),
                nn.BatchNorm1d(out_channels*block.expansion)
            )
        layers.append(
            block(self.inplanes, out_channels, stride, downsample)
        )
        self.inplanes = out_channels*block.expansion

        for _ in range(1, blocks):
            layers.append(
                block(self.inplanes, out_channels)
            )
        return nn.Sequential(*layers)

def conv3(in_chan, out_chan, stride=1):
    return nn.Conv1d(in_chan,
                     out_chan,
                     kernel_size=3,
                     stride=stride,
                     padding=1,
                     bias=False)


def conv1(in_chan, out_chan, stride=1):
    return nn.Conv1d(in_chan,
                     out_chan,
                     kernel_size=1,
                     stride=stride,
                     bias=False)