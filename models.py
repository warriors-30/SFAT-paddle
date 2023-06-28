import math
import paddle
import paddle.nn.functional as F
from paddle import nn
from torch.autograd import Variable


class MLP(nn.Layer):
    def __init__(self, dim_in, dim_hidden, dim_out):
        super(MLP, self).__init__()
        self.layer_input = nn.Linear(dim_in, dim_hidden)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout()
        self.layer_hidden = nn.Linear(dim_hidden, dim_out)
        self.softmax = nn.Softmax(axis=1)
        self.flatten = nn.Flatten()

    def forward(self, x):
        #x = x.view(-1, x.shape[1]*x.shape[-2]*x.shape[-1])
        x = self.flatten(x)
        x = self.layer_input(x)
        x = self.dropout(x)
        x = self.relu(x)
        x = self.layer_hidden(x)
        return self.softmax(x)


class CNNMnist(nn.Layer):
    def __init__(self, args):
        super(CNNMnist, self).__init__()
        self.conv1 = nn.Conv2D(args.num_channels, 10, kernel_size=5)
        self.conv2 = nn.Conv2D(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2D()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, args.num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        #x = x.view(-1, x.shape[1]*x.shape[2]*x.shape[3])
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, axis=1)


class CNNFashion_Mnist(nn.Layer):
    def __init__(self):
        super(CNNFashion_Mnist, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2D(1, 16, kernel_size=5, padding=2),
            nn.BatchNorm2D(16),
            nn.ReLU(),
            nn.MaxPool2D(2))
        self.layer2 = nn.Sequential(
            nn.Conv2D(16, 32, kernel_size=5, padding=2),
            nn.BatchNorm2D(32),
            nn.ReLU(),
            nn.MaxPool2D(2))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7*7*32, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        #out = out.view(out.size(0), -1)
        out = self.flatten(out)
        out = self.fc(out)
        return out


class CNNCifar(nn.Layer):
    def __init__(self, args):
        super(CNNCifar, self).__init__()
        self.conv1 = nn.Conv2D(3, 6, 5)
        self.pool = nn.MaxPool2D(2, 2)
        self.conv2 = nn.Conv2D(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, args.num_classes)
        self.flatten=nn.Flatten()

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        #x = x.view(-1, 16 * 5 * 5)
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, axis=1)

class modelC(nn.Layer):
    def __init__(self, input_size, n_classes=10, **kwargs):
        super(modelC, self).__init__()
        self.conv1 = nn.Conv2D(input_size, 96, 3, padding=1)
        self.conv2 = nn.Conv2D(96, 96, 3, padding=1)
        self.conv3 = nn.Conv2D(96, 96, 3, padding=1, stride=2)
        self.conv4 = nn.Conv2D(96, 192, 3, padding=1)
        self.conv5 = nn.Conv2D(192, 192, 3, padding=1)
        self.conv6 = nn.Conv2D(192, 192, 3, padding=1, stride=2)
        self.conv7 = nn.Conv2D(192, 192, 3, padding=1)
        self.conv8 = nn.Conv2D(192, 192, 1)

        self.class_conv = nn.Conv2D(192, n_classes, 1)


    def forward(self, x):
        x_drop = F.dropout(x, .2)
        conv1_out = F.relu(self.conv1(x_drop))
        conv2_out = F.relu(self.conv2(conv1_out))
        conv3_out = F.relu(self.conv3(conv2_out))
        conv3_out_drop = F.dropout(conv3_out, .5)
        conv4_out = F.relu(self.conv4(conv3_out_drop))
        conv5_out = F.relu(self.conv5(conv4_out))
        conv6_out = F.relu(self.conv6(conv5_out))
        conv6_out_drop = F.dropout(conv6_out, .5)
        conv7_out = F.relu(self.conv7(conv6_out_drop))
        conv8_out = F.relu(self.conv8(conv7_out))

        class_out = F.relu(self.class_conv(conv8_out))
        pool_out = F.adaptive_avg_pool2d(class_out, 1)
        pool_out.squeeze_(-1)
        pool_out.squeeze_(-1)
        return pool_out


class NIN(nn.Layer):
    def __init__(self, num_classes=10):
        super(NIN, self).__init__()
        self.num_classes = num_classes

        conv_weight_attr_list=[]
        conv_bias_attr_list = []
        #paddle初始化
        for i in range(9):
            conv_weight_attr_list.append(paddle.ParamAttr(
                name=f"weight_{i}",initializer=paddle.nn.initializer.Normal(mean=0.0,std=0.05)))

            conv_bias_attr_list.append(paddle.ParamAttr(
                name=f"bias_{i}",initializer=paddle.nn.initializer.Constant(value=0.0)))

        self.features = nn.Sequential(
            nn.Conv2D(3, 192, 5, padding=2,weight_attr=conv_weight_attr_list[0],bias_attr=conv_bias_attr_list[0]),
            nn.ReLU(),
            nn.Conv2D(192, 160, 1,weight_attr=conv_weight_attr_list[1],bias_attr=conv_bias_attr_list[1]),
            nn.ReLU(),
            nn.Conv2D(160, 96, 1,weight_attr=conv_weight_attr_list[2],bias_attr=conv_bias_attr_list[2]),
            nn.ReLU(),
            nn.MaxPool2D(3, stride=2, ceil_mode=True),
            nn.Dropout(),

            nn.Conv2D(96, 192, 5, padding=2,weight_attr=conv_weight_attr_list[3],bias_attr=conv_bias_attr_list[3]),
            nn.ReLU(),
            nn.Conv2D(192, 192, 1,weight_attr=conv_weight_attr_list[4],bias_attr=conv_bias_attr_list[4]),
            nn.ReLU(),
            nn.Conv2D(192, 192, 1,weight_attr=conv_weight_attr_list[5],bias_attr=conv_bias_attr_list[5]),
            nn.ReLU(),
            nn.AvgPool2D(3, stride=2, ceil_mode=True),
            nn.Dropout(),

            nn.Conv2D(192, 192, 3, padding=1,weight_attr=conv_weight_attr_list[6],bias_attr=conv_bias_attr_list[6]),
            nn.ReLU(),
            nn.Conv2D(192, 192, 1,weight_attr=conv_weight_attr_list[7],bias_attr=conv_bias_attr_list[7]),
            nn.ReLU(),
            nn.Conv2D(192, self.num_classes, 1,weight_attr=conv_weight_attr_list[8],bias_attr=conv_bias_attr_list[8]),
            nn.ReLU(),
            nn.AvgPool2D(8, stride=1)
        )
        #self._initialize_weights()
        self.flatten=nn.Flatten()

    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), self.num_classes)
        x = self.flatten(x)
        return x
                    


class VGG(nn.Layer):
    '''
    VGG model 
    '''
    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        self.flatten = nn.Flatten()
         # Initialize weights
        #for m in self.modules():
        #    if isinstance(m, nn.Conv2d):
        #        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        #        m.weight.data.normal_(0, math.sqrt(2. / n))
        #        m.bias.data.zero_()


    def forward(self, x):
        x = self.features(x)
        #x = x.view(x.size(0), -1)
        x = self.flatten(x)
        x = self.classifier(x)
        return x


def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for id,v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            n = 3 * 3 * v
            conv_weight_attr=paddle.ParamAttr(
                name=f"weight_{id}", initializer=paddle.nn.initializer.Normal(mean=0.0, std=math.sqrt(2. / n)))
            conv_bias_attr=paddle.ParamAttr(
                name=f"bias_{id}", initializer=paddle.nn.initializer.Constant(value=0.0))

            conv2d = nn.Conv2D(in_channels, v, kernel_size=3, padding=1,
                               weight_attr=conv_weight_attr,bias_attr=conv_bias_attr)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 
          512, 512, 512, 512, 'M'],
}


def vgg11():
    """VGG 11-layer model (configuration "A")"""
    return VGG(make_layers(cfg['A']))


def vgg11_bn():
    """VGG 11-layer model (configuration "A") with batch normalization"""
    return VGG(make_layers(cfg['A'], batch_norm=True))


def vgg13():
    """VGG 13-layer model (configuration "B")"""
    return VGG(make_layers(cfg['B']))


def vgg13_bn():
    """VGG 13-layer model (configuration "B") with batch normalization"""
    return VGG(make_layers(cfg['B'], batch_norm=True))


def vgg16():
    """VGG 16-layer model (configuration "D")"""
    return VGG(make_layers(cfg['D']))


def vgg16_bn():
    """VGG 16-layer model (configuration "D") with batch normalization"""
    return VGG(make_layers(cfg['D'], batch_norm=True))


def vgg19():
    """VGG 19-layer model (configuration "E")"""
    return VGG(make_layers(cfg['E']))


def vgg19_bn():
    """VGG 19-layer model (configuration 'E') with batch normalization"""
    return VGG(make_layers(cfg['E'], batch_norm=True))
    
class BasicBlock(nn.Layer):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Layer):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2D(in_planes, planes, kernel_size=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(planes)
        self.conv2 = nn.Conv2D(planes, planes, kernel_size=3, stride=stride, padding=1, bias_attr=False)
        self.bn2 = nn.BatchNorm2D(planes)
        self.conv3 = nn.Conv2D(planes, self.expansion*planes, kernel_size=1, bias_attr=False)
        self.bn3 = nn.BatchNorm2D(self.expansion*planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2D(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias_attr=False),
                nn.BatchNorm2D(self.expansion*planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Layer):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2D(3, 64, kernel_size=3, stride=1, padding=1, bias_attr=False)
        self.bn1 = nn.BatchNorm2D(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)
        self.flatten = nn.Flatten()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        #out = out.view(out.size(0), -1)
        out = self.flatten(out)
        out = self.linear(out)
        return out


def ResNet18(num_classes=100):
    return ResNet(BasicBlock, [2,2,2,2], num_classes=num_classes)

def ResNet34(num_classes=100):
    return ResNet(BasicBlock, [3,4,6,3], num_classes=num_classes)

def ResNet50(num_classes=100):
    return ResNet(Bottleneck, [3,4,6,3], num_classes=num_classes)

def ResNet101(num_classes=100):
    return ResNet(Bottleneck, [3,4,23,3], num_classes=num_classes)

def ResNet152(num_classes=100):
    return ResNet(Bottleneck, [3,8,36,3], num_classes=num_classes)

class SmallCNN(nn.Layer):
    def __init__(self):
        super(SmallCNN, self).__init__()

        self.block1_conv1 = nn.Conv2D(3, 64, 3, padding=1)
        self.block1_conv2 = nn.Conv2D(64, 64, 3, padding=1)
        self.block1_pool1 = nn.MaxPool2D(2, 2)
        self.batchnorm1_1 = nn.BatchNorm2D(64)
        self.batchnorm1_2 = nn.BatchNorm2D(64)

        self.block2_conv1 = nn.Conv2D(64, 128, 3, padding=1)
        self.block2_conv2 = nn.Conv2D(128, 128, 3, padding=1)
        self.block2_pool1 = nn.MaxPool2D(2, 2)
        self.batchnorm2_1 = nn.BatchNorm2D(128)
        self.batchnorm2_2 = nn.BatchNorm2D(128)

        self.block3_conv1 = nn.Conv2D(128, 196, 3, padding=1)
        self.block3_conv2 = nn.Conv2D(196, 196, 3, padding=1)
        self.block3_pool1 = nn.MaxPool2D(2, 2)
        self.batchnorm3_1 = nn.BatchNorm2D(196)
        self.batchnorm3_2 = nn.BatchNorm2D(196)

        self.activ = nn.ReLU()

        self.fc1 = nn.Linear(196*4*4,256)
        self.fc2 = nn.Linear(256,10)
        self.flatten = nn.Flatten()

    def forward(self, x):
        #block1
        x = self.block1_conv1(x)
        x = self.batchnorm1_1(x)
        x = self.activ(x)
        x = self.block1_conv2(x)
        x = self.batchnorm1_2(x)
        x = self.activ(x)
        x = self.block1_pool1(x)

        #block2
        x = self.block2_conv1(x)
        x = self.batchnorm2_1(x)
        x = self.activ(x)
        x = self.block2_conv2(x)
        x = self.batchnorm2_2(x)
        x = self.activ(x)
        x = self.block2_pool1(x)
        #block3
        x = self.block3_conv1(x)
        x = self.batchnorm3_1(x)
        x = self.activ(x)
        x = self.block3_conv2(x)
        x = self.batchnorm3_2(x)
        x = self.activ(x)
        x = self.block3_pool1(x)

        #x = x.view(-1,196*4*4)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.activ(x)
        x = self.fc2(x)

        return x

def small_cnn():
    return SmallCNN()

def test():
    net = small_cnn()
    #y = net(Variable(paddle.randn(1,3,32,32)))
    y = net(paddle.randn(1, 3, 32, 32))
    print(y.size())
    print(net)