from torch import nn
from torch.nn import init


def weigth_init(m):
    if isinstance(m, nn.Conv2d):
        init.xavier_uniform_(m.weight.data)  ##对参数进行xavier初始化，为了使得网络中信息更好的流动，每一层输出的方差应该尽量相等
        init.constant_(m.bias.data, 0.3)
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm1d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
    elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.03)
        m.bias.data.zero_()
