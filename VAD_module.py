import torch
import torch.nn as nn
import torch.nn.functional as F
from utils import HParams
hparamas = HParams()


class spec_conv(nn.Module):
    def __init__(self, hparamas=None):
        super(spec_conv, self).__init__()
        self.layers = hparamas.layers  # 4
        self.filter_width = hparamas.filter_width
        self.conv_channels = hparamas.conv_channels
        self.conv_layers1 = nn.ModuleList()
        self.conv_layers2 = nn.ModuleList()
        self.batch_norm = nn.ModuleList()
        self.sigmoid = nn.Sigmoid()
        self.max_pool2d = nn.ModuleList()

        for layer in range(self.layers):
            if layer == 0:
                self.conv_layers1.append(nn.Conv2d(in_channels=1, out_channels=self.conv_channels * (2 ** layer),
                                                   kernel_size=self.filter_width, padding=(1, 1)))
                self.conv_layers2.append(nn.Conv2d(in_channels=1, out_channels=self.conv_channels * (2 ** layer),
                                                   kernel_size=self.filter_width, padding=(1, 1)))
            else:
                self.conv_layers1.append(nn.Conv2d(in_channels=self.conv_channels * 2 ** (layer - 1),
                                                   out_channels=self.conv_channels * (2 ** layer),
                                                   kernel_size=self.filter_width, padding=(1, 1)))
                self.conv_layers2.append(nn.Conv2d(in_channels=self.conv_channels * 2 ** (layer - 1),
                                                   out_channels=self.conv_channels * (2 ** layer),
                                                   kernel_size=self.filter_width, padding=(1, 1)))
            self.batch_norm.append(nn.BatchNorm2d(self.conv_channels * (2 ** layer)))
            self.batch_norm.append(nn.BatchNorm2d(self.conv_channels * (2 ** layer)))
            self.max_pool2d.append(nn.MaxPool2d(kernel_size=(2, 2), dilation=(2, 1), stride=(1, 2), padding=(1, 0)))

    def forward(self, inputs):
        x = torch.unsqueeze(inputs, 1)
        num_time = list(x.size())[2]
        # conv_list = list()
        i = 0
        for conv_linear, conv_sigmoid in zip(self.conv_layers1, self.conv_layers2):
            x_linear = conv_linear(x)
            x_sigmoid = conv_sigmoid(x)
            x_linear = self.batch_norm[i](x_linear)
            x_sigmoid = self.batch_norm[i + 1](x_sigmoid)
            x = self.sigmoid(x_sigmoid) * x_linear
            x = self.max_pool2d[i // 2](x)
            # conv_list.append(x)
            i += 2
        x = torch.reshape(x, (-1, num_time, list(x.size())[1] * list(x.size())[3]))
        return x


class Prenet(nn.Module):
    def __init__(self, in_units=1280, layers_size=[256, 256], drop_rate=0.5, activation=nn.ReLU(), pipenet=True):
        super(Prenet, self).__init__()
        #in_units=640 for 80mel, 1280 for 160 mel
        self.layers_size = layers_size
        self.activation = activation
        self.pipenet = pipenet
        self.linear = nn.ModuleList()
        self.bn = nn.ModuleList()
        self.dropout = nn.ModuleList()

        self.linear.append(nn.Linear(in_units, layers_size[0]))
        self.linear.append(nn.Linear(layers_size[0], layers_size[1]))
        self.pipe_linear = nn.Linear(layers_size[1], 1)
        self.bn.append(nn.BatchNorm1d(layers_size[0]))
        self.bn.append(nn.BatchNorm1d(layers_size[1]))
        self.dropout.append(nn.Dropout(drop_rate))
        self.dropout.append(nn.Dropout(drop_rate))

    def forward(self, inputs):
        x = inputs
        for i, size in enumerate(self.layers_size):
            dense = self.linear[i](x)
            dense = self.bn[i](dense.permute([0, 2, 1]))
            dense = self.activation(dense.permute([0, 2, 1]))
            x = self.dropout[i](dense)
        if self.pipenet:
            return x, torch.squeeze(self.pipe_linear(x), axis=-1)
        else:
            return x


class multihead_attention(nn.Module):
    def __init__(self, in_units=256, num_units=128, num_heads=8, activation=nn.ReLU()):
        super().__init__()
        self.num_units = num_units
        self.num_heads = num_heads
        self.linear = nn.ModuleList()
        self.linear.append(nn.Linear(in_units, num_units))
        self.linear.append(nn.Linear(in_units, num_units))
        self.linear.append(nn.Linear(in_units, num_units))
        self.activation = activation
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, quires, keys):
        quires = quires.mean(axis=1, keepdim=True)
        Q = self.activation(self.linear[0](quires))
        K = self.activation(self.linear[1](keys))
        V = self.activation(self.linear[2](keys))
        num_heads = int(Q.shape[-1] / self.num_heads)
        Q_ = torch.cat(torch.split(Q, num_heads, dim=2), dim=0)
        K_ = torch.cat(torch.split(K, num_heads, dim=2), dim=0)
        V_ = torch.cat(torch.split(V, num_heads, dim=2), dim=0)
        outputs = torch.matmul(Q_, K_.permute(0, 2, 1)) / (self.num_units ** 0.5)
        attention_weights = self.softmax(outputs)
        outputs = torch.mul(attention_weights.permute(0, 2, 1), V_)
        num_heads = outputs.shape[0] // self.num_heads
        outputs = torch.cat(torch.split(outputs, num_heads, dim=0), dim=2)
        num_heads = attention_weights.shape[0] // self.num_heads
        attention_weights = torch.cat(torch.split(attention_weights, num_heads, dim=0), dim=1)
        return outputs, attention_weights


class Postnet(nn.Module):
    def __init__(self, in_units=128, layers_size=[256, 1], drop_rate=0.5, activation=nn.ReLU()):
        super(Postnet, self).__init__()
        self.layers_size = layers_size
        self.activation = activation
        self.linear = []

        self.linear.append(nn.Linear(in_units, layers_size[0]))
        self.linear.append(nn.Linear(layers_size[0], layers_size[1]))
        self.batch_norm = nn.BatchNorm1d(layers_size[0])
        self.dropout = nn.Dropout(drop_rate)

        self.linear = nn.ModuleList(self.linear)

    def forward(self, inputs):
        x = inputs
        for i, size in enumerate(self.layers_size):
            dense = self.linear[i](x)
            if i < len(self.layers_size) - 1:
                dense = self.batch_norm(dense.permute([0, 2, 1]))
                dense = self.activation(dense.permute([0, 2, 1]))
                x = self.dropout(dense)
            else:
                x = dense
        return torch.squeeze(x, dim=-1)


def smooth_softmax(x):
    return F.sigmoid(x) / torch.unsqueeze(torch.sum(F.sigmoid(x), dim=-1), dim=-1)


class VAD(nn.Module):
    def __init__(self, hparamas, activation=nn.ReLU(), in_units=640):
        super(VAD, self).__init__()
        self._hparamas = hparamas

        self.spec_att = spec_conv(hparamas=self._hparamas)
        self.pipenet = Prenet(pipenet=True, activation=activation, in_units=in_units)
        self.mha = multihead_attention(num_units=hparamas.num_att_units, activation=activation)
        self.postnet = Postnet(activation=activation)

    def forward(self, inputs):
        spec_att_output = self.spec_att(inputs)
        pipenet_output, midnet_output = self.pipenet(spec_att_output)
        z, alpha = self.mha(pipenet_output, pipenet_output)
        postnet_output = self.postnet(z)

        return midnet_output, postnet_output, alpha


def add_loss(model, pipenet_targets, postnet_output, pipenet_output, alpha):
    postnet_loss = (F.binary_cross_entropy_with_logits(postnet_output, pipenet_targets)).mean()
    pipenet_loss = (F.binary_cross_entropy_with_logits(pipenet_output, pipenet_targets)).mean()
    attention_loss = 0.1 * (F.binary_cross_entropy_with_logits(alpha.mean(dim=1), pipenet_targets)).mean()
    regularization = sum([torch.norm(val, 2) for name, val in model.named_parameters() if
                          'weight' in name]).item() * hparamas.vad_reg_weight
    total_loss = postnet_loss + pipenet_loss + attention_loss + regularization
    return total_loss, postnet_loss, pipenet_loss, attention_loss


class Scheduler:
    "Optim wrapper that implements rate."

    def __init__(self, optimizer, init_lr=1e-3, final_lr=1e-5, decay_rate=0.6, start_decay=50000, decay_steps=25000):
        self._step = 0
        self.init_lr = init_lr
        self.final_lr = final_lr
        self.decay_rate = decay_rate
        self.start_decay = start_decay
        self.decay_steps = decay_steps
        self.optimizer = optimizer
        self._rate = 0

    def step(self):
        "Update parameters and rate"
        self._step += 1
        rate = self.rate()
        for p in self.optimizer.param_groups:
            p['lr'] = rate
        self._rate = rate

    def rate(self):
        "Implement `lrate` above"
        if self._step < self.start_decay:
            return self.init_lr
        return min(
            max(self.init_lr * self.decay_rate ** ((self._step - self.start_decay) / self.decay_steps), self.final_lr),
            self.init_lr)
