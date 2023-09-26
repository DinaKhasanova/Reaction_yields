from torch.nn import Linear, LayerNorm, ModuleList
from torch.nn.functional import cross_entropy
from torch.nn.functional import softmax
from torch.nn.init import xavier_uniform_, kaiming_uniform_, zeros_
import pytorch_lightning as pl
import torch
import numpy as np
from adabelief_pytorch import AdaBelief  # (pip install adabelief-pytorch==0.2.1)
from sklearn.metrics import balanced_accuracy_score
import warnings
from torch.utils.data import DataLoader, TensorDataset
from pytorch_lightning.loggers import CSVLogger
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.callbacks import ModelCheckpoint

pl.seed_everything(42)


def LrScheduledOptimizer(Optimizer):
    class LROptimizer(Optimizer):
        def __init__(self, *args, **kwargs):
            self.lr_func = kwargs['lr_func']
            self.step_n = 0
            del kwargs['lr_func']
            super().__init__(*args, **kwargs)

        def __setstate__(self, state):
            super().__setstate__(state)

        def update_lr(self):
            lr = self.lr_func(self.step_n)
            for param_group in self.param_groups:
                param_group['lr'] = lr

        def step(self, closure=None):
            self.step_n += 1
            self.update_lr()
            return super().step(closure=closure)

    return LROptimizer


def lr_wm(multiplier):
    def f(n):
        return multiplier * np.power(512, -0.5) * min(np.power(n, -0.5), n * np.power(8000, -1.5))

    return f


class GroupLinear(torch.nn.Module):
    def __init__(self, dim1, dim2, out_dim, groups, shuffle):
        super().__init__()

        self.dim1 = dim1
        self.dim2 = dim2
        self.out_dim = out_dim
        self.groups = groups
        self.shuffle = shuffle
        group_inp = (dim1 + dim2) // groups
        group_out = out_dim // groups

        self.W = torch.nn.parameter.Parameter(torch.empty(groups, 1, group_inp, group_out),
                                              requires_grad=True)  # g, b(1), inp, out
        xavier_uniform_(self.W.data, gain=1.0)

        self.b = torch.nn.parameter.Parameter(torch.empty(groups, 1, 1, group_out),
                                              requires_grad=True)  # g, b(1), 1, out
        zeros_(self.b.data)

    def forward(self, x):
        batch = x.shape[0]
        seq = x.shape[1]

        # group features 
        x = x.view((batch, seq, self.groups, -1))  # batch, seq, dim + inp -> batch, seq, g, dim//g + inp//g
        x = torch.permute(x, (2, 0, 1, 3))  # batch, seq, g, dim//g + inp//g -> g, batch, seq, dim//g + inp//g

        # transform
        x = torch.matmul(x, self.W)  # g, batch, seq, dim//g + inp//g -> g, batch, seq, out_dim//g
        x += self.b

        if self.shuffle:
            x = torch.permute(x, (1, 2, 3, 0)).contiguous()  # g, batch, seq, out_dim//g -> batch, seq, out_dim//g, g
        else:
            x = torch.permute(x, (1, 2, 0, 3)).contiguous()  # g, batch, seq, out_dim//g -> batch, seq, g, out_dim//g

        return x.view((batch, seq, self.out_dim))  # -> batch, seq, out_dim


class GLT(torch.nn.Module):
    def __init__(self, dim, N, wm, out_dim=None):
        super().__init__()

        self.dim = dim
        self.N = N
        self.wm = wm
        if out_dim is None:
            out_dim = dim // 2

        # max number of groups
        g_max = dim // 32
        # num of expansion layers
        mid_point = int(np.ceil(N / 2.0))

        groups = [min(2 ** (l + 1), g_max) for l in range(mid_point)]  # expansion groups
        groups += groups[::-1][int(N % 2 != 0):]  # + reduction groups
        for i, g in enumerate(groups):
            if dim % g != 0:
                raise ValueError(f'Input dim must be divisible by group num {g} on layer {i + 1}')
        self.groups = groups

        expansion_shapes = np.linspace(dim, dim * wm, mid_point).tolist()
        reduction_shapes = np.linspace(dim * wm, out_dim, (N - mid_point)).tolist() if (N - mid_point) > 1 else [
            out_dim]
        out_shapes = expansion_shapes + reduction_shapes

        for i in range(len(out_shapes) - 1):
            # adjust dim to be divisible by number of groups on i and i-1 layers
            div = groups[i] * groups[i + 1]
            # divisible part of dim + num of groups if needed
            out_shapes[i] = (out_shapes[i] // div) * div + int(out_shapes[i] % div != 0) * div
        out_shapes[-1] = (out_shapes[-1] // groups[-1]) * groups[-1] + int(out_shapes[-1] % groups[-1] != 0) * groups[
            -1]

        inp_shapes = [dim] + out_shapes[:-1]

        self.group_linears = ModuleList()
        for i in range(N):
            # + Inp dim from mixer if not the first layer
            dim2 = dim if i > 0 else 0
            self.group_linears.append(GroupLinear(dim1=int(inp_shapes[i]),
                                                  dim2=dim2,
                                                  out_dim=int(out_shapes[i]),
                                                  groups=groups[i],
                                                  shuffle=True))

    def forward(self, x):
        # add dummy sequence dim
        if len(x.shape) == 2:
            x = x.view(x.shape[0], 1, x.shape[1])

        batch = x.shape[0]
        seq = x.shape[1]
        Inp = x

        x = self.group_linears[0](x)

        for i in range(1, len(self.group_linears)):
            g = self.groups[i]

            # input mixer 
            x = x.view((batch, seq, g, x.shape[-1] // g))  # batch, seq, dim -> batch, seq, g, dim//g
            Inp = Inp.view((batch, seq, g, self.dim // g))  # batch, seq, inp_dim -> batch, seq, g, inp_dim//g

            x = torch.cat([x, Inp], dim=-1)  # -> batch, seq, g, inp_dim//g + dim//g
            x = x.view((batch, seq, -1))  # batch, seq, g, inp_dim//g + dim//g -> batch, seq, dim + inp_dim (shuffled)
            x = self.group_linears[i](x)  # -> batch, seq, out_dim

        return x


class Net(pl.LightningModule):
    def __init__(self, in_size, glt_inp_size, N, wm, glt_out_size, hd_size, n_classes, lr_func):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = Linear(in_size, glt_inp_size)
        xavier_uniform_(self.l1.weight.data, gain=1.0)
        zeros_(self.l1.bias.data)
        self.GLT = GLT(dim=glt_inp_size, N=N, wm=wm, out_dim=glt_out_size)
        self.l2 = Linear(glt_out_size, hd_size)
        kaiming_uniform_(self.l2.weight.data)
        zeros_(self.l2.bias.data)
        self.ln2 = LayerNorm(hd_size)
        self.l3 = Linear(hd_size, n_classes)
        xavier_uniform_(self.l3.weight.data, gain=1.0)
        zeros_(self.l3.bias.data)
        self.drop = torch.nn.Dropout(0.1)
        self.lr_func = lr_func

    def FTSwishG(self, x):
        return torch.nn.functional.relu(x) * torch.sigmoid(x * 1.702) - 0.2

    def forward(self, x):
        x = self.l1(x)
        # activation
        x = self.GLT(x)
        x = torch.squeeze(x, dim=1)
        x = self.drop(x)

        x = self.l2(x)
        x = self.ln2(x)
        x = self.FTSwishG(x)

        x = self.l3(x)
        return softmax(x, dim=1)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = torch.squeeze(y).long()
        loss = cross_entropy(logits, y)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y = torch.squeeze(y).long()
        logits = self(x)
        loss = cross_entropy(logits, y)  # BCELoss #CrossEntropyLoss не равно F.nnl_loss
        preds = torch.argmax(logits, dim=1)
        acc = balanced_accuracy_score(preds.detach().cpu().numpy(), y.detach().cpu().numpy())
        warnings.simplefilter('ignore', UserWarning)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        y = torch.squeeze(y).long()
        loss = cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = balanced_accuracy_score(preds.detach().cpu().numpy(), y.detach().cpu().numpy())
        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = LrScheduledOptimizer(AdaBelief)(self.parameters(), lr=0.0001, eps=1e-14, weight_decay=1e-4,
                                                    weight_decouple=True,
                              rectify=True, lr_func=self.lr_func)
        return [optimizer]


if __name__ == "__main__":
    data = np.load('train_array.npz')
    x_values = torch.from_numpy(data['arr_0'].astype(np.float32))
    y_values = torch.from_numpy(data['arr_1'])
    print(y_values.shape)
    del data

    train_dataset = TensorDataset(x_values, y_values)
    train_loader = DataLoader(train_dataset, batch_size=200,
                              shuffle=True, drop_last=True, pin_memory=True,
                              num_workers=12)
    del x_values
    del y_values

    data = np.load('value_array.npz')
    x_values = torch.from_numpy(data['arr_x'].astype(np.float32))
    y_values = torch.from_numpy(data['arr_y'])
    print(y_values.shape)

    del data

    value_dataset = TensorDataset(x_values, y_values)
    validation_loader = DataLoader(value_dataset, batch_size=200,
                                   shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=12)
    del y_values

    weights_path = "/home/calculations/PycharmProjects/yield_prediction/weights_dan"
    logs_path = "/home/calculations/PycharmProjects/yield_prediction/logs_dan"
    # verbose-режим детализации mode-один из 'min', 'max'. В 'min'режиме обучение остановится, когда отслеживаемое
    # количество перестанет уменьшаться, а в 'max'режиме оно остановится, когда отслеживаемое количество перестанет
    # увеличиваться.

    IN_SIZE = x_values.shape[1]  # кол-во нейронов во входном слое
    del x_values

    net = Net(in_size=IN_SIZE, glt_inp_size=512, N=4, wm=2, glt_out_size=512, hd_size=256, n_classes=2,
              lr_func=lr_wm(1))
    logger = CSVLogger(logs_path, name="net_train")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath=weights_path,
                                          filename="sample-yield-best-{epoch:02d}-{val_loss:.2f}", save_top_k=1,
                                          mode="max")

    trainer = Trainer(log_every_n_steps=5, callbacks=[checkpoint_callback], gpus=1,
                      logger=logger,
                      default_root_dir="/home/calculations/PycharmProjects/yield_prediction")

    trainer.fit(net, train_loader, validation_loader)

"""
model = Net(in_size=IN_SIZE, glt_inp_size=512, N=4, wm=2, glt_out_size=512, hd_size=256, n_classes=2, lr_func=lr_wm(1))

"""
