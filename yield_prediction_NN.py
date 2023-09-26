from sklearn.metrics import balanced_accuracy_score
from torch.nn import Linear, LeakyReLU
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.init import kaiming_uniform_
from torch.nn.functional import cross_entropy
from torch.nn.functional import softmax
from torch.optim.lr_scheduler import StepLR
import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, LightningModule
from pytorch_lightning.loggers import CSVLogger

pl.seed_everything(42)
class Net(pl.LightningModule):
    def __init__(self, in_size, hd_size, n_classes=2, learning_rate=0.002):
        super().__init__()
        self.save_hyperparameters()
        self.l1 = Linear(in_size, hd_size)
        self.l3 = Linear(hd_size, n_classes)
        self.l2 = Linear(hd_size, hd_size)
        self.af = LeakyReLU()
        self.learning_rate = learning_rate

        kaiming_uniform_(self.l1.weight.data)
        kaiming_uniform_(self.l2.weight.data)

    def forward(self, x):
        x = self.l1(x)
        x = self.af(x)
        x = self.l2(x)
        x = self.af(x)
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
        # preds = torch.squeeze(preds)
        acc = balanced_accuracy_score(preds.detach().cpu().numpy(), y.detach().cpu().numpy())
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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = StepLR(optimizer, step_size=5, gamma=.1)
        return [optimizer], [scheduler]


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

    weights_path = "/home/calculations/PycharmProjects/yield_prediction/weights"
    logs_path = "/home/calculations/PycharmProjects/yield_prediction/logs"
    # verbose-режим детализации mode-один из 'min', 'max'. В 'min'режиме обучение остановится, когда отслеживаемое
    # количество перестанет уменьшаться, а в 'max'режиме оно остановится, когда отслеживаемое количество перестанет
    # увеличиваться.

    IN_SIZE = x_values.shape[1]  # кол-во нейронов во входном слое
    del x_values

    HIDDEN_SIZE = 256  # кол-во нейронов в скрытом слое
    OUTPUT_SIZE = 2  # кол-во нейронов в выходном слое
    LEARNING_RATE = .002  # скорость обучения

    net = Net(IN_SIZE, HIDDEN_SIZE, OUTPUT_SIZE, LEARNING_RATE)
    logger = CSVLogger(logs_path, name="net_train")
    checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath=weights_path,
                                          filename="sample-yield-best-{epoch:02d}-{val_loss:.2f}", save_top_k=1,
                                          mode="max")

    early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=50, verbose=True, mode="max")
    trainer = Trainer(log_every_n_steps=5, callbacks=[early_stop_callback, checkpoint_callback], gpus=1, logger=logger,
                      default_root_dir="/home/calculations/PycharmProjects/yield_prediction")

    trainer.fit(net, train_loader, validation_loader)
