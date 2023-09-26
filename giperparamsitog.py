from torch.nn.functional import cross_entropy
from torch.nn.functional import softmax
from sklearn.metrics import balanced_accuracy_score
import torch
from itertools import product
import pytorch_lightning as pl
from torch import nn
from torch.nn import LeakyReLU, Dropout
from torch.optim.lr_scheduler import StepLR
from yield_prediction_NN import Trainer
from torch.nn.init import kaiming_uniform_
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from pytorch_lightning.loggers import CSVLogger
import warnings
from time import time

pl.seed_everything(42)
class Net(pl.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.save_hyperparameters()
        self.layers = nn.ModuleList()
        self.input_size = config['input_size']
        self.lr = config['lr']
        self.weight_decay = config['weight_decay']
        input_size = config['input_size']
        for num_l, (size, activation) in enumerate(config['layers_data']):
            self.layers.append(nn.Linear(input_size, size))
            input_size = size
            if activation is not None:
                self.layers.append(activation)
            if num_l == (len(config['layers_data']) - 2):
                self.layers.append(Dropout(p=config['dropout']))
        for layer in self.layers:
            if not isinstance(layer, LeakyReLU) and not isinstance(layer, Dropout):
                kaiming_uniform_(layer.weight.data)

    def forward(self, input_data):
        for layer in self.layers:
            input_data = layer(input_data)
        return softmax(input_data, dim=1)

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
        loss = cross_entropy(logits, y)
        preds = torch.argmax(logits, dim=1)
        # try:
        acc = balanced_accuracy_score(preds.detach().cpu().numpy(), y.detach().cpu().numpy())
        warnings.simplefilter('ignore', UserWarning)
        #     warnings.simplefilter('error', UserWarning)
        # except UserWarning:
        #     print(preds)
        #     print(y)
        #     print(preds.shape, y.shape)

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
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        scheduler = StepLR(optimizer, step_size=5, gamma=.1)
        return [optimizer], [scheduler]


if __name__ == '__main__':

        weights_path_1 = "/home/calculations/PycharmProjects/yield_prediction/weights"
        logs_path_1 = "/home/calculations/PycharmProjects/yield_prediction/logs"

        OUTPUT_SIZE = 2

        layers_list = [[(256, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(512, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(1024, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(2048, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(4096, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(512, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(512, LeakyReLU()), (256, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(256, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(256, LeakyReLU()), (256, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(512, LeakyReLU()), (512, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(128, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(512, LeakyReLU()), (256, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(512, LeakyReLU()), (256, LeakyReLU()), (256, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(512, LeakyReLU()), (128, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(128, LeakyReLU()), (128, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(256, LeakyReLU()), (256, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(256, LeakyReLU()), (256, LeakyReLU()), (256, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       [(256, LeakyReLU()), (128, LeakyReLU()), (128, LeakyReLU()), (OUTPUT_SIZE, LeakyReLU())],
                       ]
        lr_list = [0.01, 0.001]
        dropout_list = [0.1, 0.2, 0.3, 0.4, 0.5]
        weight_decay_list = [0, 0.1, 0.01, 0.001]
        config_list = [layers_list, lr_list, dropout_list, weight_decay_list]


        for iii, (layers, lr, dropout, weight_decay) in enumerate(product(*config_list)):
            data = np.load('train_array.npz')
            x_values = torch.from_numpy(data['arr_0'].astype(np.float32))
            y_values = torch.from_numpy(data['arr_1'])
            print(y_values.shape)
            del data

            train_dataset = TensorDataset(x_values, y_values)
            train_loader = DataLoader(train_dataset, batch_size=200,
                                      shuffle=True, drop_last=True, pin_memory=True,
                                      num_workers=0)
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
                                           num_workers=0)
            del y_values

            print(iii)
            start_comb_time = time()
            config = {
                "input_size": x_values.shape[1],
                "layers_data": layers,
                "lr": lr,
                "num_epochs": 1000,
                "dropout": dropout,
                "weight_decay": weight_decay

            }
            with open('log.txt', 'a') as f:
                f.write(f'Comb No {iii}: {";".join([str(x) for x in config.values()])}')
            logger = CSVLogger(logs_path_1, name=f"net_train_{iii}")
            checkpoint_callback = ModelCheckpoint(monitor="val_acc", dirpath=weights_path_1,
                                                  filename="sample-yield-best-{epoch:02d}-{val_loss:.2f}"+f"-combNo{iii}", save_top_k=1,
                                                  mode="max")

            early_stop_callback = EarlyStopping(monitor="val_acc", min_delta=0.001, patience=10, verbose=True,
                                                mode="max")
            trainer = Trainer(log_every_n_steps=5, callbacks=[early_stop_callback, checkpoint_callback], gpus=1,
                              logger=logger,
                              default_root_dir="/home/calculations/PycharmProjects/yield_prediction")
            net = Net(config=config)
            trainer.fit(net, train_loader, validation_loader)
            end_comb_time = time() - start_comb_time
            with open('log.txt', 'a') as f:
                f.write(f'; time: {end_comb_time:.2f}\n')
