import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import TensorDataset, DataLoader
from torch.nn import LeakyReLU

from giperparamsitog import Net

if __name__ == '__main__':
    data = np.load('retro_array.npz')
    x_values = torch.from_numpy(data['arr_0'].astype(np.float32))
    y_values = torch.from_numpy(data['arr_1'])
    print(y_values.shape)
    del data

    test_retro_dataset = TensorDataset(x_values, y_values)
    test_retro_loader = DataLoader(test_retro_dataset, batch_size=200,
                                   shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=12)

    config = {
        "input_size": x_values.shape[1],
        "layers_data": [(512, LeakyReLU(negative_slope=0.01)),
                        (256, LeakyReLU(negative_slope=0.01)),
                        (128, LeakyReLU(negative_slope=0.01)),
                        (2, LeakyReLU(negative_slope=0.01))],
        "lr": 0.001,
        "num_epochs": 1000,
        "dropout": 0.1,
        "weight_decay": 0,

    }
    del x_values

    net = Net(config=config)
    net = net.load_from_checkpoint(
        checkpoint_path="/home/calculations/PycharmProjects/yield_prediction/weights/sample-yield-best-"
                        "epoch=20-val_loss=0.45-combNo460.ckpt",
        hparams_file="/home/calculations/PycharmProjects/yield_prediction/logs/net_train_460/version_0/hparams.yaml",
        map_location=None,
    )


    logs_path = "/home/calculations/PycharmProjects/yield_prediction/logs"
    logger = CSVLogger(logs_path, name="net_test")
    trainer = Trainer(gpus=1, logger=logger,
                      default_root_dir="/home/calculations/PycharmProjects/yield_prediction")

    trainer.test(net, test_retro_loader)


    data = np.load('prosp_array.npz')
    x_values = torch.from_numpy(data['arr_0'].astype(np.float32))
    y_values = torch.from_numpy(data['arr_1'])
    print(y_values.shape)
    del data

    test_prosp_dataset = TensorDataset(x_values, y_values)
    test_prosp_loader = DataLoader(test_prosp_dataset, batch_size=200,
                                   shuffle=True, drop_last=True, pin_memory=True,
                                   num_workers=12)
    trainer.test(net, test_prosp_loader)
