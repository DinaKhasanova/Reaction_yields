import numpy as np
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import TensorDataset, DataLoader

from yield_model_with_fixes import Net, lr_wm

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
    IN_SIZE = x_values.shape[1]
    del x_values

    net = Net(in_size=IN_SIZE, glt_inp_size=512, N=4, wm=2, glt_out_size=512, hd_size=256, n_classes=2,
              lr_func=lr_wm(1))
    net = net.load_from_checkpoint(
        checkpoint_path="/home/calculations/PycharmProjects/yield_prediction/weights_dan/sample-yield-best-"
                        "epoch=250-val_loss=0.43.ckpt",
        hparams_file="/home/calculations/PycharmProjects/yield_prediction/logs_dan/net_train/version_1/hparams.yaml",
        map_location=None,
    )


    logs_path = "/home/calculations/PycharmProjects/yield_prediction/logs_dan"
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