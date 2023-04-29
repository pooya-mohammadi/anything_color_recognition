from os.path import join
import torch
import pytorch_lightning as pl
from deep_utils import crawl_directory_dataset, dump_pickle, mkdir_incremental, BlocksTorch, TorchUtils
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from dataset import VehicleDataset
from deep_utils import ColorRecognitionCNNTorch
from settings import Config
from torch import nn
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from torch.utils.tensorboard import SummaryWriter


class LitModel(pl.LightningModule):
    def __init__(self):
        super(LitModel, self).__init__()
        self.lr = Config.train_lr
        self.model = ColorRecognitionCNNTorch(n_classes=Config.n_classes)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        logit = self.model(x)
        return logit

    def training_step(self, batch, batch_idx):
        acc, loss, bs, preds, labels = self.get_loss_acc(batch)
        return {"acc": acc, "loss": loss, "bs": bs, "preds": preds, "labels": labels}

    def test_step(self, batch, batch_idx):
        acc, loss, bs, preds, labels = self.get_loss_acc(batch)
        return {"acc": acc, "loss": loss, "bs": bs, "preds": preds, "labels": labels}

    def test_epoch_end(self, outputs) -> None:
        acc, f1_value, loss = self.calculate_metrics(outputs)
        self.log("test_f1_score", f1_value)
        self.log("test_loss", loss.item())
        self.log("test_acc", acc)

    def validation_epoch_end(self, outputs) -> None:
        acc, f1_value, loss = self.calculate_metrics(outputs)
        self.log("val_f1_score", f1_value)
        self.log("val_loss", loss.item())
        self.log("val_acc", acc)

    @staticmethod
    def calculate_metrics(outputs):
        labels, preds = [], []
        r_acc, r_loss, size = 0, 0, 0
        for row in outputs:
            r_acc += row["acc"]
            r_loss += row["loss"]
            size += row["bs"]
            preds.extend(row['preds'])
            labels.extend(row["labels"])
        f1_value = f1_score(labels, preds, average="weighted")
        loss = r_loss / size
        acc = r_acc / size
        return acc, f1_value, loss

    def training_epoch_end(self, outputs) -> None:
        acc, f1_value, loss = self.calculate_metrics(outputs)
        self.log("train_f1_score", f1_value)
        self.log("train_loss", loss.item())
        self.log("train_acc", acc, )

    def validation_step(self, batch, batch_idx):
        acc, loss, bs, preds, labels = self.get_loss_acc(batch)
        return {"acc": acc, "loss": loss, "bs": bs, "preds": preds, "labels": labels}

    def get_loss_acc(self, batch):
        images, labels = batch
        batch_size = images.size(0)
        logits = self.model(images)
        loss = self.criterion(logits, labels) * batch_size
        _, preds = torch.max(logits, 1)
        corrects = torch.sum(preds == labels.data)
        return corrects.item(), loss, batch_size, preds.cpu().numpy().tolist(), labels.cpu().numpy().tolist()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=Config.lr_reduce_factor,
                                      patience=Config.lr_patience, verbose=True)
        return {"optimizer": optimizer, "lr_scheduler": scheduler, "monitor": "val_loss"}

    @staticmethod
    def get_loaders(output_dir=None):
        x, y = crawl_directory_dataset(Config.dataset_dir)
        x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                          test_size=Config.validation_size,
                                                          stratify=y)
        class_to_id = {v: k for k, v in enumerate(set(y_train))}
        id_to_class = {v: k for k, v in class_to_id.items()}
        if output_dir:
            dump_pickle(join(output_dir, 'labels_map.pkl'), id_to_class)
        train_dataset = VehicleDataset(x_train, y_train, transform=Config.train_transform, class_to_id=class_to_id)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_size=Config.batch_size,
                                                   shuffle=True,
                                                   num_workers=Config.n_workers,
                                                   )

        val_dataset = VehicleDataset(x_val, y_val, transform=Config.val_transform, class_to_id=class_to_id)
        val_loader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_size=Config.batch_size,
                                                 shuffle=False,
                                                 num_workers=Config.n_workers,
                                                 )

        return train_loader, val_loader


def main():
    output_dir = mkdir_incremental(Config.output_dir)
    model_checkpoint = ModelCheckpoint(dirpath=output_dir,
                                       filename=Config.file_name,
                                       monitor="val_loss",
                                       verbose=True)
    learning_rate_monitor = LearningRateMonitor(logging_interval="epoch")
    trainer = pl.Trainer(gpus=1 if Config.device == "cuda" else 0,
                         max_epochs=Config.train_epochs,
                         min_epochs=Config.train_epochs // 10,
                         callbacks=[model_checkpoint, learning_rate_monitor],
                         default_root_dir=output_dir)
    lit_model = LitModel()
    lit_model.model.apply(BlocksTorch.weights_init)
    train_loader, val_loader = lit_model.get_loaders(output_dir)
    print("[INFO] Training the model")
    trainer.fit(model=lit_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    weight_path = join(output_dir, "best.ckpt")
    TorchUtils.save_config_to_weight(weight_path, Config)
    print("[INFO] Testing the model")
    trainer.test(lit_model, ckpt_path="best", dataloaders=val_loader)
    trainer.test(lit_model, ckpt_path="best", dataloaders=train_loader)


if __name__ == '__main__':
    main()
