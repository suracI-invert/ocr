from typing import Any, Dict, List, Tuple, Union

import torch
from lightning import LightningModule
from torchmetrics import MeanMetric, MinMetric
from torchmetrics.text.cer import CharErrorRate

from src.models.tokenizer import Tokenizer
from src.utils.loss import LabelSmoothingLoss


class OCRLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        tokenizer: Tokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger=True, ignore=["net", "tokenizer"])
        self.net = net
        self.tokenizer = tokenizer
        self.criterion = LabelSmoothingLoss(len(self.tokenizer), self.tokenizer.pad, smoothing=0.1)

        self.train_cer = CharErrorRate()
        self.val_cer = CharErrorRate()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_cer_best = MinMetric()

    def forward(
        self, img: torch.Tensor, tgt_input: torch.Tensor, tgt_padding_mask: torch.Tensor
    ) -> torch.Tensor:
        return self.net(img, tgt_input, tgt_padding_mask)

    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_cer.reset()
        self.val_cer_best.reset()

    def model_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        img, tgt_input, tgt_padding_mask, tgt_output = (
            batch["img"],
            batch["tgt_input"],
            batch["tgt_padding_mask"],
            batch["tgt_output"],
        )

        outputs = self.forward(img, tgt_input, tgt_padding_mask)
        outputs = outputs.view(-1, outputs.size(2))
        tgt_output = tgt_output.view(-1)

        loss = self.criterion(outputs, tgt_output)

        return loss

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)

        self.train_loss(loss)

        # log
        self.log("train_loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cer", self.train_cer, on_step=False, on_epoch=True, prog_bar=True)

        return loss

    def on_train_epoch_end(self) -> None:
        # dont understand the purpose of these?
        # for name, params in self.named_parameters():
        #     self.logger.experiment.add_histogram(name, params, self.current_epoch)
        pass

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)
        pred_ids, _ = self.net.predict(batch["img"])
        pred = self.tokenizer.batch_decode(pred_ids)
        target = self.tokenizer.batch_decode(batch["tgt_output"])

        self.val_loss(loss)
        self.val_cer(pred, target)

        self.log("val_loss", self.val_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_cer", self.val_cer, on_step=False, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self) -> None:
        cer = self.val_cer.compute()
        self.val_cer_best(cer)

        self.log("val_best_cer", self.val_cer_best.compute(), sync_dist=True, prog_bar=True)

    def predict_step(
        self, batch: Dict[str, torch.Tensor], batch_idx: int
    ) -> List[Dict[str, Union[str, torch.Tensor]]]:
        filenames = batch["filename"]
        pred_ids, prob = self.net.predict(batch["img"])
        pred = self.tokenizer.batch_decode(pred_ids)
        res = []
        for i in range(len(filenames)):
            res.append({"filename": filenames[i], "prediction": pred[i], "probability": prob[i]})
        return res

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}


if __name__ == "__main__":
    import hydra
    import pyrootutils
    from omegaconf import DictConfig, OmegaConf

    # find paths
    pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
    path = pyrootutils.find_root(search_from=__file__, indicator=".project-root")

    config_path = str(path / "configs")
    print(f"project-root: {path}")
    print(f"config path: {config_path}")

    @hydra.main(version_base="1.3", config_path=config_path, config_name="train.yaml")
    def main(cfg: DictConfig):
        print(f"config: \n {OmegaConf.to_yaml(cfg.model, resolve=True)}")

        model = hydra.utils.instantiate(cfg.model)
        print(model)

    main()
