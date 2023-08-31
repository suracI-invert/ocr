from typing import Any, Dict, List, Tuple, Union

import torch
from lightning import LightningModule
from torchmetrics.text.cer import CharErrorRate
from torchmetrics import MinMetric, MeanMetric

from src.utils.loss import LabelSmoothingLoss
from src.models.tokenizer import Tokenizer

class OCRLitModule(LightningModule):
    def __init__(
        self,
        net: torch.nn.Module,
        tokenizer: Tokenizer,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler,
        optimizer_params: Dict[str, Any] = {},
        scheduler_params: Dict[str, Any] = {},
        learning_rate: float = 0.1,
        monitor_metric: str = 'val_loss',
        interval: str = 'epoch',
        frequency: int = 5
    ) -> None:
        super().__init__()
        self.save_hyperparameters(logger= True, ignore= ['net', 'tokenizer'])
        self.net = net
        self.tokenizer = tokenizer
        self.criterion = LabelSmoothingLoss(len(self.tokenizer), self.tokenizer.pad, smoothing= 0.1)

        self.val_cer = CharErrorRate()

        self.train_loss = MeanMetric()
        self.val_loss = MeanMetric()

        self.val_cer_best = MinMetric()

    def forward(self, img: torch.Tensor, tgt_input: torch.Tensor, tgt_padding_mask: torch.Tensor) -> torch.Tensor:
        return self.net(img, tgt_input, tgt_padding_mask)
    
    def on_train_start(self) -> None:
        self.val_loss.reset()
        self.val_cer.reset()
        self.val_cer_best.reset()
    
    def model_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        img, tgt_input, tgt_padding_mask, tgt_output = batch['img'], batch['tgt_input'], batch['tgt_padding_mask'], batch['tgt_output']

        outputs = self.forward(img, tgt_input, tgt_padding_mask)
        outputs = outputs.view(-1, outputs.size(2))
        tgt_output = tgt_output.view(-1)

        loss = self.criterion(outputs, tgt_output)

        return loss
    
    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)

        self.train_loss(loss)

        # log
        self.log('train_loss', self.train_loss, on_step= True, prog_bar= True)

        return loss
    
    def on_train_epoch_end(self) -> None:
        prototype_input = torch.rand((32, 3, 70, 140))
        self.logger.experiment.log_graph(self.net, prototype_input)
        for name, params in self.named_parameters():
            self.logger.experiment.add_histogram(name, params, self.current_epoch)


    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss = self.model_step(batch)
        pred_ids, prob = self.net.predict(batch['img'])
        pred = self.tokenizer.batch_decode(pred_ids)
        target = self.tokenizer.batch_decode(batch['tgt_output'])

        self.val_loss(loss)
        self.val_cer(pred, target)

        self.log('val_loss', self.val_loss, on_step= True, prog_bar= False)
        self.log('val_cer', self.val_cer, on_step= False, on_epoch= True, prog_bar= False)
    
    def on_validation_epoch_end(self) -> None:
        cer = self.val_cer.compute()
        self.val_cer_best(cer)

        self.log('val_best_cer', self.val_cer_best.compute(), sync_dist= True, prog_bar= False)

    def predict_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> List[Dict[str, Union[str, torch.Tensor]]]:
        filenames = batch['filename']
        pred_ids, prob = self.net.predict(batch['img'])
        pred = self.tokenizer.batch_decode(pred_ids)
        res = []
        for i in range(len(filenames)):
            res.append({
                'filename': filenames[i],
                'prediction': pred[i],
                'probability': prob[i]
            })
        return res

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params= self.parameters(), lr= self.hparams.learning_rate, **self.hparams.optimizer_params)
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer= optimizer, **self.hparams.scheduler_params)
            return {
                'optimizer': optimizer,
                'lr_scheduler': {
                    'scheduler': scheduler,
                    'monitor': self.hparams.monitor_metric,
                    'interval': self.hparams.interval,
                    'frequency': self.hparams.frequency
                }
            }
        return {'optimizer': optimizer}
