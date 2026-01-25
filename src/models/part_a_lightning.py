import torch
import pytorch_lightning as pl
from transformers import get_linear_schedule_with_warmup

class Seq2SeqLightningModule(pl.LightningModule):
    def __init__(self, model, lr=2e-5, weight_decay=0.01, warmup_ratio=0.03):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.warmup_ratio = warmup_ratio

    def on_fit_start(self):
        # Ensure training mode early (fixes "modules in eval mode" warning)
        self.model.train()

    def on_train_epoch_start(self):
        self.model.train()

    def forward(self, **batch):
        return self.model(**batch)

    def training_step(self, batch, batch_idx):
        self.model.train()
        out = self.model(**batch)
        loss = out.loss
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self.model.eval()
        out = self.model(**batch)
        loss = out.loss
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        params = list(self.model.named_parameters())
        grouped = [
            {
                "params": [p for n, p in params if not any(nd in n for nd in no_decay)],
                "weight_decay": self.weight_decay,
            },
            {
                "params": [p for n, p in params if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        optimizer = torch.optim.AdamW(grouped, lr=self.lr)

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * self.warmup_ratio)

        scheduler = get_linear_schedule_with_warmup(
            optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_steps
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "interval": "step"},
        }
