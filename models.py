from typing import Dict, Optional, List, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl
from transformers import ViTModel

class SiameseNetwork(pl.LightningModule):
    def __init__(self):
        super().__init__()
        
        self.model = ViTModel.from_pretrained("google/vit-base-patch16-224")
        # self.loss = TripletLoss()
        
    def forward(self, anchor: Dict, pos: Dict, neg: Dict) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        
        anchor_feats = self.model(**anchor).pooler_output
        pos_feats = self.model(**pos).pooler_output
        neg_feats = self.model(**neg).pooler_output
        
        return anchor_feats, pos_feats, neg_feats
    
    def _compute_loss(self, anchor_feats: torch.Tensor, pos_feats: torch.Tensor, neg_feats: torch.Tensor) -> torch.Tensor:
        return F.triplet_margin_with_distance_loss(anchor_feats, pos_feats, neg_feats)
#         return self.loss(anchor_feats, pos_feats, neg_feats)
    
    def training_step(self, batch, batch_idx) -> Dict:
        
        anchor, pos, neg = batch["anchor"], batch["pos"], batch["neg"]
        
        anchor["pixel_values"] = anchor["pixel_values"].squeeze()
        pos["pixel_values"] = pos["pixel_values"].squeeze()
        neg["pixel_values"] = neg["pixel_values"].squeeze()
        
        anchor_feats, pos_feats, neg_feats = self(anchor, pos, neg)
        loss = self._compute_loss(anchor_feats, pos_feats, neg_feats)
        self.log("train_loss", loss.item(), on_step=True, on_epoch=False, prog_bar=True, logger=True)
        
        return {
            "loss": loss
        }
        
    def validation_step(self, batch, batch_idx) -> Dict:
        anchor, pos, neg = batch["anchor"], batch["pos"], batch["neg"]
        
        anchor["pixel_values"] = anchor["pixel_values"].squeeze()
        pos["pixel_values"] = pos["pixel_values"].squeeze()
        neg["pixel_values"] = neg["pixel_values"].squeeze()
        
        anchor_feats, pos_feats, neg_feats = self(anchor, pos, neg)
        loss = self._compute_loss(anchor_feats, pos_feats, neg_feats)
        self.log("val_loss", loss.item(), on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {
            "val_loss": loss
        }
        
    
    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-5, weight_decay=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 100, eta_min=1e-6)
        
        return {
            "optimizer": optimizer,
            "scheduler": lr_scheduler
        }