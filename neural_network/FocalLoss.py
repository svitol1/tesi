import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    """
    Focal Loss per classificazione multi-classe sbilanciata.

    FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    Args:
        alpha (Tensor | None): pesi per classe, shape [C].
                               Calcolati automaticamente da compute_alpha().
                               Se None, tutte le classi hanno peso 1.
        gamma (float):         focusing parameter. 0 = Cross-Entropy classica.
                               Valori consigliati: 1.0 – 3.0 (default: 2.0).
        reduction (str):       'mean' | 'sum' | 'none'
    """
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        # register_buffer: il tensore segue il modello su GPU con .to(device)
        if alpha is not None:
            self.register_buffer("alpha", alpha.float())
        else:
            self.alpha = None

    def forward(self, logits, targets):
        """
        logits  : (B, C) — output grezzo della rete (PRIMA di softmax)
        targets : (B,)   — indici interi delle classi
        """
        # Cross-entropy per ogni campione, senza riduzione
        ce_loss = F.cross_entropy(
            logits, targets,
            weight=self.alpha,   # pesa già le classi rare
            reduction="none"     # shape (B,)
        )

        # p_t = probabilità assegnata alla classe corretta
        # exp(-ce) è numericamente stabile perché ce >= 0
        pt = torch.exp(-ce_loss)                    # shape (B,)

        # Modulazione: abbatte la loss sui campioni "facili" (pt alto)
        focal_weight = (1.0 - pt) ** self.gamma     # shape (B,)

        loss = focal_weight * ce_loss               # shape (B,)

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss  # 'none'
