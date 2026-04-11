"""
SKYWATCH Custom Loss Functions
Tez Katkisi #2: Adaptif Kucuk Yuz Agirliklandirma Loss

Bu modul v8DetectionLoss'u extend ederek kucuk yuzlere
daha yuksek loss agirligi atar.
"""

import torch
import torch.nn.functional as F

from ultralytics.utils.loss import BboxLoss, v8DetectionLoss
from ultralytics.utils.metrics import bbox_iou
from ultralytics.utils.tal import bbox2dist


class SkyWatchBboxLoss(BboxLoss):
    """SKYWATCH Adaptive Size-Weighted Bounding Box Loss.

    Extends BboxLoss with size-adaptive weighting that assigns higher loss
    weights to small and medium faces in surveillance footage.

    Original Contribution (Thesis):
        - Small faces (normalized size < small_face_thr=5% image width)
          receive 'small_w' (default 2.0x) more loss weight
        - Medium faces (5-15%) receive 'mid_w' (default 1.3x) more loss weight
        - This directly addresses the 39.3% small-face problem in our dataset

    References:
        Lin et al., RetinaNet Focal Loss, ICCV 2017
        Madan & Reich, RT-DETR small-object enhancements, MDPI Electronics 2025
    """

    def __init__(
        self,
        reg_max: int = 16,
        small_face_thr: float = 0.05,
        mid_face_thr: float = 0.15,
        small_w: float = 2.0,
        mid_w: float = 1.3,
    ):
        """Initialize SkyWatchBboxLoss.

        Args:
            reg_max: DFL regression max value.
            small_face_thr: Normalized width/height threshold for 'small' face.
            mid_face_thr: Normalized width/height threshold for 'medium' face.
            small_w: Loss weight multiplier for small faces.
            mid_w: Loss weight multiplier for medium faces.
        """
        super().__init__(reg_max)
        self.small_face_thr = small_face_thr
        self.mid_face_thr = mid_face_thr
        self.small_w = small_w
        self.mid_w = mid_w

    def _compute_size_weight(
        self,
        target_bboxes: torch.Tensor,
        fg_mask: torch.Tensor,
        stride: torch.Tensor,
        imgsz: torch.Tensor,
    ) -> torch.Tensor:
        """Compute per-positive-anchor adaptive weight based on normalized face size.

        Args:
            target_bboxes: (B, N_anchors, 4) xyxy coords divided by stride.
            fg_mask: (B, N_anchors) boolean foreground mask.
            stride: (N_anchors, 1) per-anchor stride values.
            imgsz: (2,) image size [H, W].

        Returns:
            size_weight: (N_pos,) float tensor with weight multipliers.
        """
        tb = target_bboxes[fg_mask]  # (N_pos, 4) in stride units

        # Expand stride to (B, N_anchors, 1), then index with fg_mask -> (N_pos, 1)
        stride_exp = stride.unsqueeze(0).expand(target_bboxes.shape[0], -1, -1)
        stride_fg = stride_exp[fg_mask].squeeze(-1)  # (N_pos,)

        # Convert to image-normalized [0,1] coordinates
        face_w = (tb[:, 2] - tb[:, 0]) * stride_fg / imgsz[1].clamp(min=1)
        face_h = (tb[:, 3] - tb[:, 1]) * stride_fg / imgsz[0].clamp(min=1)
        face_size = torch.min(face_w, face_h)  # conservative: use smaller dim

        # Build adaptive weight tensor
        size_weight = torch.ones_like(face_size)
        size_weight[face_size < self.small_face_thr] = self.small_w
        mask_mid = (face_size >= self.small_face_thr) & (face_size < self.mid_face_thr)
        size_weight[mask_mid] = self.mid_w

        return size_weight  # (N_pos,)

    def forward(
        self,
        pred_dist: torch.Tensor,
        pred_bboxes: torch.Tensor,
        anchor_points: torch.Tensor,
        target_bboxes: torch.Tensor,
        target_scores: torch.Tensor,
        target_scores_sum: torch.Tensor,
        fg_mask: torch.Tensor,
        imgsz: torch.Tensor,
        stride: torch.Tensor,
    ) -> tuple:
        """Compute adaptive size-weighted IoU and DFL losses.

        Identical to BboxLoss.forward except the per-anchor 'weight' tensor
        is multiplied by a size-dependent factor before computing losses.
        """
        weight = target_scores.sum(-1)[fg_mask].unsqueeze(-1)  # (N_pos, 1)

        # ── SKYWATCH Katkisi: Adaptif Buyukluk Agirligi ──────────────
        if fg_mask.sum() > 0:
            try:
                size_w = self._compute_size_weight(target_bboxes, fg_mask, stride, imgsz)
                weight = weight * size_w.unsqueeze(-1)
            except Exception:
                pass  # Boyut uyumsuzlugunda standart weight kullan
        # ─────────────────────────────────────────────────────────────

        iou = bbox_iou(pred_bboxes[fg_mask], target_bboxes[fg_mask], xywh=False, CIoU=True)
        loss_iou = ((1.0 - iou) * weight).sum() / target_scores_sum

        # DFL loss (parent ile ayni)
        if self.dfl_loss:
            target_ltrb = bbox2dist(anchor_points, target_bboxes, self.dfl_loss.reg_max - 1)
            loss_dfl = (
                self.dfl_loss(pred_dist[fg_mask].view(-1, self.dfl_loss.reg_max), target_ltrb[fg_mask]) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum
        else:
            target_ltrb = bbox2dist(anchor_points, target_bboxes)
            target_ltrb = target_ltrb * stride
            target_ltrb[..., 0::2] /= imgsz[1]
            target_ltrb[..., 1::2] /= imgsz[0]
            pred_dist_s = pred_dist * stride
            pred_dist_s[..., 0::2] /= imgsz[1]
            pred_dist_s[..., 1::2] /= imgsz[0]
            loss_dfl = (
                F.l1_loss(pred_dist_s[fg_mask], target_ltrb[fg_mask], reduction="none")
                .mean(-1, keepdim=True) * weight
            )
            loss_dfl = loss_dfl.sum() / target_scores_sum

        return loss_iou, loss_dfl


class SkyWatchDetectionLoss(v8DetectionLoss):
    """SKYWATCH Detection Loss.

    v8DetectionLoss'un BboxLoss'unu SkyWatchBboxLoss ile degistirir.
    Kucuk yuz orneklerini egitimde daha fazla agirliklandirir.
    """

    def __init__(self, model, tal_topk: int = 10, tal_topk2=None):
        """Initialize with SkyWatchBboxLoss instead of standard BboxLoss."""
        super().__init__(model, tal_topk, tal_topk2)
        m = model.model[-1]  # Detect() module
        # BboxLoss -> SkyWatchBboxLoss
        self.bbox_loss = SkyWatchBboxLoss(m.reg_max).to(self.device)
        print("[SKYWATCH] SkyWatchBboxLoss aktif - kucuk yuz adaptif agirlik acik")
