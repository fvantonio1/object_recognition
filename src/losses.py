import torch.nn.functional as F
import torch

def focal_loss(logits, targets, alpha=0.25, gamma=2.0):
    prob = torch.sigmoid(logits)

    ce_loss = F.binary_cross_entropy_with_logits(
        logits,
        targets,
        reduction="none"
    )

    p_t = prob * targets + (1 - prob) * (1 - targets)

    loss = ce_loss * ((1 - p_t) ** gamma)

    if alpha >= 0:
        alpha_t = alpha * targets + (1 - alpha) * (1 - targets)
        loss = alpha_t * loss

    return loss

def bbox_loss(pred_boxes, target_boxes):
    return F.smooth_l1_loss(
        pred_boxes,
        target_boxes,
        reduction='none',
        beta=1/9
    )

def compute_losses(cls_logits, bbox_preds, box_labels, cls_labels, box_targets, num_classes):
    num_pos = (box_labels == 1).sum()

    valid_mask = box_labels >= 0
    pos_mask = box_labels == 1

    cls_targets = torch.zeros(
        cls_logits.shape[0],
        num_classes,
        device=cls_logits.device
    )

    cls_targets[pos_mask, cls_labels[pos_mask]] = 1

    cls_loss = focal_loss(
        cls_logits[valid_mask],
        cls_targets[valid_mask]
    )

    cls_loss = cls_loss.sum() / max(1, num_pos)

    box_loss = bbox_loss(
        bbox_preds[pos_mask],
        box_targets[pos_mask],
    )

    box_loss = box_loss.sum() / max(1, num_pos)

    return cls_loss, box_loss