import cv2
import torch
from torchvision.ops import box_iou

def draw_boxes(img, boxes, labels, scores, threshold=0.5):
    result = img.copy()

    for bbox, label, score in zip(boxes, labels, scores):
        x1, y1, x2, y2 = bbox.int().tolist()
        if score > threshold:
            cv2.rectangle(
                result, (x1, y1), (x2, y2),
                (0, 255, 0), 2
            )

            cv2.putText(
                result, str(label), (x1, y1-10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.3, (0,255,0), 2
            )

    return result


def match_anchors(anchors, gt_boxes, pos_thresh=0.5, neg_thresh=0.4):
    device = anchors.device

    if gt_boxes.numel() == 0:

        labels = torch.zeros(
            anchors.shape[0],
            dtype=torch.long,
            device=device
        )

        gt_idx = torch.zeros(
            anchors.shape[0],
            dtype=torch.long,
            device=device
        )

        return labels, gt_idx

    iou = box_iou(anchors, gt_boxes)

    max_iou, gt_idx = iou.max(dim=1)

    labels = torch.full(
        (anchors.shape[0],), -1, dtype=torch.long, device=device
    )

    labels[max_iou < neg_thresh] = 0
    labels[max_iou >= pos_thresh] = 1

    # guarantee each gt has at least one anchor
    best_anchor_per_gt = iou.argmax(dim=0)

    labels[best_anchor_per_gt] = 1

    return labels, gt_idx

def encode_boxes(anchors, gt_boxes):
    if gt_boxes.numel() == 0:
        return torch.zeros_like(anchors)

    ax = (anchors[:,0] + anchors[:,2]) * 0.5
    ay = (anchors[:,1] + anchors[:,3]) * 0.5
    aw = anchors[:,2] - anchors[:,0]
    ah = anchors[:,3] - anchors[:,1]

    gx = (gt_boxes[:,0] + gt_boxes[:,2]) * 0.5
    gy = (gt_boxes[:,1] + gt_boxes[:,3]) * 0.5
    gw = gt_boxes[:,2] - gt_boxes[:,0]
    gh = gt_boxes[:,3] - gt_boxes[:,1]

    tx = (gx - ax) / aw
    ty = (gy - ay) / ah
    tw = torch.log(gw / aw)
    th = torch.log(gh / ah)

    targets = torch.stack((tx, ty, tw, th), dim=1)

    return targets