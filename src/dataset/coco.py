import torch
from torchvision.datasets import CocoDetection
import torchvision.transforms.functional as F
import random

class DetectionTransform:

    def __init__(self, config):

        self.sizes = config['MIN_SIZES']
        self.mean = [0.485, 0.456, 0.406]
        self.std = [0.229, 0.224, 0.225]

        self.hflip_prob = config['HFLIP_PROB']

    def resize(self, image, boxes):

        size = random.choice(self.sizes)

        w, h = image.size
        scale = size / min(h, w)

        new_w = int(w*scale)
        new_h = int(h*scale)

        image = F.resize(image, (new_h, new_w))

        boxes = boxes * torch.tensor(
            [scale, scale, scale, scale]
        )

        return image, boxes

    def random_flip(self, image, boxes):

        if random.random() < self.hflip_prob:
            w, _ = image.size

            image = F.hflip(image)

            boxes[:, [0, 2]] = w - boxes[:, [2, 0]]

        return image, boxes
    
    def __call__(self, image, boxes):

        image, boxes = self.resize(image, boxes)

        image, boxes = self.random_flip(image, boxes)

        image = F.to_tensor(image)

        image = F.normalize(image, self.mean, self.std)

        return image, boxes
    

class COCODataset(CocoDetection):

    def __init__(self, img_dir, ann_file, cfg_transforms):
        super().__init__(img_dir, ann_file)

        self.transform = DetectionTransform(cfg_transforms)

        self.cat_ids = sorted(self.coco.getCatIds())

        self.cat_id_to_idx = {
            cat_id: i for i, cat_id in enumerate(self.cat_ids)
        }

    def __getitem__(self, idx):

        img, anns = super().__getitem__(idx)

        boxes = []
        labels = []

        for obj in anns:

            x,y,w,h = obj["bbox"]

            boxes.append([x, y, x+w, y+h])

            labels.append(self.cat_id_to_idx[obj["category_id"]])

        if len(boxes) == 0:

            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.long)

        else:

            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.long)

        img, boxes = self.transform(img, boxes)

        target = {
            "boxes": boxes,
            "labels": labels
        }

        return img, target
    