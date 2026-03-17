import torch

class AnchorGenerator:

    def __init__(self, sizes, aspect_ratios, strides, scales):

        self.sizes = sizes
        self.strides = strides
        self.aspect_ratios = aspect_ratios
        self.scales = scales

    def generate_anchors(self, size, ratios, device):

        anchors = []

        for scale in self.scales:

            size = size * scale

            for ratio in ratios:
                w = size * (1 / ratio) ** 0.5
                h = size * (ratio) ** 0.5

                anchors.append([-w/2, -h/2, w/2, h/2])

        return torch.tensor(anchors, device=device)
    
    def grid_anchors(self, fmap_size, stride, base_anchors, device):

        H, W = fmap_size

        shifts_x = torch.arange(W, device=device) * stride
        shifts_y = torch.arange(H, device=device) * stride

        shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)

        shifts = torch.stack(
            (shift_x, shift_y, shift_x, shift_y),
            dim=-1
        ).reshape(-1, 4)

        anchors = base_anchors[None, :, :] + shifts[:, None, :]

        anchors = anchors.reshape(-1, 4)

        return anchors

    def __call__(self, feature_maps):
        device = feature_maps[0].device

        anchors_all = []

        for i, fmap in enumerate(feature_maps):

            _, _, H, W = fmap.shape

            size = self.sizes[i]
            stride = self.strides[i]

            base_anchors = self.generate_anchors(size, self.aspect_ratios, device)

            anchors = self.grid_anchors((H, W), stride, base_anchors, device=device)

            anchors_all.append(anchors)

        return torch.cat(anchors_all, dim=0)


