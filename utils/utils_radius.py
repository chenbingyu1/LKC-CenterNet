import torch
import torch.nn.functional as F

from utils.utils_bbox import pool_nms


def _grad_mag(images):
    gray = images.mean(dim=1, keepdim=True)
    sobel_x = torch.tensor(
        [[-1.0, 0.0, 1.0],
         [-2.0, 0.0, 2.0],
         [-1.0, 0.0, 1.0]],
        device=images.device,
        dtype=images.dtype
    ).view(1, 1, 3, 3)
    sobel_y = torch.tensor(
        [[-1.0, -2.0, -1.0],
         [0.0, 0.0, 0.0],
         [1.0, 2.0, 1.0]],
        device=images.device,
        dtype=images.dtype
    ).view(1, 1, 3, 3)

    gx = F.conv2d(gray, sobel_x, padding=1)
    gy = F.conv2d(gray, sobel_y, padding=1)
    return torch.sqrt(gx * gx + gy * gy + 1e-6)


def unsup_radius_loss(
    images,
    hm,
    offset,
    radius_map,
    confidence=0.1,
    radius_min=3.0,
    radius_max=8.0,
    edge_width=1.0,
    lambda_in=1.0,
    lambda_edge=1.0,
    lambda_prior=0.1,
):
    device = images.device
    dtype = images.dtype
    hm_nms = pool_nms(hm)

    b, c, out_h, out_w = hm_nms.shape
    in_h, in_w = images.shape[2], images.shape[3]
    stride = float(in_h) / float(out_h)

    grad = _grad_mag(images)
    if stride != 1.0:
        k = int(stride)
        grad = F.avg_pool2d(grad, kernel_size=k, stride=k)
    grad = grad.squeeze(1)

    yv, xv = torch.meshgrid(torch.arange(0, out_h, device=device), torch.arange(0, out_w, device=device))
    xv = xv.flatten().float()
    yv = yv.flatten().float()
    grid_x = xv.view(out_h, out_w)
    grid_y = yv.view(out_h, out_w)

    total_in = torch.zeros((), device=device, dtype=dtype)
    total_edge = torch.zeros((), device=device, dtype=dtype)
    total_prior = torch.zeros((), device=device, dtype=dtype)
    count = 0

    for i in range(b):
        heat_map = hm_nms[i].permute(1, 2, 0).contiguous().view(-1, c)
        pred_offset = offset[i].permute(1, 2, 0).contiguous().view(-1, 2)
        class_conf, _ = torch.max(heat_map, dim=-1)
        mask = class_conf > confidence
        if mask.sum() == 0:
            continue

        xv_mask = xv[mask] + pred_offset[mask][:, 0]
        yv_mask = yv[mask] + pred_offset[mask][:, 1]
        r_vals = radius_map[i, 0].contiguous().view(-1)[mask]
        r_vals = F.softplus(r_vals)
        r_vals = torch.clamp(r_vals, min=1.0 / stride)

        grad_map = grad[i]
        for j in range(r_vals.shape[0]):
            cx = xv_mask[j]
            cy = yv_mask[j]
            r = r_vals[j]
            dist = torch.sqrt((grid_x - cx) ** 2 + (grid_y - cy) ** 2)

            disk = dist <= r
            ring = (dist >= (r - edge_width)) & (dist <= (r + edge_width))

            if disk.any():
                total_in = total_in + grad_map[disk].mean()
            if ring.any():
                total_edge = total_edge + grad_map[ring].mean()

            r_in = r * stride
            total_prior = total_prior + F.relu(r_in - radius_max) + F.relu(radius_min - r_in)
            count += 1

    if count == 0:
        return torch.zeros((), device=device, dtype=dtype)

    mean_in = total_in / count
    mean_edge = total_edge / count
    mean_prior = total_prior / count
    return lambda_in * mean_in + lambda_edge * (-mean_edge) + lambda_prior * mean_prior
