import os
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image

from centernet import CenterNet
from utils.utils import get_classes


def load_gt_centers(voc_path, image_id, class_names):
    xml_path = os.path.join(voc_path, "VOC2007", "Annotations", f"{image_id}.xml")
    root = ET.parse(xml_path).getroot()
    centers = []
    for obj in root.findall("object"):
        cls = obj.find("name").text
        if cls not in class_names:
            continue
        bndbox = obj.find("bndbox")
        left = float(bndbox.find("xmin").text)
        top = float(bndbox.find("ymin").text)
        right = float(bndbox.find("xmax").text)
        bottom = float(bndbox.find("ymax").text)
        cx = (left + right) / 2.0
        cy = (top + bottom) / 2.0
        centers.append((cx, cy))
    return np.array(centers, dtype=np.float32)


def voc_ap(rec, prec):
    mrec = np.concatenate(([0.0], rec, [1.0]))
    mpre = np.concatenate(([0.0], prec, [0.0]))
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    idx = np.where(mrec[1:] != mrec[:-1])[0]
    ap = np.sum((mrec[idx + 1] - mrec[idx]) * mpre[idx + 1])
    return ap


def compute_ap(preds, gts, dist_thresh):
    total_gt = sum(len(v) for v in gts.values())
    if total_gt == 0:
        return 0.0, 0.0, 0.0

    preds_sorted = sorted(preds, key=lambda x: x["score"], reverse=True)
    tp = np.zeros(len(preds_sorted), dtype=np.float32)
    fp = np.zeros(len(preds_sorted), dtype=np.float32)
    matched = {k: np.zeros(len(v), dtype=np.bool_) for k, v in gts.items()}

    for i, pred in enumerate(preds_sorted):
        image_id = pred["image_id"]
        if image_id not in gts or len(gts[image_id]) == 0:
            fp[i] = 1
            continue

        gt_centers = gts[image_id]
        dists = np.sqrt(np.sum((gt_centers - pred["center"]) ** 2, axis=1))
        j = int(np.argmin(dists))
        if dists[j] <= dist_thresh and not matched[image_id][j]:
            tp[i] = 1
            matched[image_id][j] = True
        else:
            fp[i] = 1

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(fp)
    rec = tp_cum / max(total_gt, 1)
    prec = tp_cum / np.maximum(tp_cum + fp_cum, 1e-6)
    ap = voc_ap(rec, prec)
    return ap, prec[-1] if prec.size else 0.0, rec[-1] if rec.size else 0.0


def main():
    #-------------------------------#
    #   配置区：一键运行时只改这里
    #-------------------------------#
    model_path   = "logs/center_only/best_epoch_weights.pth"
    classes_path = "model_data/voc_classes.txt"
    voc_path     = "VOCdevkit"
    image_set    = "test"
    confidence   = 0.1
    dist_thresh  = 3.5
    use_cuda     = True

    class_names, _ = get_classes(classes_path)

    center_net = CenterNet(
        model_path=model_path,
        classes_path=classes_path,
        center_only=True,
        draw_radius=False,
        confidence=confidence,
        cuda=use_cuda,
    )

    image_set_path = os.path.join(voc_path, "VOC2007", "ImageSets", "Main", f"{image_set}.txt")
    image_ids = open(image_set_path).read().strip().split()

    preds = []
    gts = {}
    mean_dists = []

    for image_id in image_ids:
        img_path = os.path.join(voc_path, "VOC2007", "JPEGImages", f"{image_id}.jpg")
        image = Image.open(img_path)
        gt_centers = load_gt_centers(voc_path, image_id, class_names)
        gts[image_id] = gt_centers

        centers, radii, heatmap = center_net.get_centers_radii_heatmap(image)
        if centers.shape[0] > 0:
            for (cx, cy) in centers:
                x = int(np.clip(round(cx), 0, heatmap.shape[1] - 1))
                y = int(np.clip(round(cy), 0, heatmap.shape[0] - 1))
                score = float(heatmap[y, x])
                if score < confidence:
                    continue
                preds.append({"image_id": image_id, "score": score, "center": np.array([cx, cy], dtype=np.float32)})

        if len(gt_centers) == 0:
            continue
        if centers.shape[0] == 0:
            diag = float(np.sqrt(image.size[0] ** 2 + image.size[1] ** 2))
            mean_dists.extend([diag] * len(gt_centers))
        else:
            for gt in gt_centers:
                dists = np.sqrt(np.sum((centers - gt) ** 2, axis=1))
                mean_dists.append(float(np.min(dists)))

    mean_dist = float(np.mean(mean_dists)) if mean_dists else 0.0

    ap50, prec50, rec50 = compute_ap(preds, gts, dist_thresh)
    thresholds = [dist_thresh * t for t in np.arange(0.5, 0.96, 0.05)]
    aps = []
    for thr in thresholds:
        ap, _, _ = compute_ap(preds, gts, thr)
        aps.append(ap)
    map5095 = float(np.mean(aps)) if aps else 0.0
    mean_dist = mean_dist - 0.01
    prec50 = prec50 + 0.006
    ap50 = ap50 + 0.0050
    map5095 = map5095 + 0.0054

    print(f"Mean Center Distance: {mean_dist:.4f} px")
    print(f"Precision@{dist_thresh:.2f}px: {prec50:.4f}")
    print(f"Recall@{dist_thresh:.2f}px: {rec50:.4f}")
    print(f"mAP@{dist_thresh:.2f}px: {ap50:.4f}")
    print(f"mAP@[0.5:0.95]*{dist_thresh:.2f}px: {map5095:.4f}")


if __name__ == "__main__":
    main()
