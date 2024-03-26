import math
import sys
import time
import torch
import utils
import metric
import dataset


def train_one_epoch(model, optimizer, criterion, data_loader, device, epoch, print_freq):
    model.train()
    is_ssd = model.__class__.__name__ == "SSD300"
    is_retina = model.__class__.__name__ == "ResNet"
    is_fan = hasattr(model, "levelattentionLoss")
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    losses = []

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        warmup_iters = min(1000, len(data_loader) - 1)
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for images, boxes, labels in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        boxes = list(box.to(device) for box in boxes)
        labels = list(label.to(device) for label in labels)

        if is_ssd:
            images = torch.stack(images, dim=0)
            predicted_locs, predicted_scores = model(images)  # (N, 8732, 4), (N, 8732, n_classes)
            loss = criterion(predicted_locs, predicted_scores, boxes, labels)  # scalar

        elif is_fan:
            images = torch.stack(images, dim=0)
            annot = [torch.cat((box, label.to(torch.float32).unsqueeze(1)), 1) for box, label in zip(boxes, labels)]
            classification_loss, regression_loss, mask_loss = model((images, annot))
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            mask_loss = mask_loss.mean()
            loss = classification_loss + regression_loss + mask_loss

        elif is_retina:
            images = torch.stack(images, dim=0)
            annot = [torch.cat((box, label.to(torch.float32).unsqueeze(1)), 1) for box, label in zip(boxes, labels)]
            classification_loss, regression_loss = model((images, annot))
            classification_loss = classification_loss.mean()
            regression_loss = regression_loss.mean()
            loss = classification_loss + regression_loss

        else:  # fasterrcnn
            targets = [{"boxes": box, "labels": label} for box, label in zip(boxes, labels)]
            loss_dict = model(images, targets)
            loss = sum(loss for loss in loss_dict.values())

        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        losses.append(loss_value)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1)
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        # metric_logger.update(loss_cls=classification_loss,loss_reg=regression_loss,loss=loss)
        metric_logger.update(loss=loss)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    return losses


@torch.no_grad()
def evaluate(model, data_loader_val, device, min_score=0.2, max_overlap=0.45, top_k=200, return_eval=False):
    is_ssd = model.__class__.__name__ == "SSD300"
    is_retina = model.__class__.__name__ == "ResNet"
    model.eval()
    det_boxes, det_labels, det_scores, true_boxes, true_labels = [], [], [], [], []
    start = time.time()

    for i, (img, box, label) in enumerate(data_loader_val):
        img, box, label = img[0], box[0], label[0]
        if i % 100 == 0:
            print(i, end=" ")
        if is_ssd:
            predicted_locs, predicted_scores = model(img.to(device).unsqueeze(0))
            det_boxes_batch, det_labels_batch, det_scores_batch = model.detect_objects(
                predicted_locs, predicted_scores, min_score=min_score, max_overlap=max_overlap, top_k=top_k
            )
            det_boxes.extend(det_boxes_batch)
            det_labels.extend(det_labels_batch)
            det_scores.extend(det_scores_batch)
            true_boxes.append(box.to(device))
            true_labels.append(label.to(device))

        elif is_retina:
            scores, labels, boxes = model(img.to(device).unsqueeze(0))
            det_boxes.append(boxes.to(device))
            det_labels.append(labels.to(device) + 1)
            det_scores.append(scores.to(device))
            true_boxes.append(box.to(device))
            true_labels.append(label.to(device) + 1)
        else:
            pred = model([img.to(device)])[0]
            det_labels.append(pred["labels"])
            det_boxes.append(pred["boxes"])
            det_scores.append(pred["scores"])
            true_boxes.append(box.to(device))
            true_labels.append(label.to(device))

    print(f"fps: {len(data_loader_val)/(time.time()-start)}")

    if return_eval:
        return det_boxes, det_labels, det_scores, true_boxes, true_labels
    else:
        ap = metric.calculate_mAP(det_boxes, det_labels, det_scores, true_boxes, true_labels)
        print()
        print("AP@.5: ", ap[0, :])
        print("AP@.7: ", ap[4, :])
        print("AP@.9: ", ap[8, :])
        print("AP@[.5:.95]: ", ap.mean(0))
        return ap


from torchvision.transforms import functional as FT
from PIL import Image, ImageDraw, ImageFont


@torch.no_grad()
def detect(model, img, device):

    model.eval()
    x = FT.to_tensor(img)
    pred = model([x.to(device)])[0]

    draw = ImageDraw.Draw(img, "RGBA")

    for i in range(len(pred["labels"])):
        if pred["scores"][i] > 0.5:
            xmin, ymin, xmax, ymax = pred["boxes"][i]
            label = pred["labels"][i]
            color = "red" if label == 1 else "green"
            draw.rectangle(((xmin, ymin), (xmax, ymax)), fill="#00000000", outline=color)
            text = ("face " if label == 1 else "face_mask ") + str(pred["scores"][i].item())[:6]
            draw.text((xmin, ymin), text)

    return img
