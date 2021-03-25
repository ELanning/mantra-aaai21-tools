import os
import copy
import matplotlib.pyplot as plt
import json
import numpy as np
from collections import defaultdict
from PIL import Image


class OrderEstimator:
    def reorder_object(self, pagedata):
        if len(pagedata["frame"]) == 0 or len(pagedata["text"]) == 0:
            return pagedata
        org_panels = pagedata["frame"]
        panel_boxes = make_bbs(pagedata["frame"])
        panel_orders = PanelOrderEstimator(panel_boxes, thresh=0.2).get_panel_orders()
        pagedata["frame"] = [{} for _ in range(len(panel_boxes))]
        for bb, o, fr in zip(panel_boxes, panel_orders, org_panels):
            pagedata["frame"][o] = fr
        pagedata = self.estimate_and_correct_panelid(pagedata)
        pagedata = self.estimate_and_change_text_order(pagedata)
        return pagedata

    def estimate_and_correct_panelid(self, objects):
        """
        Correct panelID assigned to each text
        """
        assert len(objects["frame"]) > 0 and len(objects["text"]) > 0
        panel_bbs = make_bbs(objects["frame"])
        text_bbs = make_bbs(objects["text"])
        panel_ids = self.get_panelid_text_belong(panel_bbs, text_bbs)
        for t, panel_id in zip(objects["text"], panel_ids):
            t["panel_id"] = panel_id
        return objects

    def get_panelid_text_belong(self, panel_bbs, text_bbs):
        panel_bbs = np.array(panel_bbs, dtype=np.float32)
        text_bbs = np.array(text_bbs, dtype=np.float32)
        panel_ids = bbox_iou(panel_bbs, text_bbs).argmax(axis=0)
        return panel_ids.astype(np.int).tolist()

    def estimate_and_change_text_order(self, objects):
        """
        Change the order of texts using panelID and simple heuristics
        """
        texts_inv = defaultdict(list)
        for o in objects["text"]:
            texts_inv[o["panel_id"]].append(o)
        panel_ids = sorted(set([o["panel_id"] for o in objects["text"]]))
        objects_return = []
        for panel_id in panel_ids:
            if panel_id >= len(objects["frame"]):
                continue
            panel = objects["frame"][panel_id]
            sorted_texts = sorted(
                texts_inv[panel_id],
                key=lambda x: abs(
                    ((x["x"] + x["w"]) - (panel["x"] + panel["w"])) ** 2
                    + (x["y"] - panel["y"]) ** 2
                ),
            )
            objects_return.extend(sorted_texts)
        objects["text"] = objects_return
        return objects


class PanelOrderEstimator:
    def __init__(self, boxes, thresh=0.2):
        self.thresh = thresh
        if type(boxes) == list:
            self.boxes = np.array(boxes, dtype=np.int)
        else:
            self.boxes = boxes
        if len(boxes) > 0:
            self.W, self.H = self.boxes[:, 2].max(), self.boxes[:, 3].max()
            self.box_orders = np.ones(len(boxes), np.int) * -1
            self.boxes = np.hstack(
                (self.boxes, np.arange(len(self.boxes)).reshape(len(self.boxes), 1))
            )
            self.boxes[:, (0, 2)] = self.W - self.boxes[:, (2, 0)]

    def split_box_set(self, boxes, order_offset, vertical, first_flag=False):
        boxes = sorted(
            boxes, key=lambda x: x[0 if vertical else 1]
        )  # sort by y position
        split_candidates = []
        for split_index in range(1, len(boxes)):
            boxes_upper = boxes[:split_index]
            boxes_lower = boxes[split_index:]
            scores = self.compute_split_score(
                boxes_upper, boxes_lower, vertical=vertical
            )
            ind = scores.argmin()
            split_candidates.append([split_index, ind, scores[ind]])
        split_points = list(filter(lambda x: x[2] < self.thresh, split_candidates))
        if len(split_points) == 0:
            if first_flag:
                return self.split_box_set(self.boxes, 0, True)
            else:
                for i, box in enumerate(boxes):
                    index = box[4]
                    self.box_orders[index] = order_offset + i
        else:
            prev = 0
            for split_point in split_points:
                self.split_box_set(
                    boxes[prev : split_point[0]],
                    order_offset + prev,
                    vertical=not vertical,
                )
                prev = split_point[0]
            self.split_box_set(boxes[prev:], order_offset + prev, vertical=not vertical)
        return self.box_orders

    def compute_split_score(self, boxes_upper, boxes_lower, vertical=True):
        if vertical:
            starts_upper = [b[0] for b in boxes_upper]
            starts_lower = [b[0] for b in boxes_lower]
            ends_upper = [b[2] for b in boxes_upper]
            ends_lower = [b[2] for b in boxes_lower]
            maxv = self.W
        else:
            starts_upper = [b[1] for b in boxes_upper]
            starts_lower = [b[1] for b in boxes_lower]
            ends_upper = [b[3] for b in boxes_upper]
            ends_lower = [b[3] for b in boxes_lower]
            maxv = self.H

        scores = np.zeros(maxv)
        for start, end in zip(starts_upper, ends_upper):
            scores[:end] += (end - np.arange(end)) / (end - start)
        for start, end in zip(starts_lower, ends_lower):
            scores[start:] += np.arange(maxv - start) / (end - start)
        return scores

    def get_panel_orders(self):
        if len(self.boxes) <= 1:
            return np.array([0] * len(self.boxes), dtype=np.int)
        return self.split_box_set(self.boxes, 0, False, first_flag=True)


def make_bbs(objects):
    return [
        [max(0, o["x"]), max(0, o["y"]), o["x"] + o["w"], o["y"] + o["h"]]
        for o in objects
    ]


def bbox_iou(bbox_a, bbox_b):
    """Compute IoU between bounding boxes (from ChainerCV)
    https://github.com/chainer/chainercv/blob/master/chainercv/utils/bbox/bbox_iou.py
    """
    if bbox_a.shape[1] != 4 or bbox_b.shape[1] != 4:
        raise IndexError
    # top left
    tl = np.maximum(bbox_a[:, None, :2], bbox_b[:, :2])
    # bottom right
    br = np.minimum(bbox_a[:, None, 2:], bbox_b[:, 2:])
    area_i = np.prod(br - tl, axis=2) * (tl < br).all(axis=2)
    area_a = np.prod(bbox_a[:, 2:] - bbox_a[:, :2], axis=1)
    area_b = np.prod(bbox_b[:, 2:] - bbox_b[:, :2], axis=1)

    return area_i / (area_a[:, None] + area_b - area_i)


def visualize_boxes_with_order(impath, boxes_lists, orders_lists, output_path):
    fig, ax = plt.subplots(figsize=(12, 12))
    image = Image.open(impath).convert("RGB")
    ax.imshow(image, aspect="equal")
    colors = ["firebrick", "green"]
    for list_ind, (boxes, orders) in enumerate(zip(boxes_lists, orders_lists)):
        for i, (bbox, order) in enumerate(zip(boxes, orders)):
            bbox = [int(v) for v in bbox]
            ax.add_patch(
                plt.Rectangle(
                    (bbox[0], bbox[1]),
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                    fill=False,
                    edgecolor=colors[list_ind],
                    linewidth=4,
                )
            )
            ax.text(
                bbox[0],
                bbox[1],
                int(order),
                bbox=dict(facecolor=colors[list_ind], alpha=0.6),
                fontsize=30,
                color="white",
            )
    plt.axis("off")
    plt.tight_layout()
    if not os.path.exists(os.path.dirname(output_path)):
        os.makedirs(os.path.dirname(output_path))
    plt.savefig(output_path, bbox_inches="tight")
    plt.close()
    print("Saved at {}".format(output_path))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d",
        "--dataset_root_path",
        default="./open-mantra-dataset",
        type=str,
        help="path to manga dataset",
    )
    parser.add_argument(
        "-o",
        "--output_dir",
        default="./output",
        help="directory to save images with the annotation of text order estimation",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    with open(os.path.join(args.dataset_root_path, "annotation.json")) as f:
        books = json.load(f)

    for bookdata in books:
        for page_id, pagedata in enumerate(bookdata["pages"]):
            pagedata = OrderEstimator().reorder_object(pagedata)
            image_path = os.path.join(
                args.dataset_root_path, pagedata["image_paths"]["ja"]
            )
            visualize_boxes_with_order(
                image_path,
                [make_bbs(pagedata["frame"]), make_bbs(pagedata["text"])],
                [range(len(pagedata["frame"])), range(len(pagedata["text"]))],
                os.path.join(
                    args.output_dir, f"{bookdata['book_title']}_{page_id}.jpg"
                ),
            )
