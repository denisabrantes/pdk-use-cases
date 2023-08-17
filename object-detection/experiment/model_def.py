"""
This is an object detection finetuning example.  We finetune a Faster R-CNN
model pretrained on COCO to detect pedestrians in the relatively small PennFudan
dataset.

Useful References:
    https://docs.determined.ai/latest/reference/api/pytorch.html
    https://www.cis.upenn.edu/~jshi/ped_html/

Based on: https://pytorch.org/tutorials/intermediate/torchvision_tutorial.html
"""
import copy
import os
from typing import Any, Dict, Sequence, Union

import torch
import torchvision
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.tensorboard import SummaryWriter
import numpy as np

import determined as det
from determined.pytorch import DataLoader, LRScheduler, PyTorchTrial

from data import get_repo_path, download_data, get_transform, collate_fn, PennFudanDataset

TorchData = Union[Dict[str, torch.Tensor], Sequence[torch.Tensor], torch.Tensor]


class ObjectDetectionModel(PyTorchTrial):
    def __init__(self, context: det.TrialContext) -> None:
        self.context = context
        self.evaluate_count = 0
        self.writer = SummaryWriter(log_dir="/tmp/tensorboard")
        #self.current_step = context.env.first_step()

        # Create a unique download directory for each rank so they don't
        # overwrite each other.
        self.repo_dir = get_repo_path(self.data_config["pachyderm"]["host"], 
                                      self.data_config["pachyderm"]["port"], 
                                      self.data_config["pachyderm"]["repo"], 
                                      self.data_config["pachyderm"]["branch"], 
                                      self.data_config["pachyderm"]["token"], 
                                      self.data_config["pachyderm"]["project"]
                                      )
        
        self.download_directory = f"/tmp/data-rank{self.context.distributed.get_rank()}"

        self.data_config = self.context.get_data_config()

        download_data(self.data_config, self.download_directory)

        # download_data(download_directory=self.download_directory, data_config=self.context.get_data_config(),)

        os.environ['TORCH_HOME'] = self.download_directory

        dataset_folder = f"{self.download_directory}/{self.repo_dir}/PennFudanPed"
        print("--> Loading Dataset from this folder: ", dataset_folder)
        dataset = PennFudanDataset(dataset_folder, get_transform())

        # Split 80/20 into training and validation datasets.
        train_size = int(0.8 * len(dataset))
        test_size = len(dataset) - train_size
        self.dataset_train, self.dataset_val = torch.utils.data.random_split(
            dataset, [train_size, test_size]
        )

        self.model = self.context.wrap_model(self._build_model())
        self.opt = self.context.wrap_optimizer(self._build_optimizer(self.model))
        self.lr_scheduler = self.context.wrap_lr_scheduler(
            self._build_lr_scheduler(self.opt),
            step_mode=LRScheduler.StepMode.STEP_EVERY_EPOCH
        )

    def build_training_data_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset_train,
            batch_size=self.context.get_per_slot_batch_size(),
            collate_fn=collate_fn,
        )

    def build_validation_data_loader(self) -> DataLoader:
        return DataLoader(
            self.dataset_val,
            batch_size=self.context.get_per_slot_batch_size(),
            collate_fn=collate_fn,
        )

    def _build_model(self) -> nn.Module:
        model = fasterrcnn_resnet50_fpn(pretrained=True)

        # Replace the classifier with a new two-class classifier.  There are
        # only two "classes": pedestrian and background.
        num_classes = 2
        in_features = model.roi_heads.box_predictor.cls_score.in_features
        model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
        return model

    def _build_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.SGD(
            model.parameters(),
            lr=self.context.get_hparam("learning_rate"),
            momentum=self.context.get_hparam("momentum"),
            weight_decay=self.context.get_hparam("weight_decay"),
        )

    def _build_lr_scheduler(self, optimizer):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    def train_batch(
        self, batch: TorchData, epoch_idx: int, batch_idx: int
    ) -> Dict[str, torch.Tensor]:
        images, targets = batch
        loss_dict = self.model(list(images), list(targets))
        total_loss = sum([loss_dict[l] for l in loss_dict])
        self.context.backward(total_loss)
        self.context.step_optimizer(self.opt, self.clip_grads)

        # Set current step based on 100 batches / step
        self.current_batch = batch_idx

        return {"loss": total_loss}

    def evaluate_batch(self, batch: TorchData) -> Dict[str, Any]:
        images, targets = batch
        output = self.model(list(images), copy.deepcopy(list(targets)))
        sum_iou = 0
        num_boxes = 0
        # Instantiate the Tensorboard writer and set the log_dir to /tmp/tensorboard where Determined looks for events
        #writer = SummaryWriter(log_dir="/tmp/tensorboard")

        # Only write out one image per evaluation run
        #self.new_batch = True
        #selected_image_idx = min(len(targets), self.evaluate_count)
        #print(f"self.evaluate_count: {self.evaluate_count}")
        #print(f"selected_image_idx: {selected_image_idx}")

        # Our eval metric is the average best IoU (across all predicted
        # pedestrian bounding boxes) per target pedestrian.  Given predicted
        # and target bounding boxes, IoU is the area of the intersection over
        # the area of the union.
        for idx, target in enumerate(targets):
            # Filter out overlapping bounding box predictions based on
            # non-maximum suppression (NMS)
            predicted_boxes = output[idx]["boxes"]
            prediction_scores = output[idx]["scores"]
            keep_indices = torchvision.ops.nms(predicted_boxes, prediction_scores, 0.1)
            predicted_boxes = torch.index_select(predicted_boxes, 0, keep_indices)
            prediction_scores = torch.index_select(prediction_scores, 0, keep_indices)

            # Tally IoU with respect to the ground truth target boxes
            target_boxes = target["boxes"]
            boxes_iou = torchvision.ops.box_iou(target_boxes, predicted_boxes)
            sum_iou += sum(max(iou_result) for iou_result in boxes_iou)
            num_boxes += len(target_boxes)

            # boxes are ordered by confidence, so get the top 5 bounding boxes and write out to Tensorboard
            # new_predicted_boxes = output[idx]["boxes"][:5]
            threshold = 0.7
            cutoff = 0
            for i, score in enumerate(prediction_scores):
                if score < threshold:
                    break
                cutoff = i
            new_predicted_boxes = output[idx]["boxes"][:cutoff]

            #if self.new_batch and idx == selected_image_idx:
            self.writer.add_image_with_boxes("batch_"+str(self.current_batch)+"_"+str(idx), images[idx], predicted_boxes)
            #self.new_batch = False
            #self.evaluate_count += 1

        self.writer.close()

        return {"val_avg_iou": sum_iou / num_boxes}

    def clip_grads(self, params):
        torch.nn.utils.clip_grad_norm_(params, self.context.get_hparam("clip_grad"))
