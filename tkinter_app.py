import tkinter
import cv2
import PIL.Image, PIL.ImageTk
import time

import torch
from torch import nn
import torch.nn.functional as F
from fastai.vision.learner import create_cnn
from torchvision.models import resnet34
import numpy as np
import json
import os
from pathlib import Path
import fastai
import numpy as np
import pandas as pd
from pathlib import Path

from fastai.vision.transform import get_transforms
from fastai.vision.data import ObjectItemList, imagenet_stats#, bb_pad_collate
from fastai import *
from fastai.vision import *
import ssdoil

def conv_params(in_size, out_size):
    filters = [3, 2, 5, 4]
    strides = [1, 2, 3]  # max_stride = 3
    pads = [0, 1, 2, 3]  # max pad

    if out_size == 1:
        return 1, 0, in_size

    for filter_size in filters:
        for pad in pads:
            for stride in strides:
                if ((out_size - 1) * stride == (in_size - filter_size) + 2 * pad):
                    return stride, pad, filter_size
    return None, None, None


class StdConv(nn.Module):
    def __init__(self, nin, nout, filter_size=3, stride=2, padding=1, drop=0.1):
        super().__init__()
        self.conv = nn.Conv2d(nin, nout, filter_size, stride=stride, padding=padding)
        self.bn = nn.BatchNorm2d(nout)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        return self.drop(self.bn(F.relu(self.conv(x))))


def flatten_conv(x, k):
    bs, nf, gx, gy = x.size()
    x = x.permute(0, 2, 3, 1).contiguous()
    return x.view(bs, -1, nf // k)


class OutConv(nn.Module):
    def __init__(self, k, nin, num_classes, bias):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(nin, (num_classes) * k, 3, padding=1)
        self.oconv2 = nn.Conv2d(nin, 3 * k, 3, padding=1)
        self.oconv1.bias.data.zero_().add_(bias)

    def forward(self, x):
        return [flatten_conv(self.oconv1(x), self.k),
                flatten_conv(self.oconv2(x), self.k)]


class SSDHead(nn.Module):
    def __init__(self, grids, anchors_per_cell, num_classes, drop=0.3, bias=-4.):
        super().__init__()
        self.drop = nn.Dropout(drop)

        self.sconvs = nn.ModuleList([])
        self.oconvs = nn.ModuleList([])

        self.anc_grids = grids

        self._k = anchors_per_cell

        self.sconvs.append(StdConv(512, 256, stride=1, drop=drop))

        for i in range(len(grids)):

            if i == 0:
                stride, pad, filter_size = conv_params(7, grids[i])  # get '7' by base model
            else:
                stride, pad, filter_size = conv_params(grids[i - 1], grids[i])

            if stride is None:
                print(grids[i - 1], ' --> ', grids[i])
                raise Exception('cannot create model for specified grids')

            self.sconvs.append(StdConv(256, 256, filter_size, stride=stride, padding=pad, drop=drop))
            self.oconvs.append(OutConv(self._k, 256, num_classes=num_classes, bias=bias))

    def forward(self, x):
        x = self.drop(F.relu(x))
        x = self.sconvs[0](x)
        out_classes = []
        out_bboxes = []
        for sconv, oconv in zip(self.sconvs[1:], self.oconvs):
            x = sconv(x)
            out_class, out_bbox = oconv(x)
            out_classes.append(out_class)
            out_bboxes.append(out_bbox)

        return [torch.cat(out_classes, dim=1),
                torch.cat(out_bboxes, dim=1)]


def one_hot_embedding(labels, num_classes):
    return torch.eye(num_classes)[labels.data.cpu()]


class BCE_Loss(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, pred, targ):
        t = one_hot_embedding(targ, self.num_classes)
        t = torch.Tensor(t[:, 1:].contiguous()).cuda()
        x = pred[:, 1:]
        w = self.get_weight(x, t)
        return F.binary_cross_entropy_with_logits(x, t, w, size_average=False) / (self.num_classes - 1)

    def get_weight(self, x, t): return None


class FocalLoss(BCE_Loss):
    def get_weight(self, x, t):
        alpha, gamma = 0.25, 1
        p = x.sigmoid()
        pt = p * t + (1 - p) * (1 - t)
        w = alpha * t + (1 - alpha) * (1 - t)
        w = w * (1 - pt).pow(gamma)
        return w.detach()


class _EmptyData():
    def __init__(self, path, c, loss_func: None):
        self.path = path
        self.device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')
        self.c = c
        self.loss_func = loss_func


class SingleShotDetector(object):

    def __init__(self, data, grids=[4, 2, 1], zooms=[0.7, 1., 1.3], ratios=[[1., 1.], [1., 0.5], [0.5, 1.]],
                 backbone=None, drop=0.3, bias=-4., focal_loss=False, pretrained_path=None):

        super().__init__()

        self._device = torch.device('cuda') if torch.cuda.is_available else torch.device('cpu')

        if backbone is None:
            backbone = resnet34

        self._create_anchors(grids, zooms, ratios)

        ssd_head = SSDHead(grids, self._anchors_per_cell, data.c, drop=drop, bias=bias)

        self._data = data
        self.learn = cnn_learner(data=data, base_arch=backbone, custom_head=ssd_head)
        self.learn.model = self.learn.model.to(self._device)

        if pretrained_path is not None:
            self.load(pretrained_path)

        if focal_loss:
            self._loss_f = FocalLoss(data.c)
        else:
            self._loss_f = BCE_Loss(data.c)

        self.learn.loss_func = self._ssd_loss

    @classmethod
    def from_emd(cls, data, emd_path):
        emd = json.load(open(emd_path))
        class_mapping = {i['Value']: i['Name'] for i in emd['Classes']}
        if data is None:
            empty_data = _EmptyData(path='str', loss_func=None, c=len(class_mapping) + 1)
            return cls(empty_data, emd['Grids'], emd['Zooms'], emd['Ratios'], pretrained_path=emd['ModelFile'])
        else:
            return cls(data, emd['Grids'], emd['Zooms'], emd['Ratios'], pretrained_path=emd['ModelFile'])

    def lr_find(self):
        from IPython.display import clear_output
        self.learn.lr_find()
        clear_output()
        self.learn.recorder.plot()

    def fit(self, epochs=10, lr=slice(1e-4, 3e-3)):
        self.learn.fit(epochs, lr)

    def fit_one_cycle(self, epochs=10, lr=slice(1e-4, 3e-3)):
        self.learn.fit_one_cycle(epochs, lr)

    def unfreeze(self):
        self.learn.unfreeze()

    def _create_anchors(self, anc_grids, anc_zooms, anc_ratios):

        self.grids = anc_grids
        self.zooms = anc_zooms
        self.ratios = anc_ratios

        anchor_scales = [(anz * i, anz * j) for anz in anc_zooms for (i, j) in anc_ratios]

        self._anchors_per_cell = len(anchor_scales)

        anc_offsets = [1 / (o * 2) for o in anc_grids]

        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1 - ao, ag), ag)
                                for ao, ag in zip(anc_offsets, anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1 - ao, ag), ag)
                                for ao, ag in zip(anc_offsets, anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x, anc_y], axis=1), self._anchors_per_cell, axis=0)

        anc_sizes = np.concatenate([np.array([[o / ag, p / ag] for i in range(ag * ag) for o, p in anchor_scales])
                                    for ag in anc_grids])

        self._grid_sizes = torch.Tensor(
            np.concatenate([np.array([1 / ag for i in range(ag * ag) for o, p in anchor_scales])
                            for ag in anc_grids])).unsqueeze(1).to(self._device)

        self._anchors = torch.Tensor(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float().to(self._device)

        self._anchor_cnr = self._hw2corners(self._anchors[:, :2], self._anchors[:, 2:])

    def _hw2corners(self, ctr, hw):
        return torch.cat([ctr - hw / 2, ctr + hw / 2], dim=1)

    def _get_y(self, bbox, clas):
        bbox = bbox.view(-1, 4)  # /sz
        bb_keep = ((bbox[:, 2] - bbox[:, 0]) > 0).nonzero()[:, 0]
        return bbox[bb_keep], clas[bb_keep]

    def _actn_to_bb(self, actn, anchors, grid_sizes):
        actn_bbs = torch.tanh(actn)
        actn_centers = (actn_bbs[..., :2] / 2 * grid_sizes) + anchors[:, :2]
        actn_hw = (actn_bbs[..., 2:] / 2 + 1) * anchors[:, 2:]
        return self._hw2corners(actn_centers, actn_hw)

    def _map_to_ground_truth(self, overlaps, print_it=False):
        prior_overlap, prior_idx = overlaps.max(1)
        if print_it: print(prior_overlap)
        gt_overlap, gt_idx = overlaps.max(0)
        gt_overlap[prior_idx] = 1.99
        for i, o in enumerate(prior_idx): gt_idx[o] = i
        return gt_overlap, gt_idx

    def _ssd_1_loss(self, b_c, b_bb, bbox, clas, print_it=False):
        bbox, clas = self._get_y(bbox, clas)
        bbox = self._normalize_bbox(bbox)

        a_ic = self._actn_to_bb(b_bb, self._anchors, self._grid_sizes)
        overlaps = self._jaccard(bbox.data, self._anchor_cnr.data)
        try:
            gt_overlap, gt_idx = self._map_to_ground_truth(overlaps, print_it)
        except Exception as e:
            return 0., 0.
        gt_clas = clas[gt_idx]
        pos = gt_overlap > 0.4
        pos_idx = torch.nonzero(pos)[:, 0]
        gt_clas[1 - pos] = 0  # data.c - 1 # CHANGE
        gt_bbox = bbox[gt_idx]
        loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
        clas_loss = self._loss_f(b_c, gt_clas)
        return loc_loss, clas_loss

    def _ssd_loss(self, pred, targ1, targ2, print_it=False):
        lcs, lls = 0., 0.
        for b_c, b_bb, bbox, clas in zip(*pred, targ1, targ2):
            loc_loss, clas_loss = self._ssd_1_loss(b_c, b_bb, bbox.cuda(), clas.cuda(), print_it)
            lls += loc_loss
            lcs += clas_loss
        if print_it: print(f'loc: {lls}, clas: {lcs}')  # CHANGE
        return lls + lcs

    def _intersect(self, box_a, box_b):
        max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
        min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
        inter = torch.clamp((max_xy - min_xy), min=0)
        return inter[:, :, 0] * inter[:, :, 1]

    def _box_sz(self, b):
        return ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))

    def _jaccard(self, box_a, box_b):
        inter = self._intersect(box_a, box_b)
        union = self._box_sz(box_a).unsqueeze(1) + self._box_sz(box_b).unsqueeze(0) - inter
        return inter / union

    def _normalize_bbox(self, bbox):
        return (bbox + 1.) / 2.

    def save(self, name_or_path):
        if '\\' in name_or_path or '/' in name_or_path:
            path = Path(name_or_path)
            name = path.stem
            # to make fastai save to both path and with name
            temp = self.learn.path
            self.learn.path = path.parent
            self.learn.model_dir = ''
            if not os.path.exists(self.learn.path):
                os.makedirs(self.learn.path)
            saved_path = self.learn.save(name, return_path=True)
            # undoing changes to self.learn.path and self.learn.model
            self.learn.path = temp
            self.learn.model_dir = 'models'
        else:
            temp = self.learn.path
            # fixing fastai bug
            self.learn.path = self.learn.path.parent
            if not os.path.exists(self.learn.path / self.learn.model_dir):
                os.makedirs(self.learn.path / self.learn.model_dir)
            saved_path = self.learn.save(name_or_path, return_path=True)
            # undoing changes to self.learn.path
            self.learn.path = temp

    def load(self, name_or_path):
        if '\\' in name_or_path or '/' in name_or_path:
            path = Path(name_or_path)
            name = path.stem
            # to make fastai from both path and with name
            temp = self.learn.path
            self.learn.path = path.parent
            self.learn.model_dir = ''
            self.learn.load(name)
            # undoing changes to self.learn.path and self.learn.model_dir
            self.learn.path = temp
            self.learn.model_dir = 'models'
        else:
            temp = self.learn.path
            # fixing fastai bug
            self.learn.path = self.learn.path.parent
            self.learn.load(name_or_path)
            # undoing changes to self.learn.path
            self.learn.path = temp

    def show_results(self, rows=5, thresh=0.5, nms_overlap=0.1):
        self.learn.show_results(rows=rows, thresh=thresh, nms_overlap=nms_overlap, ssd=self)


data = _EmptyData("data/cardBB/img", c=2, loss_func = None)


def card_cutout(img, bboxes):
    n = len(bboxes)
    avgSize = int((bboxes[:,2] - bboxes[:,0]).mean())
    bbox_boundary = img.size(1) - avgSize
    bboxes = bboxes.type(torch.int)
    #out = torch.cuda.IntTensor(n,mean,mean)
    out = (img[:, min(bbox_boundary, max(0,bboxes[0][0])):min(bbox_boundary, max(0,bboxes[0][0]))+avgSize, bboxes[0][1]:bboxes[0][1]+avgSize]).unsqueeze(0)
    for i in range(1,n):
        #out[i] = img[:, bboxes[i][0]:bboxes[i][0]+mean, bboxes[i][1]:bboxes[i][1]+mean]
        card = img[:, min(bbox_boundary, max(0,bboxes[i][0])):min(bbox_boundary, max(0,bboxes[i][0]))+avgSize, bboxes[i][1]:bboxes[i][1]+avgSize]
        if card.size(1) != avgSize:
            card = img[:, bboxes[i][0]:bboxes[i][0] + avgSize, bboxes[i][1]:bboxes[i][1] + avgSize]
        out = torch.cat((out, card.unsqueeze(0)))
    return out


class App:
    def __init__(self, window, window_title, video_source=0):
        print(torch.get_default_dtype())
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source

        # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        print(self.vid.width, self.vid.height)

        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()

        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.offset = int((self.vid.width - self.vid.height) / 2)
        self.numCards = 0
        self.i = 0
        self.update()

        self.window.mainloop()


    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        boxframe = frame[:, self.offset:self.vid.width - self.offset, :]
        boxframe = cv2.resize(boxframe, (224, 224), interpolation=cv2.INTER_AREA)
        boxframe = np.transpose(boxframe, (2, 0, 1))
        boxframe = tensor(boxframe)
        boxframe = boxframe.float()
        boxframe = boxframe.unsqueeze(0)
        boxframe = boxframe.cuda()
        boxframe = boxframe / 255
        classifyframe = np.transpose(frame, (2, 0, 1))
        classifyframe = tensor(classifyframe)
        classifyframe = classifyframe.float()
        classifyframe = classifyframe.cuda()
        classifyframe = classifyframe / 255

        y = ssd.learn.pred_batch(DatasetType.Valid, batch=[boxframe, one_y])
        bboxes = ssdoil.SSDObjectCategoryList.analyze_pred(pred=grab_idx(y, 0), thresh=0.2, ssd=ssd)
        matches = []
        if bboxes is not None:
            bboxes = bboxes[0]
            bboxes = ((bboxes + 1) * self.vid.height)/2
            bboxes[:, 1] = bboxes[:, 1] + self.offset
            bboxes[:, 3] = bboxes[:, 3] + self.offset
            cards = card_cutout(classifyframe, bboxes)
            cardsFlipped = cards.flip(2)
            cardsFlipped2 = cards.flip(3)
            joined = torch.cat((cards, cardsFlipped, cardsFlipped2))
            #print(classGuesser.model(joined))
            #print(classGuesser.model(joined).view(3,bboxes.size(0),-1))
            #print(classGuesser.model(joined).view(3, bboxes.size(0), -1).sum(0))
            if bboxes.size(0) != self.numCards:
                self.numCards = bboxes.size(0)
                self.historicPreds = torch.zeros(10, self.numCards, 54, requires_grad = False)
            self.historicPreds[self.i] = classGuesser.model(joined).view(3, bboxes.size(0), -1).sum(0).detach()
            self.i = (self.i+1) % 10
            #print(self.historicPreds)
            avgPred = self.historicPreds.sum(0)
            #print(avgPred.size())
            idx = np.array(torch.topk(avgPred, 8)[1])
            icons = []
            for x in idx:
                icons.append([classGuesser.data.classes[i] for i in x])
            for i in range(len(cards)):
                for j in range(i + 1, len(cards)):
                    matches.append(np.intersect1d(idx[i], idx[j]))
            for i in range(len(matches)):
                matches[i] = [classGuesser.data.classes[j] for j in matches[i]]
            print(matches)
            for i in range(len(bboxes)):
                frame = cv2.rectangle(frame, (bboxes[i][1], bboxes[i][0]), (bboxes[i][3], bboxes[i][2]), (0, 255, 0), 2)
                frame = cv2.putText(frame, " ".join(icons[i][:4]), (bboxes[i][1], bboxes[i][0]), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)
                frame = cv2.putText(frame, " ".join(icons[i][4:]), (bboxes[i][1], bboxes[i][0]+20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        frame = cv2.rectangle(frame, (self.offset, 0), (self.vid.width - self.offset, self.vid.height), (255, 0, 0), 2)
        spacer = 20
        for x in matches:
            frame = cv2.putText(frame, " ".join(x), (20, spacer), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            spacer += 20
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)

        self.window.after(self.delay, self.update)


class MyVideoCapture:
    def __init__(self, video_source=0):
        # Open the video source
        self.vid = cv2.VideoCapture(video_source)
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)

        # Get video source width and height
        self.width = int(self.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
             # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)

    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()


ssd = SingleShotDetector(data, ratios=[[1.0, 1.0]])
ssd.load('ssd-tuesday2-20')
classGuesser = load_learner("models")
one_y = [torch.FloatTensor(1,1,4),torch.FloatTensor(1,5)]

# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")