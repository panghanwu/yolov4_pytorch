from torchvision import ops
from torch import nn
import torch.nn.functional as fun
import torch
import numpy as np
    
    
    
def cal_ciou(b1, b2):
    """Calculate CIoU
    Calculate CIoU of two bboxes.
    b: bounding boxes tensor in (batch, head_w, head_h, num_anchors, 4), 
       4 for (x, y, w, h)

    Return
    ciou in tensor(batch, head_w, head_h, num_anchors, 1)
    """

    # transform box from (x, y, w, h) to (xmin, ymin, xmax, ymax)
    b1_xy = b1[..., 0:2]
    b1_wh = b1[..., 2:4]
    b1_minis = b1_xy - b1_wh/2
    b1_maxes = b1_xy + b1_wh/2

    b2_xy = b2[..., 0:2]
    b2_wh = b2[..., 2:4]
    b2_minis = b2_xy - b2_wh/2
    b2_maxes = b2_xy + b2_wh/2

    # calculate IoU
    # intersection
    inter_minis = torch.max(b1_minis, b2_minis)
    inter_maxes = torch.min(b1_maxes, b2_maxes)
    inter_wh = torch.max(inter_maxes-inter_minis, torch.zeros_like(inter_maxes))
    inter_area = inter_wh[:, 0] * inter_wh[:, 1]
    # union
    b1_area = b1_wh[:, 0] * b1_wh[:, 1]
    b2_area = b2_wh[:, 0] * b2_wh[:, 1]
    union_area = b1_area + b2_area - inter_area
    # iou
    iou = inter_area / union_area

    # calculate distance IoU
    # centers
    center_square = torch.sum(torch.pow(b1_xy-b2_xy,2), axis=1)
    # diagonal of enclosing box
    enc_minis = torch.min(b1_minis, b2_minis)
    enc_maxes = torch.max(b1_maxes, b2_maxes)
    enc_wh = enc_maxes - enc_minis
    diagonal_square = torch.sum(torch.pow(enc_wh,2), axis=1)
    # D-IoU
    diou = iou - center_square/diagonal_square

    # calculate radian (angle) IoU
    b1_radian = torch.atan(b1_wh[:,0]/b1_wh[:,1])
    b2_radian = torch.atan(b2_wh[:,0]/b2_wh[:,1])
    v = (4/np.pi**2) * torch.pow(b1_radian-b2_radian, 2)
    alpha = v / (1. - iou + v)

    # C-IoU
    ciou = diou - alpha*v
    return ciou    
    
    
    
class YOLOLoss(nn.Module):
    """
    anchors: total anchors in sequence
    """
    def __init__(
            self, 
            anchors, 
            num_classes, 
            img_size, 
            num_heads = 3,
            label_smoothing = 0.,
            cuda = True
        ):
        
        super(YOLOLoss, self).__init__()
        self.anchors = torch.Tensor(anchors)
        self.n_classes = num_classes
        self.img_size = img_size
        self.device = 'cuda' if cuda else 'cpu'
        
        # factors
        self.EGS_factor = 0.1  # for eliminate grid sensitivity
        self.ignoring_threshold = 0.5
        self.label_smoothing = label_smoothing
        self.lambda_conf = 1.0
        self.lambda_cls = 1.0
        self.lambda_ciou = 1.0
        
        # num anchors in each head
        self.n_head_anchors = len(anchors) // num_heads
        
        # for getting head index
        anchor_idx = np.array([a for a in range(len(self.anchors))])
        self.anchor_group = anchor_idx.reshape(-1, self.n_head_anchors)
        # /32 =  head 1, /16 =  head 2, /8 =  head 3
        self.head_widths = [img_size[0]//32, img_size[0]//16, img_size[0]//8]
        
    
    def smooth_labels(self, gt_cls, num_cls, factor):
        return gt_cls*(1.-factor) + factor/num_cls
    
    
    def get_anchor_map(self, targets):
        """
        Match each bbox to most similar anchor,and return indexes of anchors in list.
        targets: (N, 4), N objects in an image
        """
        scaled_anchors = torch.Tensor(self.anchors) / torch.Tensor(self.img_size)
        gt_w = targets[:, 2:3] 
        gt_h = targets[:, 3:4]
        # transform format of bboxes and anchors to (0, 0, xmax, ymax)
        # for computing box-anchor IOU metrix regardless of centers
        box_for_iou = torch.cat([torch.zeros_like(gt_w)]*2+[gt_w,gt_h], dim=1).to(self.device)
        anc_for_iou = torch.cat([torch.zeros(len(self.anchors),2),scaled_anchors], dim=1).to(self.device)
        iou_metrix = ops.box_iou(box_for_iou, anc_for_iou)
        anchor_maps = torch.argmax(iou_metrix, dim=1)
        return anchor_maps
    
    
    def encode_targets(self, targets, anchor_ids:list, head_size:list):
        """
        targets: list of tensor annotations in 
                 [tensor_bboxes1, tensor_bboxes2, ...]
        head_size: (h, w)
        """
        batch_size = len(targets)
        head_h, head_w = head_size
        # make containers (batch, num anchors in this head, H, W, ...)
        # turn off grad in training
        # obj: 1-> has object, 0 -> ignore
        obj_mask = torch.zeros(batch_size, self.n_head_anchors, *head_size, requires_grad=False).to(self.device)
        # noobj: 1-> not has object, 0 -> ignore
        noobj_mask = torch.ones(batch_size, self.n_head_anchors, *head_size, requires_grad=False).to(self.device)
        # 5+num_classes = (x, y, h, w, conf, num_classes)
        t_head = torch.zeros(batch_size, self.n_head_anchors, *head_size, 5+self.n_classes, requires_grad=False).to(self.device)
        
        box_scale = torch.zeros(batch_size, self.n_head_anchors, *head_size, requires_grad=False).to(self.device)
        
        for bs, tar in enumerate(targets):
            if len(tar) == 0:
                continue
            
            # groundtruth on this head size
            gt_x = tar[:, 0:1] * head_w
            gt_y = tar[:, 1:2] * head_h
            gt_w = tar[:, 2:3] * head_w
            gt_h = tar[:, 3:4] * head_h
            
            # distribute objects to grid (i, j) by center (x, y)
            gt_i = torch.floor(gt_x)
            gt_j = torch.floor(gt_y)
            
            # get which anchor gt bboxes belongs
            anchor_map = self.get_anchor_map(tar)

            for n, anc_id in enumerate(anchor_map):
                if int(anc_id) not in anchor_ids:
                    continue
                # shift anchor id of this bbox for this head
                anc_id = anc_id - anchor_ids[0]
                i, j = gt_i[n].long(), gt_j[n].long()  # grid coordinate
                # put object state to grids
                obj_mask[bs, anc_id, j, i] = 1
                noobj_mask[bs, anc_id, j, i] = 0
                
                # put box to grids
                t_head[bs, anc_id, j, i, 0] = gt_x[n]
                t_head[bs, anc_id, j, i, 1] = gt_y[n]
                t_head[bs, anc_id, j, i, 2] = gt_w[n]
                t_head[bs, anc_id, j, i, 3] = gt_h[n]
                t_head[bs, anc_id, j, i, 4] = 1
                
                # label
                label = tar[n, 4].long()
                t_head[bs, anc_id, j, i, 5+label] = 1
                
                # box scale = w*h
                box_scale[bs, anc_id, j, i] = tar[n, 2] * tar[n, 3]
        return obj_mask, noobj_mask, t_head, box_scale
            
    
    def set_ignoring(self, noobj_mask, inference, targets, head_anchors, head_size):
        """
        Args:
            head_anchors: anchors of this head
        """
        batch_size = len(targets)
        head_h, head_w = head_size
        # cx, cy, w, h
        x = (1+self.EGS_factor) * torch.sigmoid(inference[...,0]) - 0.5*self.EGS_factor
        y = (1+self.EGS_factor) * torch.sigmoid(inference[...,1]) - 0.5*self.EGS_factor
        w = inference[..., 2]
        h = inference[..., 3]
        
        # set device
        FloatTensor = torch.cuda.FloatTensor if self.device=='cuda' else torch.FloatTensor
        # generate coordinate grids
        grid_x = torch.linspace(0, head_w-1, head_w)
        grid_x = grid_x.repeat(head_h, 1).repeat(batch_size*self.n_head_anchors, 1, 1)
        grid_x = grid_x.view(x.shape).type(FloatTensor)

        grid_y = torch.linspace(0, head_h-1, head_h)
        grid_y = grid_y.repeat(head_w, 1).t().repeat(batch_size*self.n_head_anchors, 1, 1)
        grid_y = grid_y.view(y.shape).type(FloatTensor)
        
        # generate anchors for coordinate grids
        anchor_w = FloatTensor(head_anchors)[:, 0].unsqueeze(1)
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, head_h*head_w).view(w.shape)

        anchor_h = FloatTensor(head_anchors)[:, 1].unsqueeze(1)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, head_h*head_w).view(h.shape)

        # calculate bboxes
        infer_boxes = FloatTensor(inference[...,:4].shape)
        infer_boxes[..., 0] = x.data + grid_x
        infer_boxes[..., 1] = y.data + grid_y
        infer_boxes[..., 2] = torch.exp(w.data) * anchor_w
        infer_boxes[..., 3] = torch.exp(h.data) * anchor_h
        
        for bs, tar in enumerate(targets):
            if len(tar) == 0:
                continue
            # (num_anchors, 4)
            ignored_boxes = infer_boxes[bs].view(-1, 4)
            
            # groundtruth on this head size
            gt_x = tar[:, 0:1] * head_w
            gt_y = tar[:, 1:2] * head_h
            gt_w = tar[:, 2:3] * head_w
            gt_h = tar[:, 3:4] * head_h
            gt_box = FloatTensor(torch.cat([gt_x,gt_y,gt_w,gt_h],-1))
            
            # calculate IoU
            iou_metrix = ops.box_iou(gt_box, ignored_boxes)
            # get nearest groundtruth for each anchor
            max_iou, _ = torch.max(iou_metrix, dim=0)
            max_iou = max_iou.view(infer_boxes[bs].shape[:3])
            # turn off noobj_mask according to threshold
            noobj_mask[bs][max_iou > self.ignoring_threshold] = 0
        return noobj_mask, infer_boxes
            
    
    def forward(self, yolo_head, targets):
        # get head sizes tensor (B, C, H, W)
        batch_size = yolo_head.size(0)
        head_h = yolo_head.size(2)
        head_w = yolo_head.size(3)
        head_size = (head_h, head_w)
        
        # grid sizes of head
        grid_h = self.img_size[1] / head_h
        grid_w = self.img_size[0] / head_w 
        
        # get anchor ids of this head
        head_id = self.head_widths.index(head_w)
        anchor_ids = self.anchor_group[head_id]
        head_anchors = self.anchors[anchor_ids] / torch.Tensor([grid_w,grid_h])
        head_anchors = head_anchors.to(self.device)
        
        # rearrange tensor to predicted attributes of anchors
        # (batches, anchors, H, W, 5+num_classes)
        inference = yolo_head.reshape(
            batch_size,
            self.n_head_anchors,  # anchors in this head
            -1,  # adapt to (5 + num_classes)
            head_h,
            head_w
        ).permute(0, 1, 3, 4, 2)
        # confidence of bbox existence
        infer_conf = torch.sigmoid(inference[..., 4])
        # classes
        infer_cls = torch.sigmoid(inference[..., 5:])
        
        # get targets
        obj_mask, noobj_mask, t_head, box_scale = self.encode_targets(targets, anchor_ids, head_size)
        # set ignored grids near the ground truth
        noobj_mask, infer_boxes = self.set_ignoring(noobj_mask, inference, targets, head_anchors, head_size)
        
        # set device
        obj_mask, noobj_mask = obj_mask.to(self.device), noobj_mask.to(self.device)
        box_scale = box_scale.to(self.device)
        t_head = t_head.to(self.device)
        infer_boxes = infer_boxes.to(self.device)
        
        # CALCULATE LOSS!
        # bbox loss
        loss_weight = 2 - box_scale
        loss_ciou = (1 - cal_ciou(infer_boxes[obj_mask.bool()],t_head[...,:4][obj_mask.bool()])) * loss_weight[obj_mask.bool()]
        loss_ciou = torch.sum(loss_ciou)
        # object confidence loss
        loss_obj = fun.binary_cross_entropy(infer_conf, obj_mask, weight=obj_mask, reduction='sum')
        loss_noobj = fun.binary_cross_entropy(infer_conf, obj_mask, weight=noobj_mask, reduction='sum')
        loss_conf = loss_obj + loss_noobj
        # class loss
        smoothed_labels = self.smooth_labels(t_head[...,5:][obj_mask.bool()], self.n_classes, self.label_smoothing)
        loss_cls = fun.binary_cross_entropy(infer_cls[obj_mask.bool()], smoothed_labels, reduction='sum')
        # loss
        loss = self.lambda_conf*loss_conf + self.lambda_cls*loss_cls + self.lambda_ciou*loss_ciou
        return loss