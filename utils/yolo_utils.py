from torchvision import ops
import torch
import torch.nn as nn
import numpy as np



class HeadDecoder(nn.Module):
    """
    Decode YOLO head from (B, (5 + num_classes), H, W) to 
    (B, num_anchors*H*W, (5 + num_classes)).
    Also, calculate coordinate of bounding boxes from predicted factors.
    """
    def __init__(self, anchors:list, num_classes:int, img_size:tuple):
        super(HeadDecoder, self).__init__()
        self.anchors = np.array(anchors)
        self.num_anchors = len(anchors)
        self.num_classes = num_classes
        self.img_size = img_size
        self.EGS_factor = 0.1  # for eliminate grid sensitivity

    def forward(self, yolo_head):
        # get head sizes tensor (B, C, H, W)
        batch_size = yolo_head.size(0)
        head_height = yolo_head.size(2)
        head_width = yolo_head.size(3)

        # grid sizes of head
        grid_height = self.img_size[1] / head_height
        grid_width = self.img_size[0] / head_width

        # scale anchors according to head
        scaled_anchors = self.anchors / np.array([grid_width, grid_width])

        # rearrange tensor to predicted attributes of anchors
        inference = yolo_head.reshape(
            batch_size,
            self.num_anchors,
            -1,  # adapt to (5 + num_classes)
            head_height,
            head_width
        ).permute(0, 1, 3, 4, 2)

        # centers of bbox
        x = (1+self.EGS_factor) * torch.sigmoid(inference[...,0]) - 0.5*self.EGS_factor
        y = (1+self.EGS_factor) * torch.sigmoid(inference[...,1]) - 0.5*self.EGS_factor
        # width and height of bbox
        w = inference[..., 2]
        h = inference[..., 3]
        # confidence of bbox existence
        conf = torch.sigmoid(inference[..., 4])
        # classes
        infer_cls = torch.sigmoid(inference[..., 5:])

        # set device
        FloatTensor = torch.cuda.FloatTensor if x.is_cuda else torch.FloatTensor

        # generate coordinate grids
        grid_x = torch.linspace(0, head_width-1, head_width)
        grid_x = grid_x.repeat(head_height, 1).repeat(batch_size*self.num_anchors, 1, 1)
        grid_x = grid_x.view(x.shape).type(FloatTensor)

        grid_y = torch.linspace(0, head_height-1, head_height)
        grid_y = grid_y.repeat(head_width, 1).t().repeat(batch_size*self.num_anchors, 1, 1)
        grid_y = grid_y.view(y.shape).type(FloatTensor)

        # generate anchors for coordinate grids
        anchor_w = FloatTensor(scaled_anchors)[:, 0].unsqueeze(1)
        anchor_w = anchor_w.repeat(batch_size, 1).repeat(1, 1, head_height*head_width).view(w.shape)

        anchor_h = FloatTensor(scaled_anchors)[:, 1].unsqueeze(1)
        anchor_h = anchor_h.repeat(batch_size, 1).repeat(1, 1, head_height*head_width).view(h.shape)

        # calculate bboxes
        infer_boxes = FloatTensor(inference[...,:4].shape)
        infer_boxes[..., 0] = x.data + grid_x
        infer_boxes[..., 1] = y.data + grid_y
        infer_boxes[..., 2] = torch.exp(w.data) * anchor_w
        infer_boxes[..., 3] = torch.exp(h.data) * anchor_h

        # reversely scale the grids to image size
        scale = FloatTensor([grid_width,grid_height] * 2)  # [w, h, w, h]
        output = torch.cat((
            infer_boxes.reshape(batch_size, -1, 4) * scale,
            conf.reshape(batch_size, -1, 1),
            infer_cls.reshape(batch_size, -1, self.num_classes)
        ), -1)
        return output


def non_max_suppression(inferred_bboxes, conf_threshold=0.5, iou_threshold=0.3):
    """
    inferred_bboxes: torch.Tensor in (batch, total_anchors, (5 + num_classes)).
                     bbox format in (x, y, w, h)
    """
    # transfer (x, y, w, h) to (x_min, y_min, x_max, y_max)
    box_corners = inferred_bboxes.new(inferred_bboxes.shape)  # temp container
    box_corners[..., 0] = inferred_bboxes[..., 0] - inferred_bboxes[..., 2] / 2
    box_corners[..., 1] = inferred_bboxes[..., 1] - inferred_bboxes[..., 3] / 2
    box_corners[..., 2] = inferred_bboxes[..., 0] + inferred_bboxes[..., 2] / 2
    box_corners[..., 3] = inferred_bboxes[..., 1] + inferred_bboxes[..., 3] / 2
    # save back to inferred-bbox tensor
    inferred_bboxes[..., :4] = box_corners[..., :4]
    
    # for each batch
    output = [None for _ in range(len(inferred_bboxes))]  # container
    for batch_id, bbx_inf in enumerate(inferred_bboxes):
        # get max confidence for each class
        cls_conf, cls_inf = torch.max(bbx_inf[:, 5:], 1, keepdim=True)

        # phase 1: kill confidence < conf_threshold
        conf_mask = (bbx_inf[:, 4] * cls_conf[:, 0] >= conf_threshold).squeeze()
        bbx_inf = bbx_inf[conf_mask]
        cls_conf = cls_conf[conf_mask]
        cls_inf = cls_inf[conf_mask]
        # (num_bboxes, (coordinates, bbox conf, class conf, class))
        detections = torch.cat([bbx_inf[:,:5], cls_conf.float(), cls_inf.float()], 1)
        if not bbx_inf.size(0):
            continue  # if get objects

        # phase 2: non max suppression by torch.nms
        # get unique labels
        labels = detections[:, -1].cpu().unique()
        # set device
        if inferred_bboxes.is_cuda:
            labels = labels.cuda()
            detections = detections.cuda()
        # for each class in labels
        for c in labels:
            # get detections of c class
            class_detections = detections[detections[:,-1] == c]

            keep = ops.nms(
                boxes = class_detections[:, :4], 
                scores = class_detections[:, 4], 
                iou_threshold = iou_threshold
            ) 

            nms_result = detections[keep]

            # add to output container
            if output[batch_id] is not None:
                torch.cat([output[batch_id], nms_result])
            else: 
                output[batch_id] = nms_result
    return output
            
        

