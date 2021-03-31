# from PIL import
from PIL import Image, ImageDraw, ImageFont
import numpy as np
import torch.nn as nn
import torch
import os

# local module
from models.yolov4 import YOLOv4
from utils.yolo_utils import (
    HeadDecoder, 
    non_max_suppression
)
from configs import DetectionConfig


# load configuration
config = DetectionConfig()



class Detector:
    def __init__(self):
        self.class_names = config.class_names
        self.input_size = config.input_size
        self.anchors = np.array(config.anchors)
        self.confidence = config.confidence
        self.iou = config.iou
        self.text_font = config.text_font
        
        
    # set/reset model
    def init(self, weight_path, device='cpu'):
        print('Initializing model...')
        
        
        # set decive
        assert device in ['cpu', 'cuda']
        if device == 'cuda':
            if torch.cuda.is_available():
                device = torch.device('cuda')
                self.use_cuda = True
                print('Set device to "cuda" successfully!')
            else:
                device = torch.device('cpu')
                print('Cannot reach available "cuda". The device will set to "cpu".')
                self.use_cuda = False
        elif device == 'cpu':
            device = torch.device('cpu')
            self.use_cuda = False
            print('Set device to "cpu" successfully!')
        
        
        # load model
        print('Loading weights into state dict...')
        self.net = YOLOv4(len(self.anchors[0]), len(self.class_names)).eval()
        state_dict = torch.load(weight_path, map_location=device)
        self.net.load_state_dict(state_dict, strict=True)

        # I dont know what is this.
        # Seem like parallel computation setting...
        if self.use_cuda:
            os.environ["CUDA_VISIBLE_DEVICES"] = '0'
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()
        
        # TODO change to 3 heads
        # bbox decoder
        self.yolo_decodes = []
        self.anchors_mask = [[3,4,5],[1,2,3]]
        for i in range(2):
            self.yolo_decodes.append(HeadDecoder(
                np.reshape(self.anchors,[-1,2])[self.anchors_mask[i]], 
                len(self.class_names),  
                (self.input_size[1], self.input_size[0])
            ))      
            
        print('Finished!')
        
        
    def detect(self, image, grayscale=False): 
        """
        image: PIL image
        """
        image_shape = np.array(np.shape(image)[0:2])
        # make a copy
        self.image = image.copy()

        if grayscale:
            image = image.convert('L')
        # set channels to 3
        image = image.convert('RGB')
        # resize to fit input
        image = image.resize((self.input_size[1],self.input_size[0]), Image.BICUBIC)
        # normalize
        image = np.array(image, dtype=np.float32) / 255.0
        # transpose (W, H, C) to (C, W, H)
        image = np.transpose(image, (2, 0, 1))
        
        # turn off autograd
        with torch.no_grad():
            self.result = []  # create result container
            image = torch.from_numpy(image).unsqueeze(0)  # (B, C, H, W)

            if self.use_cuda:
                image = image.cuda()
                
            # detect via model
            outputs = self.net(image)
            
            # decode yolo heads
            output_list = []
            for i in range(2):
                output_list.append(self.yolo_decodes[i](outputs[i]))
                
            # NMS
            output = torch.cat(output_list, 1)
            batch_detections = non_max_suppression(
                output, 
                len(self.class_names),
                conf_threshold = self.confidence,
                iou_threshold = self.iou
            )
            
            # if no object is detected
            try:
                batch_detections = batch_detections[0].cpu().numpy()
            except:
                return self.result
            
            # filter bbox under threshold
            top_index = batch_detections[:, 4]*batch_detections[:, 5] > self.confidence
            top_conf = batch_detections[top_index, 4] * batch_detections[top_index, 5]
            top_label = np.array(batch_detections[top_index,-1], np.int32)
            top_bboxes = np.array(batch_detections[top_index,:4])
            top_xmin = np.expand_dims(top_bboxes[:,0],-1)
            top_ymin = np.expand_dims(top_bboxes[:,1],-1)
            top_xmax = np.expand_dims(top_bboxes[:,2],-1)
            top_ymax = np.expand_dims(top_bboxes[:,3],-1)
            
            # align bbox back to origin size
            top_xmin = top_xmin/self.input_size[1] * image_shape[1]
            top_ymin = top_ymin/self.input_size[0] * image_shape[0]
            top_xmax = top_xmax/self.input_size[1] * image_shape[1]
            top_ymax = top_ymax/self.input_size[0] * image_shape[0]
            boxes = np.concatenate([top_xmin,top_ymin,top_xmax,top_ymax], axis=-1)
            
            # gather bbox to list (c, s, (top, left, bottom, right))
            for i, c in enumerate(top_label):
                predicted_class = self.class_names[c]
                score = top_conf[i]
                self.result.append([c,predicted_class,score,list(boxes[i])])
            return self.result
        