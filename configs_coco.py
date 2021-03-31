class Config:
    def __init__(self):
        self.model_name = 'YOLOv4 COCO dataset'
        self.input_size = (416, 416)  # (w, h)
        # threshold
        self.confidence = 0.5
        self.iou = 0.3
        self.text_font = 'simhei.ttf'

        # anchors
        self.anchors = [
            [(12, 16),  (19, 36),  (40, 28)],  
            [(36, 75),  (76, 55),  (72, 146)],  
            [(142, 110),  (192, 243),  (459, 401)]  
        ]    

        # classes
        self.class_names = [
            'person',
            'bicycle',
            'car',
            'motorbike',
            'aeroplane',
            'bus',
            'train',
            'truck',
            'boat',
            'traffic light',
            'fire hydrant',
            'stop sign',
            'parking meter',
            'bench',
            'bird',
            'cat',
            'dog',
            'horse',
            'sheep',
            'cow',
            'elephant',
            'bear',
            'zebra',
            'giraffe',
            'backpack',
            'umbrella',
            'handbag',
            'tie',
            'suitcase',
            'frisbee',
            'skis',
            'snowboard',
            'sports ball',
            'kite',
            'baseball bat',
            'baseball glove',
            'skateboard',
            'surfboard',
            'tennis racket',
            'bottle',
            'wine glass',
            'cup',
            'fork',
            'knife',
            'spoon',
            'bowl',
            'banana',
            'apple',
            'sandwich',
            'orange',
            'broccoli',
            'carrot',
            'hot dog',
            'pizza',
            'donut',
            'cake',
            'chair',
            'sofa',
            'pottedplant',
            'bed',
            'diningtable',
            'toilet',
            'tvmonitor',
            'laptop',
            'mouse',
            'remote',
            'keyboard',
            'cell phone',
            'microwave',
            'oven',
            'toaster',
            'sink',
            'refrigerator',
            'book',
            'clock',
            'vase',
            'scissors',
            'teddy bear',
            'hair drier',
            'toothbrush'
        ]

        
