import torchvision.transforms as tsf

class Config:
    def __init__(self):
        self.model_name = 'YOLOv4'

        # ------
        # Input size
        # (w, h)
        # YOLO accepts any input sizes as long as the size is multiples of 32
        # ------
        self.input_size = (672, 512)
        
        # ------
        # Classe names
        # ------
        self.class_names = [
            '0', '1', '2', '3', '4', '5', '6', '7', '8', '9'
        ]
        
        # ------
        # Anchors
        # [(w, h), ...]
        # 3 (YOLO heads) x n (num of anchors for each head)
        # big <------size------> small
        # ------
        self.anchors = [  
            # yolo default anchors for (608, 608)
            # (459, 401), (192, 243), (142, 110), (72, 146), (76, 55), (36, 75), (40, 28), (91, 36), (12, 16)
            # BBTv1 for (672, 512)
            (38, 60), (50, 43), (25, 70), (29, 54), (37, 42), (47, 31), (20, 54), (24, 37), (32, 26)
        ]


        # ------
        # Grayscale for every input image
        # ------
        self.grayscale = True

        # ------
        # Resize mode
        # "stretch" or "pad"
        # ------
        self.resize_mode = 'stretch'

        # ------
        # Normalization
        # [means, stds]
        # ------
        self.normalization = [
            (0.5, 0.5, 0.5), (0.22, 0.22, 0.22)
        ]
            

        # ------
        # YOLO structure
        # ------
        self.num_heads = 3


class DetectionConfig(Config):
    def __init__(self):
        # get general configs
        super(DetectionConfig, self).__init__()
        # threshold
        self.confidence = 0.5
        self.iou = 0.3
        self.text_font = 'simhei.ttf'
        
        # ------
        # Cuda for detection
        # ------
        self.cuda = True
        
        

class TrainingConfig(Config):
    def __init__(self):
        # get general configs
        super(TrainingConfig, self).__init__()
        self.log_dir = 'logs/'
        # ------
        # Basic hyperparameters
        # total epochs = freeze + unfreeze
        # ------
        self.initial_epoch = 1
        self.freeze_epochs = 200
        self.unfreeze_epochs = 3
        self.batch_size = 4
        self.learning_rate = 1e-3

        # ------
        # Training dictionary
        # ------
        self.train_dict = 'logs/yolo_train_dict.txt'

        # ------
        # Pre-train weight
        # set "None" to get Kaiming initialization
        # ------
        self.pretrained = None

        # ------
        # Ratio of validation
        # ------
        self.valid_ratio = 0.1

        # ------
        # Center crop
        # aspect ratio (w, h)
        # set "None" to turn off
        # ------
        self.center_crop_ratio = (4, 3)

        # ------
        # Color Jitter
        # set "None" to turn off
        # ------
        self.color_jitter = tsf.ColorJitter(
            brightness = 0.2, 
            contrast = 0.2, 
            saturation = 0, 
            hue = 0
        )

        # ------
        # Random Flip
        # mode in "all", "horizontal", "vertical"
        # set "None" to turn off
        # ------
        self.random_flip = 'all'

        # ------
        # Mosaic augmentation
        # image sizes from dataset better similar to each other
        # ------
        self.mosaic = True

        # ------
        # Label smoothing
        # ------
        self.label_smoothing = 0.

        # ------
        # Size augmentation
        # a float in [0, 1] for probability of chagning input size
        # the input size will +-(32, 32) randomly for training
        # set "None" to turn off
        # ------ 
        self.size_aug = None

        # ------
        # Threads of dataloader
        # ------
        self.num_workers = 2

        # ------
        # Consine learning rate
        # ------
        self.cos_lr = True
        
        # ------
        # Cuda for training
        # ------
        self.cuda = True
