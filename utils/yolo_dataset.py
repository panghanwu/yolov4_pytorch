from PIL import Image
from torch.utils.data import Dataset
from torchvision import ops
import torchvision.transforms as tsf
import numpy as np
import torch
import random


def collator(batch):
    imgs = [img[0].unsqueeze(0) for img in batch]
    annos = [anno[1] for anno in batch]
    imgs = torch.cat(imgs)
    return imgs, annos


class YOLODataset(Dataset):
    """
    yolo_line: img_path xmin,ymin,xmax,ymax,label xmin,ymin,...
    output_size: int for square output
    : probability
    color_jitter: transforms.ColorJitter(brightness, contrast, saturation, hue)
    """
    def __init__(
            self,
            yolo_line_list,
            output_size = (672, 512),
            batch_size = 4,
            normalization:list = None,
            resize_mode = 'stretch',
            grayscale = False,
            size_augmentation:float = None,
            center_crop_ratio:tuple = None,
            random_flip = None,
            color_jitter = None,
            mosaic = False
        ):

        self.data = yolo_line_list
        self.size = output_size
        self.grayscale = grayscale
        self.crop_r = center_crop_ratio
        self.resize_mode = resize_mode
        self.size_aug = size_augmentation
        self.random_flip = random_flip
        self.color_jitter = color_jitter
        self.normalization = normalization
        self.mosaic = mosaic
        self.batch_size = batch_size
        
        self.counter = 0  # for counting a batch
        self.sign = None  # for size augmentation


    def decode_line(self, yolo_line):
        yolo_line = yolo_line.split()
        yolo_line[1:] = [s.split(',') for s in yolo_line[1:]]
        return yolo_line


    def normalize_bbox(self, tensor_bboxes, size):
        w, h = size
        tensor_bboxes[:,[0,2]] = tensor_bboxes[:,[0,2]] / w
        tensor_bboxes[:,[1,3]] = tensor_bboxes[:,[1,3]] / h
        return tensor_bboxes


    def center_crop(self, ratio:tuple, tensor_img, tensor_bboxes, rm_thres=16):
        """
        tensor_bboxes: [left, top, right, bottom]
        """
        _, h, w = tensor_img.size()
        r = ratio[0] / ratio[1]
        size = (int(min(w/r, h)), int(min(h*r, w)))  # (h, w)
        # img
        tensor_img = tsf.CenterCrop(size)(tensor_img)
        # align offset bbox
        x_offset, y_offset = (w-size[1])/2, (h-size[0])/2
        offsets = torch.Tensor([x_offset, y_offset, x_offset, y_offset])

        if len(tensor_bboxes):
            tensor_bboxes[:,[0,1,2,3]] = tensor_bboxes[:,[0,1,2,3]] - offsets
            # align bboxes out of the image
            tensor_bboxes[:,0][tensor_bboxes[:,0] < 0] = 0  # left side
            tensor_bboxes[:,1][tensor_bboxes[:,1] < 0] = 0  # top side
            tensor_bboxes[:,2][tensor_bboxes[:,2] > size[1]] = size[1]  # right side
            tensor_bboxes[:,3][tensor_bboxes[:,3] > size[0]] = size[0]  # bottom side
            # remove bboxes out of the image
            keep_x = tensor_bboxes[:,2] - tensor_bboxes[:,0] > rm_thres
            keep_y = tensor_bboxes[:,3] - tensor_bboxes[:,1] > rm_thres
            keep_mask = torch.logical_and(keep_x, keep_y)
            tensor_bboxes = tensor_bboxes[keep_mask]
        return tensor_img, tensor_bboxes


    def resize(self, size:tuple, tensor_img, tensor_bboxes, mode='stretch'):
        """
        size: tuple in (w, h)
        """
        assert mode in ['stretch', 'pad']
        _, h, w = tensor_img.size()

        if mode=='stretch' and len(tensor_bboxes):
            # bbox size and size align
            tensor_bboxes[:,[0,2]] = tensor_bboxes[:,[0,2]]*size[0] / w  # x resize
            tensor_bboxes[:,[1,3]] = tensor_bboxes[:,[1,3]]*size[1] / h  # y resize
        elif mode == 'pad':
            ratio = min(size[0]/w, size[1]/h)
            # pad img
            pads = (int(size[0]/ratio-w)//2, int(size[1]/ratio-h)//2)  # (w, h)
            tensor_img = tsf.Pad(pads)(tensor_img)
            
            if len(tensor_bboxes):
                # bbox center align
                tensor_bboxes[:,0] = tensor_bboxes[:,0] + pads[0]
                tensor_bboxes[:,1] = tensor_bboxes[:,1] + pads[1]
                
                # bbox size align
                tensor_bboxes[:,[0,2]] = tensor_bboxes[:,[0,2]]*ratio  # x resize
                tensor_bboxes[:,[1,3]] = tensor_bboxes[:,[1,3]]*ratio  # y resize

        # resize img
        tensor_img = tsf.Resize((size[1],size[0]))(tensor_img)  # (h, w)
        return tensor_img, tensor_bboxes
    
    
    def flip_randomly(self, tensor_img, tensor_bboxes, mode='all'):
        _, h, w = tensor_img.size()
        assert mode in ['all', 'horizontal', 'vertical']

        # horizontal 0.5
        if mode in ['all', 'horizontal']:
            if random.getrandbits(1):
                tensor_img = tsf.RandomHorizontalFlip(1)(tensor_img)
                # bbox center align
                if len(tensor_bboxes):
                    tensor_bboxes[:,0] = w - tensor_bboxes[:,0]
            
        # vertical 0.5
        if mode in ['all', 'vertical']:
            if random.getrandbits(1):
                tensor_img = tsf.RandomVerticalFlip(1)(tensor_img)
                # bbox center align
                if len(tensor_bboxes):
                    tensor_bboxes[:,1] = h - tensor_bboxes[:,1]

        return tensor_img, tensor_bboxes


    def get_mosaic(self, imgs:list, bboxes:list, rm_thres=16):
        """
        Every image size shold be same.
        imgs: list of 4 PIL images
        bboxes: list of 4 bbox list
        rm_thres: threshold for removing too small size bboxes
        """
        # get longest edges as size of new image
        new_size = imgs[0].size
        rs =  np.random.uniform(0.5, 1.5, [2])  # random shift
        center = (int(new_size[0]*rs[0]/2), int(new_size[1]*rs[1]/2))
        
        # crop each image
        imgs[1] = imgs[1].crop((center[0], 0, new_size[0], center[1]))
        imgs[2] = imgs[2].crop((0, center[1], center[0], new_size[1]))
        imgs[3] = imgs[3].crop((center[0], center[1], new_size[0], new_size[1]))
        
        # paste other image to main image
        imgs[0].paste(imgs[1], (center[0],0))
        imgs[0].paste(imgs[2], (0,center[1]))
        imgs[0].paste(imgs[3], (center[0],center[1]))

        # align bboxes
        b0 = np.array(bboxes[0])
        b1 = np.array(bboxes[1])
        b2 = np.array(bboxes[2])
        b3 = np.array(bboxes[3])

        box_list = list()
        # bbox 0
        if b0.any():
            b0[:,0][b0[:,0] > center[0]] = center[0]
            b0[:,2][b0[:,2] > center[0]] = center[0]
            b0[:,1][b0[:,1] > center[1]] = center[1]
            b0[:,3][b0[:,3] > center[1]] = center[1]
            box_list.append(b0)
        # bbox 1
        if b1.any():
            b1[:,0][b1[:,0] < center[0]] = center[0]
            b1[:,2][b1[:,2] < center[0]] = center[0]
            b1[:,1][b1[:,1] > center[1]] = center[1]
            b1[:,3][b1[:,3] > center[1]] = center[1]
            box_list.append(b1)
        # bbox 2
        if b2.any():
            b2[:,0][b2[:,0] > center[0]] = center[0]
            b2[:,2][b2[:,2] > center[0]] = center[0]
            b2[:,1][b2[:,1] < center[1]] = center[1]
            b2[:,3][b2[:,3] < center[1]] = center[1]
            box_list.append(b2)
        # bbox 3
        if b3.any():
            b3[:,0][b3[:,0] < center[0]] = center[0]
            b3[:,2][b3[:,2] < center[0]] = center[0]
            b3[:,1][b3[:,1] < center[1]] = center[1]
            b3[:,3][b3[:,3] < center[1]] = center[1]
            box_list.append(b3)

        if box_list:
            mosaic_bboxes = np.concatenate(box_list)
            # remove no area bbox
            keep_x = mosaic_bboxes[:,2] - mosaic_bboxes[:,0] > rm_thres
            keep_y = mosaic_bboxes[:,3] - mosaic_bboxes[:,1] > rm_thres
            keep_mask = np.logical_and(keep_x, keep_y)
            mosaic_bboxes = mosaic_bboxes[keep_mask]
        else:
            mosaic_bboxes = list()
        return imgs[0], mosaic_bboxes


    def __getitem__(self, index:int): 
        anno = self.decode_line(self.data[index])
        
        img_path = anno[0]  # read img
        img = Image.open(img_path).convert('RGB')  # make sure it is 3 channels
        
        bboxes = [list(map(int,map(float,b))) for b in anno[1:]]  # get bboxes
        
        # ------
        # Mosaic augmentation
        # ------
        if self.mosaic:
            # randomly get 3 others
            lines = random.sample(self.data, 3)  # get lines
            annos_list = list(map(self.decode_line, lines))  # decode lines
            # img1 + other imgs
            img_list = [img]
            bbox_list = [bboxes]
            for anno in annos_list:
                img_list.append(Image.open(anno[0]).convert('RGB'))  # img
                bbox_list.append([list(map(int,map(float,b))) for b in anno[1:]])
            
            # get mosaic from 4 imgs
            img, bboxes = self.get_mosaic(img_list, bbox_list, rm_thres=16)

        # ------
        # To tensor, img grayscale, 
        # ------
        img = tsf.ToTensor()(img)  # to tensors
        if self.grayscale:
            tsf.Grayscale(3)
        
        bboxes = torch.Tensor(bboxes)  # to tensors
        
        # ------
        # Center Crop by aspect ratio
        # ------
        if self.crop_r is not None:
            img, bboxes = self.center_crop(self.crop_r, img, bboxes)

        # ------
        # change bbox format
        # from (left, top, right, bottom) to (x, y, w, h)
        # ------
        if len(bboxes):
            bboxes[:, :4] = ops.box_convert(bboxes[:,:4], in_fmt='xyxy', out_fmt='cxcywh')

        # ------
        # Resize an size augmentation
        # size augmentation will +- 32 depending the probability
        # ------
        if self.size_aug is not None:
            p = self.size_aug
            if self.counter%self.batch_size == 0:
                [self.sign] = random.choices([0,1,-1], [1-p,p/2,p/2])
            size = torch.Tensor(self.size).long() + self.sign*32
            self.counter += 1
        else:
            size = self.size
        img, bboxes = self.resize(size, img, bboxes, self.resize_mode)

        # ------
        # Flip and color jitter
        # ------
        if self.random_flip is not None:
            img, bboxes = self.flip_randomly(img, bboxes, self.random_flip)
        
        if self.color_jitter is not None:
            img = self.color_jitter(img)
        
        # ------
        # Normalization
        # ------
        if self.normalization is not None:
            img = tsf.Normalize(*self.normalization)(img)
        
        if len(bboxes):
            bboxes = self.normalize_bbox(bboxes, size)

        return img, bboxes


    def __len__(self):
        return len(self.data)