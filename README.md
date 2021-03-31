# YOLOv4

## Input size constraint
YOLO accepts any input sizes as long as the size follows multiples of 32. 

It is recommended to have input sizes large than 416 pixels. Becase the `SPP` structure includes the largest pooling layer 13x13, the minimum size (width and height) is 13*32 = 416. Any input size less than 416 leads `SPP` pooling 13x13 useless, although the model can still work.
