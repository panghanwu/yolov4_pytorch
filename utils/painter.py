from PIL import Image, ImageDraw, ImageFont
import numpy as np
import colorsys



def set_colors(class_n, satu=1., light=1.):
    hsv_tuples = [(x/class_n, satu, light) for x in range(class_n)]
    colors = list(map(lambda x: colorsys.hsv_to_rgb(*x),hsv_tuples))
    colors = list(map(lambda x: (int(x[0]*255),int(x[1]*255),int(x[2]*255)),colors))
    return colors



def paint_mask(mask_path, colors, alpha=.8, background=None):
    colors = np.array(colors)
    img_array = np.array(Image.open(mask_path))
    colors = np.array(colors).astype(int)
    mask_array = colors[img_array.astype(int)]
    img_mask = Image.fromarray(np.uint8(mask_array))
    img_mask.putalpha(int(255*alpha))
    
    if background == None:
        return img_mask
    else:
        img = Image.open(background)
        img_mask = img_mask.resize(img.size, Image.NEAREST)
        img.paste(img_mask, mask=img_mask)
        return img
    
    
    
def mark_bboxes(pil_image, bboxes, colors=None, show_text=True, show_cfd=True, line_thick=3, text_size=1):
    """
    bboxse: list in [[class_id, class_name, conf, [x_min, y_min, x_max, y_max]], ...]
    """
    img = pil_image.copy()
    if show_text:
        text_size = np.floor(text_size * 3e-2 * np.shape(img)[1] + 0.5).astype('int32')
        font = ImageFont.truetype(font='simhei.ttf', size=text_size)

    draw = ImageDraw.Draw(img)
    # sort bboxes by score
    for bbox in sorted(bboxes, key=lambda s: s[2]):
        # bbox
        draw.rectangle(bbox[3], fill=None, outline=colors[bbox[0]], width=line_thick)
        # tag
        if show_text:
            if show_cfd:
                tag = f'{bbox[1]} {bbox[2]:.2f}'
            else:
                tag = f'{bbox[1]}'
            text_loc = [bbox[3][0], bbox[3][1]-text_size, bbox[3][2], bbox[3][1]]
            draw.rectangle(text_loc, fill=colors[bbox[0]], outline=colors[bbox[0]], width=3)
            draw.text(text_loc, tag, fill=(0,0,0), font=font)
    del draw
    return img



def mark_centers(pil_image, bboxes, colors=None, show_text=True, point_size=6, text_size=1):
    img = pil_image.copy()
    if show_text:
        text_size = np.floor(text_size * 3e-2 * np.shape(img)[1] + 0.5).astype('int32')
        font = ImageFont.truetype(font='simhei.ttf', size=text_size)

    draw = ImageDraw.Draw(img)
    # sort bboxes by score
    for bbox in sorted(bboxes, key=lambda s: s[2]):
        # set color
        if colors is None:
            c = (255, 0, 255)
        else:
            c = colors[bbox[0]]

        # get center from bbox
        x = (bbox[3][0]+bbox[3][2]) / 2
        y = (bbox[3][1]+bbox[3][3]) / 2
        point_loc = [x-point_size, y-point_size, x+point_size, y+point_size]
        draw.ellipse(point_loc, fill=c, outline=None, width=1)
        # tag
        if show_text:
            tag = '{}'.format(bbox[1])
            text_loc = [x, y-2*point_size]
            draw.text(text_loc, tag, anchor='ms', fill=c, font=font)
    del draw
    return img