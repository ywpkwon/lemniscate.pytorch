import numpy as np
import os
import sys
import cv2

from PIL import Image, ImageFont, ImageDraw

# copied from https://stackoverflow.com/questions/2328339/how-to-generate-n-different-colors-for-any-natural-number-n
PALETTE_HEX = [
    "#000000", "#FFFF00", "#1CE6FF", "#FF34FF", "#FF4A46", "#008941", "#006FA6", "#A30059",
    "#FFDBE5", "#7A4900", "#0000A6", "#63FFAC", "#B79762", "#004D43", "#8FB0FF", "#997D87",
    "#5A0007", "#809693", "#FEFFE6", "#1B4400", "#4FC601", "#3B5DFF", "#4A3B53", "#FF2F80",
    "#61615A", "#BA0900", "#6B7900", "#00C2A0", "#FFAA92", "#FF90C9", "#B903AA", "#D16100",
    "#DDEFFF", "#000035", "#7B4F4B", "#A1C299", "#300018", "#0AA6D8", "#013349", "#00846F",
    "#372101", "#FFB500", "#C2FFED", "#A079BF", "#CC0744", "#C0B9B2", "#C2FF99", "#001E09",
    "#00489C", "#6F0062", "#0CBD66", "#EEC3FF", "#456D75", "#B77B68", "#7A87A1", "#788D66",
    "#885578", "#FAD09F", "#FF8A9A", "#D157A0", "#BEC459", "#456648", "#0086ED", "#886F4C",
    "#34362D", "#B4A8BD", "#00A6AA", "#452C2C", "#636375", "#A3C8C9", "#FF913F", "#938A81",
    "#575329", "#00FECF", "#B05B6F", "#8CD0FF", "#3B9700", "#04F757", "#C8A1A1", "#1E6E00",
    "#7900D7", "#A77500", "#6367A9", "#A05837", "#6B002C", "#772600", "#D790FF", "#9B9700",
    "#549E79", "#FFF69F", "#201625", "#72418F", "#BC23FF", "#99ADC0", "#3A2465", "#922329",
    "#5B4534", "#FDE8DC", "#404E55", "#0089A3", "#CB7E98", "#A4E804", "#324E72", "#6A3A4C",
    "#83AB58", "#001C1E", "#D1F7CE", "#004B28", "#C8D0F6", "#A3A489", "#806C66", "#222800",
    "#BF5650", "#E83000", "#66796D", "#DA007C", "#FF1A59", "#8ADBB4", "#1E0200", "#5B4E51",
    "#C895C5", "#320033", "#FF6832", "#66E1D3", "#CFCDAC", "#D0AC94",
    "#7ED379", "#012C58"]


def _parse_hex_color(s):
    r = int(s[1:3], 16)
    g = int(s[3:5], 16)
    b = int(s[5:7], 16)
    return (r, g, b)


PALETTE_RGB = np.asarray(
    list(map(_parse_hex_color, PALETTE_HEX)),
    dtype='int32')

PALETTE_BGR = PALETTE_RGB[:, ::-1]

class Bbox():
    def __init__(self, coord):
        self.X = coord[0]
        self.Y = coord[1]
        self.W = coord[2]
        self.H = coord[3]

def bbox_iou(boxA, boxB):
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    
    # compute the area of intersection rectangle
    interArea = abs(max((xB - xA, 0)) * max((yB - yA), 0))
    if interArea == 0:
        return 0
    
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = abs((boxA[2] - boxA[0]) * (boxA[3] - boxA[1]))
    boxBArea = abs((boxB[2] - boxB[0]) * (boxB[3] - boxB[1]))
    
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    
    # return the intersection over union value
    return iou

def bboxes_iou(bboxes1, bboxes2):
    """ bboxes1: N x 4
        bboxes2: M x 4
        return: N x M
    """
    #import pdb; pdb.set_trace()
    if bboxes1.size == 0 or bboxes2.size == 0:
        return np.ndarray((bboxes1.size, bboxes2.size))
    # bboxes1
    width1 = bboxes1[:,2] - bboxes1[:,0]
    height1 = bboxes1[:,3] - bboxes1[:,1]
    area1 = (width1 * height1).reshape(-1, 1) # N x 1
    # bboxes2
    width2 = bboxes2[:,2] - bboxes2[:,0]
    height2 = bboxes2[:,3] - bboxes2[:,1]
    area2 = width2 * height2 # M
    # intersection
    xmin = np.maximum(bboxes1[:,0].reshape(-1, 1), bboxes2[:,0])
    ymin = np.maximum(bboxes1[:,1].reshape(-1, 1), bboxes2[:,1])
    xmax = np.minimum(bboxes1[:,2].reshape(-1, 1), bboxes2[:,2])
    ymax = np.minimum(bboxes1[:,3].reshape(-1, 1), bboxes2[:,3])
    width = np.maximum(xmax - xmin, 0)
    height = np.maximum(ymax - ymin, 0)
    area = width * height # N x M
    iou = area / (area1 + area2 - area)
    return iou

def bboxes_nms(bboxes, iou_threshold=0.5):
    if bboxes.size == 0: return bboxes
    #print(bboxes)
    bboxes = bboxes_sort(bboxes)
    id = bboxes[:,-1].astype(np.int32) # class
    keep = np.ones(len(bboxes), dtype=np.bool)
    for i, bbox in enumerate(bboxes[:-1]):
        iou = bboxes_iou(bbox.reshape(1,-1), bboxes[i+1:])
        suppress = np.logical_and(iou > iou_threshold, id[i] == id[i+1:])
        keep[i+1:] = np.logical_and(keep[i+1:], ~suppress)
    return bboxes[keep]

def bboxes_sort(bboxes):
    if bboxes.size == 0: return bboxes
    idx = np.argsort(-bboxes[:,-2]) # objectness
    bboxes = bboxes[idx]
    return bboxes

def draw_box(image, box, color):
    image = Image.fromarray(np.uint8(image))
    draw = ImageDraw.Draw(image, 'RGBA')

    xmin, ymin, xmax, ymax = box
    (left, right, top, bottom) = (xmin, xmax, ymin, ymax)
    # draw bounding box
    draw.line([(left, top), (left, bottom), (right, bottom), (right, top), (left, top)],
              width=2, fill=color)

    return np.asarray(image)

def coordinates_to_pts(coordinates):
    # [[x1, y1], [x2, y2]]
    pts = np.array([(coord['x'], coord['y']) for coord in coordinates])
    pts = pts.round().astype(np.int32)
    return pts

def xywh_to_coord(box, src_img_size=(1, 1), dst_img_size=(1, 1)):
    x1 = box[0] / src_img_size[0] * dst_img_size[0]
    y1 = box[1] / src_img_size[1] * dst_img_size[1]
    x2 = (box[0] + box[2]) / src_img_size[0] * dst_img_size[0]
    y2 = (box[1] + box[3]) / src_img_size[1] * dst_img_size[1]
    
    return [x1, y1, x2, y2]


def draw_text(img, x, y, display_str,
              font=cv2.FONT_HERSHEY_SIMPLEX,
              font_scale=1,
              fgcolor=(255, 255, 255), bgcolor=(0, 0, 0),
              thickness=1, alpha=125, align='center', valign='bottom'):
    text_width, text_height = cv2.getTextSize(display_str, font, font_scale, thickness)[0]
    margin = np.ceil(0.05 * text_height)

    if align == 'center':
        text_left = x - text_width * 0.5
        text_right = x + text_width * 0.5
    elif align == 'left':
        text_left = x
        text_right = x + text_width

    if valign == 'center':
        text_top = y - text_height * 0.5
        text_bottom = y + text_height * 0.5
    elif valign == 'bottom':
        text_top = y - text_height
        text_bottom = y
    elif valign == 'top':
        text_top = y
        text_bottom = y + text_height

    text_top, text_bottom, text_left, text_right = \
        int(text_top), int(text_bottom), int(text_left), int(text_right)

    img = cv2.rectangle(img, (text_left, text_top), (text_right, text_bottom), bgcolor, -1)
    img = cv2.putText(img, display_str, (text_left, text_bottom), font, font_scale, fgcolor, thickness, cv2.LINE_AA)
    # cv2.rectangle(img, (x - w, y - h), (x + w, y + h), clr, width)
    return img

def draw_text_pil(img, x, y, display_str,
                  font,
                  fgcolor=(255, 255, 255), bgcolor=(0, 0, 0),
                  align='center', valign='bottom'):

    # Make into PIL Image
    imgp = Image.fromarray(img)
    draw = ImageDraw.Draw(imgp)
    text_width, text_height = draw.textsize(display_str, font=font)
    margin = np.ceil(0.05 * text_height)

    if align == 'center':
        text_left = x - text_width * 0.5
        text_right = x + text_width * 0.5
    elif align == 'left':
        text_left = x
        text_right = x + text_width
    elif align == 'right':
        text_left = x - text_width
        text_right = x

    if valign == 'center':
        text_top = y - text_height * 0.5
        text_bottom = y + text_height * 0.5
    elif valign == 'bottom':
        text_top = y - text_height
        text_bottom = y
    elif valign == 'top':
        text_top = y
        text_bottom = y + text_height

    text_top, text_bottom, text_left, text_right = \
        int(text_top), int(text_bottom), int(text_left), int(text_right)

    # img = cv2.rectangle(img, (text_left, text_top), (text_right, text_bottom), bgcolor, -1)
    # img = cv2.putText(img, display_str, (text_left, text_bottom), font, font_scale, fgcolor, thickness, cv2.LINE_AA)
    draw.text((text_left, text_top), display_str, fgcolor, font=font)
    # cv2.rectangle(img, (x - w, y - h), (x + w, y + h), clr, width)

    img = np.array(imgp)
    return img

def draw_light(img, label, x1, y1, x2, y2, brake_width=0.4, height_ratio=0.2, margin_ratio=0.05):

    height = min(20, int((y2 - y1) * height_ratio))
    margin = min(10, int((y2 - y1) * margin_ratio))

    lr_width = (1 - brake_width) / 2.0

    m1 = int(x1 * (1 - lr_width) + x2 * lr_width)
    m2 = int(x1 * lr_width + x2 * (1 - lr_width))

    gap = int(margin / 2)
    left_x = (x1, m1 - gap)
    brake_x = (m1 + gap, m2 - margin)
    right_x = (m2 + gap, x2)

    lr_bg = (20, 20, 20)
    lr_fg = (250, 218, 94)
    brake_bg = (20, 20, 20)
    brake_fg = (155, 28, 49)

    l_cl = lr_fg if "L" in label else lr_bg
    r_cl = lr_fg if "R" in label else lr_bg
    b_cl = brake_fg if "B" in label else brake_bg

    img = cv2.rectangle(img,
                        (left_x[0], y1 - height - margin),
                        (left_x[1], y1 - margin), l_cl, -1)

    img = cv2.rectangle(img,
                        (brake_x[0], y1 - height - margin),
                        (brake_x[1], y1 - margin), b_cl, -1)

    img = cv2.rectangle(img,
                        (right_x[0], y1 - height - margin),
                        (right_x[1], y1 - margin), r_cl, -1)
    return img


PANEL = [
    ("s", "toggle play/frame"),
    ("j,k", "play faster/slower"),
    ("h,l", "play forward/backward"),
    ("q", "quit"),
]


def draw_panel(panel, font, help=PANEL):
    # panel = np.zeros((180, 500, 3), np.uint8)

    # Make into PIL Image
    # im_p = Image.fromarray(panel)

    # just estimate size using a dummy image
    im = Image.new("RGB", (50, 50))
    draw = ImageDraw.Draw(im)

    keys = "\n".join([p[0] for p in PANEL])
    expl = "\n".join([p[1] for p in PANEL])
    width_k, height_k = draw.textsize(keys, font=font)
    width_e, height_e = draw.textsize(expl, font=font)

    height, width, _ = panel.shape
    panel = draw_text_pil(
        panel,
        width - width_e - width_k - 5,
        height - height_k - 5,
        keys,
        font,
        align="right",
        valign="top",
    )
    panel = draw_text_pil(
        panel,
        width - width_e - 5,
        height - height_e - 5,
        expl,
        font,
        align="left",
        valign="top",
        fgcolor=(249, 146, 69),
    )
    return panel
