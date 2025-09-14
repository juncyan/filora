import math
import numpy as np
import cv2
from skimage.measure import label,regionprops

def one_hot_it(label, label_info):
    semantic_map = []
    for info in label_info:
        color = label_info[info].values
        # print("label:\n", label.shape,label)
        # print("color:\n", color)
        equality = np.equal(label, color)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    return np.stack(semantic_map, axis=-1)

def rbox2poly(rrect):
    '''
    :param rrect: [x_ctr, y_ctr, w, h, angle]
    :return:  [x0, y0, x1, y1, x2, y2, x3, y3]
    '''

    x_ctr, y_ctr, width, height, angle = rrect[:5]
    angle = np.pi * angle / 180.
    t1_x, t1_y, br_x, br_y = -width/2, -height/2, width/2, height/2
    rect = np.array([[t1_x, br_x, br_x, t1_x], [t1_y, t1_y, br_y, br_y]])
    r = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = r.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    return [x0, y0, x1, y1, x2, y2, x3, y3]

def poly2rbox(bbox):
    '''
    :param bbox: [x0, y0, x1, y1, x2, y2, x3, y3]
    :return: [x_ctr, y_ctr, w, h, angle]
    '''

    bbox = np.array(bbox, np.float32)
    bbox = np.reshape(bbox, newshape=(2,4), order='F')
    # angle = math.atan2(-(bbox[0,1]-bbox[0,0]),bbox[1,1]-bbox[1,0])
    angle = math.atan2(bbox[1,2] - bbox[1,1], bbox[0,2] - bbox[0,1])
    if angle > -math.pi /2 and angle < 0:
        angle += math.pi / 2
    elif angle > math.pi / 2 and angle < math.pi:
        angle -= math.pi / 2

    center = [[0], [0]]
    for i in range(4):
        center[0] += bbox[0, i]
        center[1] += bbox[1, i]
    center = np.array(center, np.float32) / 4.0

    rotate = np.array([[np.cos(angle), -np.sin(angle)],
                       [np.sin(angle), np.cos(angle)]], np.float32)
    normalized = np.matmul(rotate.transpose(), bbox - center)
    xmin = np.min(normalized[0, :])
    xmax = np.max(normalized[0, :])
    ymin = np.min(normalized[1, :])
    ymax = np.max(normalized[1, :])
    w = xmax - xmin + 1
    h = ymax - ymin + 1
    return [float(center[0]), float(center[1]), w, h, angle]

def random_color_jitter(cv_img, saturation_range, brightness_range, contrast_range, u=0.5):
    def saturation_jitter(cv_img, jitter_range):
        """
        调节图像饱和度
        Args:
            cv_img(numpy.ndarray): 输入图像
            jitter_range(float): 调节程度，0-1
        Returns:
            饱和度调整后的图像
        """
        greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        greyMat = greyMat[:, :, None] * np.ones(3, dtype=int)[None, None, :]
        cv_img = cv_img.astype(np.float32)
        cv_img = cv_img * (1 - jitter_range) + jitter_range * greyMat
        cv_img = np.where(cv_img > 255, 255, cv_img)
        cv_img = cv_img.astype(np.uint8)
        return cv_img

    def brightness_jitter(cv_img, jitter_range):
        """
        调节图像亮度
        Args:
            cv_img(numpy.ndarray): 输入图像
            jitter_range(float): 调节程度，0-1
        Returns:
            亮度调整后的图像
        """
        cv_img = cv_img.astype(np.float32)
        cv_img = cv_img * (1.0 - jitter_range)
        cv_img = np.where(cv_img > 255, 255, cv_img)
        cv_img = cv_img.astype(np.uint8)
        return cv_img

    def contrast_jitter(cv_img, jitter_range):
        """
        调节图像对比度
        Args:
            cv_img(numpy.ndarray): 输入图像
            jitter_range(float): 调节程度，0-1
        Returns:
            对比度调整后的图像
        """
        greyMat = cv2.cvtColor(cv_img, cv2.COLOR_BGR2GRAY)
        mean = np.mean(greyMat)
        cv_img = cv_img.astype(np.float32)
        cv_img = cv_img * (1 - jitter_range) + jitter_range * mean
        cv_img = np.where(cv_img > 255, 255, cv_img)
        cv_img = cv_img.astype(np.uint8)
        return cv_img
    """
    图像亮度、饱和度、对比度调节，在调整范围内随机获得调节比例，并随机顺序叠加三种效果
    Args:
        cv_img(numpy.ndarray): 输入图像
        saturation_range(float): 饱和对调节范围，0-1
        brightness_range(float): 亮度调节范围，0-1
        contrast_range(float): 对比度调节范围，0-1
    Returns:
        亮度、饱和度、对比度调整后图像
    """
    if np.random.random() < u:
        saturation_ratio = np.random.uniform(-saturation_range, saturation_range)
        brightness_ratio = np.random.uniform(-brightness_range, brightness_range)
        contrast_ratio = np.random.uniform(-contrast_range, contrast_range)
        order = [0, 1, 2]
        np.random.shuffle(order)
        for i in range(3):
            if order[i] == 0:
                cv_img = saturation_jitter(cv_img, saturation_ratio)
            if order[i] == 1:
                cv_img = brightness_jitter(cv_img, brightness_ratio)
            if order[i] == 2:
                cv_img = contrast_jitter(cv_img, contrast_ratio)
        return cv_img
    return cv_img


def randomHueSaturationValue(image, hue_shift_limit=(-180, 180),sat_shift_limit=(-255, 255),val_shift_limit=(-255, 255), u=0.5):
    if np.random.random() < u:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(image)
        hue_shift = np.random.randint(hue_shift_limit[0], hue_shift_limit[1] + 1)
        hue_shift = np.uint8(hue_shift)
        h += hue_shift
        sat_shift = np.random.uniform(sat_shift_limit[0], sat_shift_limit[1])
        s = cv2.add(s, sat_shift)
        val_shift = np.random.uniform(val_shift_limit[0], val_shift_limit[1])
        v = cv2.add(v, val_shift)
        image = cv2.merge((h, s, v))
        image = cv2.cvtColor(image, cv2.COLOR_HSV2BGR)
    return image


def randomShiftScaleRotate(image, mask, shift_limit=(-0.0, 0.0),scale_limit=(-0.0, 0.0),rotate_limit=(-0.0, 0.0),
                           aspect_limit=(-0.0, 0.0),borderMode=cv2.BORDER_REFLECT, u=0.5):
    if np.random.random() < u:
        height, width, channel = image.shape
        angle = np.random.uniform(rotate_limit[0], rotate_limit[1])
        scale = np.random.uniform(1 + scale_limit[0], 1 + scale_limit[1])
        aspect = np.random.uniform(1 + aspect_limit[0], 1 + aspect_limit[1])
        sx = scale * aspect / (aspect ** 0.5)
        sy = scale / (aspect ** 0.5)
        dx = round(np.random.uniform(shift_limit[0], shift_limit[1]) * width)
        dy = round(np.random.uniform(shift_limit[0], shift_limit[1]) * height)

        cc = np.math.cos(angle / 180 * np.math.pi) * sx
        ss = np.math.sin(angle / 180 * np.math.pi) * sy
        rotate_matrix = np.array([[cc, -ss], [ss, cc]])

        box0 = np.array([[0, 0], [width, 0], [width, height], [0, height], ])
        box1 = box0 - np.array([width / 2, height / 2])
        box1 = np.dot(box1, rotate_matrix.T) + np.array([width / 2 + dx, height / 2 + dy])

        box0 = box0.astype(np.float32)
        box1 = box1.astype(np.float32)
        mat = cv2.getPerspectiveTransform(box0, box1)
        image = cv2.warpPerspective(image, mat, (width, height), flags=cv2.INTER_LINEAR, borderMode=borderMode,borderValue=(0, 0,0,))
        mask = cv2.warpPerspective(mask, mat, (width, height), flags=cv2.INTER_NEAREST, borderMode=borderMode,borderValue=(0, 0, 0,))
    return image, mask


def randomHorizontalFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 1)
        mask = cv2.flip(mask, 1)
    return image, mask


def randomVerticleFlip(image, mask, u=0.5):
    if np.random.random() < u:
        image = cv2.flip(image, 0)
        mask = cv2.flip(mask, 0)
    return image, mask


def randomRotate90(image, mask, u=0.5):
    if np.random.random() < u:
        angle = np.random.randint(1,4)
        for i in range(angle):
            image = np.rot90(image)
            mask = np.rot90(mask)
        # return image, mask
    return image, mask


def resize(image, gt,insize, outsize):
    x = np.random.randint(-128, 128)
    y = np.random.randint(-128, 128)
    if x < 0:
        if y < 0:
            image = image[0:x, 0:y, :]
            gt = gt[0:x, 0:y]
        else:
            image = image[0:x, y:insize, :]
            gt = gt[0:x, y:insize]
    else:
        if y < 0:
            image = image[x:insize, 0:y, :]
            gt = gt[x:insize, 0:y]
        else:
            image = image[x:insize, y:insize, :]
            gt = gt[x:insize, y:insize]
    image = cv2.resize(image, (outsize, outsize), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (outsize, outsize), interpolation=cv2.INTER_NEAREST)

    return image,gt


def grade(img):
    x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    absX = cv2.convertScaleAbs(x)
    absY = cv2.convertScaleAbs(y)
    dst = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
    mi = np.min(dst)
    ma = np.max(dst)
    res = (dst - mi) / (0.000000001 + (ma - mi))
    res[np.isnan(res)] = 0
    return res


def Raster2Bbox(rasterlab):
    bbox=[]
    lab = rasterlab.copy()
    labeled_img, num = label(255 * lab, background=0, return_num=True, connectivity=1)
    bboxs = regionprops(labeled_img)
    for index in range(num):
        if bboxs[index].area>400:
            bbox.append(bboxs[index].bbox)
    return bbox