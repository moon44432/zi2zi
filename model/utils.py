# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import os
import glob

import imageio
import scipy.misc as misc
import numpy as np
from io import BytesIO
from PIL import Image
from scipy.misc import imresize


def tight_crop_image(img, verbose=False, resize_fix=False):
    img_size = img.shape[0]
    full_white = img_size
    tmp_img = np.sum(img, axis=2)
    col_sum = np.where(full_white * 255 * 3 - np.sum(tmp_img, axis=0) > 1)
    row_sum = np.where(full_white * 255 * 3 - np.sum(tmp_img, axis=1) > 1)
    y1, y2 = row_sum[0][0], row_sum[0][-1]
    x1, x2 = col_sum[0][0], col_sum[0][-1]
    cropped_image = img[y1:y2, x1:x2]
    cropped_image_size = cropped_image.shape

    if verbose:
        print('(left x1, top y1):', (x1, y1))
        print('(right x2, bottom y2):', (x2, y2))
        print('cropped_image size:', cropped_image_size)

    if type(resize_fix) == int:
        origin_h, origin_w = cropped_image.shape[:2]
        if origin_h > origin_w:
            resize_w = int(origin_w * (resize_fix / origin_h))
            resize_h = resize_fix
        else:
            resize_h = int(origin_h * (resize_fix / origin_w))
            resize_w = resize_fix
        if verbose:
            print('resize_h:', resize_h)
            print('resize_w:', resize_w, \
                  '[origin_w %d / origin_h %d * target_h %d]' % (origin_w, origin_h, target_h))

        # resize
        cropped_image = imresize(cropped_image, (resize_h, resize_w))
        # cropped_image = normalize_image(cropped_image)
        cropped_image_size = cropped_image.shape
        if verbose:
            print('resized_image size:', cropped_image_size)

    elif type(resize_fix) == float:
        origin_h, origin_w = cropped_image.shape
        resize_h, resize_w = int(origin_h * resize_fix), int(origin_w * resize_fix)
        if resize_h > 120:
            resize_h = 120
            resize_w = int(resize_w * 120 / resize_h)
        if resize_w > 120:
            resize_w = 120
            resize_h = int(resize_h * 120 / resize_w)
        if verbose:
            print('resize_h:', resize_h)
            print('resize_w:', resize_w)

        # resize
        cropped_image = imresize(cropped_image, (resize_h, resize_w))
        # cropped_image = normalize_image(cropped_image)
        cropped_image_size = cropped_image.shape
        if verbose:
            print('resized_image size:', cropped_image_size)

    return cropped_image


def add_padding(img, image_size=256, verbose=False, pad_value=None):
    height, width = img.shape[:2]
    if pad_value is None:
        pad_value = 255
    if verbose:
        print('original cropped image size:', img.shape)

    # Adding padding of x axis - left, right
    pad_x_width = (image_size - width) // 2
    pad_x = np.full((height, pad_x_width, 3), pad_value, dtype=np.float32)
    img = np.concatenate((pad_x, img), axis=1)
    img = np.concatenate((img, pad_x), axis=1)

    width = img.shape[1]

    # Adding padding of y axis - top, bottom
    pad_y_height = (image_size - height) // 2
    pad_y = np.full((pad_y_height, width, 3), pad_value, dtype=np.float32)
    img = np.concatenate((pad_y, img), axis=0)
    img = np.concatenate((img, pad_y), axis=0)

    # Match to original image size
    width = img.shape[1]
    if img.shape[0] % 2:
        pad = np.full((1, width, 3), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=0)
    height = img.shape[0]
    if img.shape[1] % 2:
        pad = np.full((height, 1, 3), pad_value, dtype=np.float32)
        img = np.concatenate((pad, img), axis=1)

    if verbose:
        print('final image size:', img.shape)

    return img


def centering_image(img, image_size=256, verbose=False, resize_fix=160, pad_value=None):
    if pad_value is None:
        pad_value = 255
    try:
        cropped_image = tight_crop_image(img, verbose=verbose, resize_fix=resize_fix)
        centered_image = add_padding(cropped_image, image_size=image_size, verbose=verbose, pad_value=pad_value)
        return centered_image.astype(np.uint8)
    except:
        return img


def pad_seq(seq, batch_size):
    # pad the sequence to be the multiples of batch_size
    seq_len = len(seq)
    if seq_len % batch_size == 0:
        return seq
    padded = batch_size - (seq_len % batch_size)
    seq.extend(seq[:padded])
    return seq



def bytes_to_file(bytes_img):
    return BytesIO(bytes_img)


def normalize_image(img):
    """
    Make image zero centered and in between (-1, 1)
    """
    normalized = (img / 127.5) - 1.
    return normalized


def read_split_image(img):
    mat = misc.imread(img).astype(np.float)
    side = int(mat.shape[1] / 2)
    assert side * 2 == mat.shape[1]
    img_A = mat[:, :side]  # target
    img_B = mat[:, side:]  # source

    return img_A, img_B


def shift_and_resize_image(img, shift_x, shift_y, nw, nh):
    w, h, _ = img.shape
    enlarged = misc.imresize(img, [nw, nh])
    return enlarged[shift_x:shift_x + w, shift_y:shift_y + h]


def scale_back(images):
    return (images + 1.) / 2.


def merge(images, size):
    h, w = images.shape[1], images.shape[2]
    img = np.zeros((h * size[0], w * size[1], 3))
    for idx, image in enumerate(images):
        i = idx % size[1]
        j = idx // size[1]
        img[j * h:j * h + h, i * w:i * w + w, :] = image

    return img


def save_concat_images(imgs, img_path):
    concated = np.concatenate(imgs, axis=1)
    misc.imsave(img_path, concated)


def save_images(imgs, img_path, batch_size):
    idx = 0
    for img in imgs:
        step = int(img.shape[0] / batch_size)
        for img_slice in range(0, img.shape[0] - step + 1, step):
            misc.imsave(img_path + "_%04d.png" % idx, img[img_slice:img_slice + step])
            idx += 1


def compile_frames_to_gif(frame_dir, gif_file):
    frames = sorted(glob.glob(os.path.join(frame_dir, "*.png")))
    print(frames)
    images = [misc.imresize(imageio.imread(f), interp='nearest', size=0.33) for f in frames]
    imageio.mimsave(gif_file, images, duration=0.1)
    return gif_file
