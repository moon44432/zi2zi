# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import absolute_import

import tensorflow as tf
import os
import argparse
from model.unet import UNet
from model.utils import compile_frames_to_gif

parser = argparse.ArgumentParser(description='Inference for unseen data')
parser.add_argument('--model_dir', dest='model_dir', required=True,
                    help='directory that saves the model checkpoints')
parser.add_argument('--batch_size', dest='batch_size', type=int, default=16, help='number of examples in batch')
parser.add_argument('--source_obj', dest='source_obj', type=str, required=True, help='the source images for inference')
parser.add_argument('--embedding_id', default='embedding_id', type=str, help='embeddings involved')
parser.add_argument('--save_dir', default='save_dir', type=str, help='path to save inferred images')
parser.add_argument('--inst_norm', dest='inst_norm', type=int, default=0,
                    help='use conditional instance normalization in your model')

args = parser.parse_args()


def main(_):
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(config=config) as sess:
        model = UNet(batch_size=args.batch_size)
        model.register_session(sess)
        model.build_model(is_training=False, inst_norm=args.inst_norm)

        model.test(model_dir=args.model_dir, source_obj=args.source_obj, embedding_id=args.embedding_id,
                   save_dir=args.save_dir)


if __name__ == '__main__':
    tf.app.run()
