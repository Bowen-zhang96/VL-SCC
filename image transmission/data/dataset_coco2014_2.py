import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import tensorflow_datasets as tfds
SHUFFLE_BUFFER=50
_NUM_IMAGES = {
    'train': 82783,  #82783
    'validation': 100,  #40504
    'test': 2000,#40775
}

def get_dataset(is_training, data_dir):
    """Returns a dataset object"""
    # maybe_download_and_extract(data_dir)
    builder=tfds.builder('coco/2014',data_dir=data_dir)
    builder.download_and_prepare()
    if is_training:
        split=['train[:82783]','validation[:1000]','train[:100]']
    else:
        split='test[:2000]'

    return builder.as_dataset(split=split)


def parse_record(raw_record, _mode, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
   # crop_size=np.random.randint(64,256,1)
   # crop_size=crop_size[0]
    crop_size=128
    crop_size_float=tf.constant(crop_size,dtype)
    smallest_fac=tf.constant(0.75,dtype)
    biggest_fac=tf.constant(0.95,dtype)
    image=raw_record['image']
    image=tf.cast(image, dtype)
    image=image/255.
    if _mode=='train':
        image_shape = tf.cast(tf.shape(image), dtype)
        height, width = image_shape[0], image_shape[1]
        height_crop=tf.cast(tf.math.floor(height /32.)*32.,tf.int32)
        width_crop = tf.cast(tf.math.floor(width / 32.) * 32.,tf.int32)
        image=tf.image.resize(image,[224,224])
    else:
        image_shape = tf.cast(tf.shape(image), dtype)
        height, width = image_shape[0], image_shape[1]
        height_crop=tf.cast(tf.math.floor(height /32.)*32.,tf.int32)
        width_crop = tf.cast(tf.math.floor(width / 32.) * 32.,tf.int32)
        image=tf.image.resize(image,[224,224])
    return image




