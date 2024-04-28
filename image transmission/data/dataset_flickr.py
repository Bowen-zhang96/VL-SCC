import os
import sys
import tarfile
from six.moves import urllib
import tensorflow as tf
import tensorflow_datasets as tfds
import glob
SHUFFLE_BUFFER=50
_NUM_IMAGES = {
    'train': 800,  #82783
    'validation': 112,  #40504
    'test': 112,#40775
}

def get_dataset(mode, data_dir):
    """Returns a dataset object"""
    image_path=os.path.join(data_dir,'flickr',mode,'*_L.png')
    images = sorted(glob.glob(image_path))
    filenames = tf.data.Dataset.from_tensor_slices(images)
    dataset = filenames.map(lambda x: tf.concat([tf.io.decode_png(tf.io.read_file(x)),tf.io.decode_png(tf.io.read_file(tf.strings.regex_replace(x,'L','R')))],axis=-1))
    # dataset = dataset.map(lambda x: tf.concat([x['L'],x['R']],axis=-1))
    return dataset
    # maybe_download_and_extract(data_dir) , 'R':tf.io.decode_png(tf.io.read_file(x[:-5]+'R.png'))
    # builder=tfds.builder('coco/2014',data_dir=data_dir)
    # builder.download_and_prepare()
    # if is_training:
    #     split=['train[:82783]','validation[:100]']
    # else:
    #     split='test[:500]'
    #
    # return builder.as_dataset(split=split)


def parse_record(raw_record, _mode, dtype):
    """Parse CIFAR-10 image and label from a raw record."""
    crop_size=64
    crop_size_float=tf.constant(crop_size,dtype)
    smallest_fac=tf.constant(0.75,dtype)
    biggest_fac=tf.constant(0.95,dtype)
    image = raw_record
    image=tf.cast(image, dtype)
    image=image/255.

    if _mode=='train':
        image_shape=tf.cast(tf.shape(image),dtype)
        height,width=image_shape[0],image_shape[1]
        smallest_side=tf.math.minimum(height,width)
        image_smallest_fac=crop_size_float/smallest_side
        min_fac=tf.math.maximum(smallest_fac,image_smallest_fac)
        max_fac=tf.math.maximum(min_fac,biggest_fac)
        scale=tf.random.uniform([],minval=min_fac,maxval=max_fac,dtype=dtype,seed=42,name=None)
        image=tf.image.resize(image,[tf.math.ceil(scale*height),tf.math.ceil(scale*width)])
        image=tf.image.random_crop(image,[crop_size,crop_size,6])
    else:
        image_shape = tf.cast(tf.shape(image), dtype)
        height, width = image_shape[0], image_shape[1]
        height_crop=tf.cast(tf.math.floor(height /4.)*4.,tf.int32)
        width_crop = tf.cast(tf.math.floor(width / 4.) * 4.,tf.int32)
        image=tf.image.resize(image,[height_crop,width_crop])
    return image




