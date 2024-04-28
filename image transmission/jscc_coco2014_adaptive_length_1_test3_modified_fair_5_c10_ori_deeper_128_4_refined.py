import os
os.environ["CUDA_VISIBLE_DEVICES"]="5"
import glob
import time
from datetime import datetime
import tensorflow as tf
import numpy as np
import configargparse
from tensorflow.keras import layers
import tensorflow_compression as tfc
import data_new.dataset_cifar10
import data_new.dataset_imagenet
import data_new.dataset_kodak
import data_new.dataset_coco2014_1
import urllib.request
from PIL import Image

_LPIPS_URL = "http://rail.eecs.berkeley.edu/models/lpips/net-lin_alex_v0.1.pb"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

DATASETS = {
    "cifar": data_new.dataset_cifar10,
    "imagenet": data_new.dataset_imagenet,
    "coco2014": data_new.dataset_coco2014_1,
    'kodak':data_new.dataset_kodak
}


class NBatchLogger(tf.keras.callbacks.Callback):
    """
    A Logger that log average performance per `display` steps.
    """

    def __init__(self, display):
        super(NBatchLogger, self).__init__()
        self.step = 0
        self.display = display
        self.metric_cache = {}
        self._start_time = time.time()

    def on_batch_end(self, batch, logs={}):
        self.step += 1
        for k in self.params["metrics"]:
            if k in logs:
                self.metric_cache[k] = self.metric_cache.get(k, 0) + logs[k]

        if self.step % self.display == 0:
            cur_time = time.time()
            duration = cur_time - self._start_time
            self._start_time = cur_time
            sec_per_step = duration / self.display

            metrics_log = ""
            for (k, v) in self.metric_cache.items():
                val = v / self.display
                if abs(val) > 1e-3:
                    metrics_log += " - %s: %.4f" % (k, val)
                else:
                    metrics_log += " - %s: %.4e" % (k, val)
            print(
                "{} step: {}/{} {} - {:3f} sec/step".format(
                    datetime.now(),
                    self.step,
                    self.params["steps"],
                    metrics_log,
                    sec_per_step,
                )
            )
            self.metric_cache.clear()


class PSNRsVar(tf.keras.metrics.Metric):
    """Calculate the variance of a distribution of PSNRs across batches

    """

    def __init__(self, name="variance", **kwargs):
        super(PSNRsVar, self).__init__(name=name, **kwargs)
        self.count = self.add_weight(name="count", shape=(), initializer="zeros")
        self.mean = self.add_weight(name="mean", shape=(), initializer="zeros")
        self.var = self.add_weight(name="M2", shape=(), initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnrs = tf.image.psnr(y_true, y_pred, max_val=1.0)
        samples = tf.cast(psnrs, self.dtype)
        batch_count = tf.size(samples)
        batch_count = tf.cast(batch_count, self.dtype)
        batch_mean = tf.math.reduce_mean(samples)
        batch_var = tf.math.reduce_variance(samples)

        # compute new values for variables
        new_count = self.count + batch_count
        new_mean = (self.count * self.mean + batch_count * batch_mean) / (
            self.count + batch_count
        )
        new_var = (
            (self.count * (self.var + tf.square(self.mean - new_mean)))
            + (batch_count * (batch_var + tf.square(batch_mean - new_mean)))
        ) / (self.count + batch_count)

        self.count.assign(new_count)
        self.mean.assign(new_mean)
        self.var.assign(new_var)

    def result(self):
        return self.var

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.count.assign(np.zeros(self.count.shape))
        self.mean.assign(np.zeros(self.mean.shape))
        self.var.assign(np.zeros(self.var.shape))


class TargetPSNRsHistogram(tf.keras.metrics.Metric):
    def __init__(self, name="PSNR target", min_psnr=20, max_psnr=45, step=1, **kwargs):
        super(TargetPSNRsHistogram, self).__init__(name=name, **kwargs)
        self.bins_labels = np.arange(min_psnr, max_psnr + 1, step)
        self.bins = self.add_weight(
            name="bins", shape=self.bins_labels.shape, initializer="zeros"
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        psnrs = tf.image.psnr(y_true, y_pred, max_val=1.0)
        counts = []
        # count how many images fit in each psnr range
        for b, bin_label in enumerate(self.bins_labels):
            counts.append(tf.math.count_nonzero(tf.greater_equal(psnrs, bin_label)))

        self.bins.assign_add(counts)

    def result(self):
        return self.bins

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.bins.assign(np.zeros(self.bins.shape))


def psnr_metric(x_in, x_out):
    if type(x_in) is list:
        img_in = x_in[0]
    else:
        img_in = x_in
    return tf.image.psnr(img_in, x_out, max_val=1.0)


class Encoder(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, conv_depth, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 256
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters,
                (9, 9),
                name="layer_0",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            layers.PReLU(shared_axes=[1, 2])]
        
        
            
        

    def call(self, x):
        
        for sublayer in self.sublayers:
            x = sublayer(x)
            
            
        return x, x
        
        
class Encoder1(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, conv_depth, name="encoder1", **kwargs):
        super(Encoder1, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 256
        
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters*2,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters*2,
                (5, 5),
                name="layer_3",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_3"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters*2,
                (5, 5),
                name="layer_4",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_4"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                256,
                (3, 3),
                name="layer_7",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_7"),
            ),
        ]


    def call(self, x):
        
        for sublayer in self.sublayers:
            x = sublayer(x)

        return x


class Decoder(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, n_channels, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 256
        
        
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters*2,
                (3, 3),
                name="layer_p0",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_p0", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters*2,
                (3, 3),
                name="layer_p1",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_p1", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters*2,
                (5, 5),
                name="layer_p2",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_p2", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_1", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_2",
                corr=False,
                strides_up=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_3",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_3", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                n_channels,
                (9, 9),
                name="layer_4",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tf.nn.sigmoid,
            ),
        ]
        
    def call(self, x):
        
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x

        
        

    


class Rate_fun_encoder(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, name="Rate_fun_encoder", **kwargs):
        super(Rate_fun_encoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 128
        self.sublayers = [
            tfc.SignalConv2D(
                num_filters*2,
                (3, 3),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters*2,
                (3, 3),
                name="layer_3",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_3"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters*2,
                (3, 3),
                name="layer_4",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_4"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters*2,
                (3, 3),
                name="layer_5",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_5"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (3, 3),
                name="layer_6",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_6"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                1,
                (3, 3),
                name="layer_out",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_out"),
            ),
            layers.Activation('sigmoid')
        ]
        
       
        
    

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
      
        return x
        
class Rate_fun_decoder(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, name="Rate_fun_decoder", **kwargs):
        super(Rate_fun_decoder, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 64
        self.sublayers =[
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_0",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                num_filters,
                (5, 5),
                name="layer_1",
                corr=False,
                strides_up=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="igdn_2", inverse=True),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            layers.Conv2D(
                1,
                (5, 5),
                name="layer_out",
                strides=(1, 1),
                padding="same",
                use_bias=True
            ),
            layers.Activation('sigmoid')
        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x


class Weight_adjust_enc(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, name="Weight_adjust_enc", **kwargs):
        super(Weight_adjust_enc, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 128
        conv_depth=32
        self.sublayers = [
            layers.Conv2D(
                num_filters,
                (1, 1),
                name="layer_0",
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation='relu',
            ),
            layers.Conv2D(
                num_filters,
                (1, 1),
                name="layer_1",
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation='relu',
            ),
            layers.Conv2D(
                256*128+128,
                (1, 1),
                name="layer_2",
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=None,
            ),

        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x

class Weight_adjust_dec(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, name="Weight_adjust_dec", **kwargs):
        super(Weight_adjust_dec, self).__init__(name=name, **kwargs)
        self.data_format = "channels_last"
        num_filters = 128
        conv_depth=32
        self.sublayers = [
            layers.Conv2D(
                num_filters,
                (1, 1),
                name="layer_0",
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation='relu',
            ),
            layers.Conv2D(
                num_filters,
                (1, 1),
                name="layer_1",
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation='relu',
            ),
            layers.Conv2D(
                256*128+256,
                (1, 1),
                name="layer_2",
                strides=(1, 1),
                padding='same',
                use_bias=True,
                activation=None,
            )

        ]

    def call(self, x):
        for sublayer in self.sublayers:
            x = sublayer(x)
        return x

@tf.custom_gradient
def Rate_Quantizer(x):
    L=32.
    y=tf.zeros_like(x)
    for l in range(32):
       l=tf.cast(l,tf.float32)
       y=tf.where(tf.math.logical_and(x>=tf.cast(l/(L-1)-0.5/(L-1),tf.float32),x<tf.cast(l/(L-1)+0.5/(L-1),tf.float32)),l*tf.ones_like(x),y)
    def grad(upstream):
        multiplier= (L-1)*tf.ones_like(x)
        return upstream*multiplier
    # y=tf.where(x>0.5,tf.ones_like(x),tf.zeros_like(x))
    return y, grad

@tf.custom_gradient
def Quantizer(x):
    L=16.
    y=tf.zeros_like(x)
    for l in range(16):
        l=tf.cast(l,tf.float32)
        y=tf.where(tf.math.logical_and(x>=tf.cast(l/(L-1)-0.5/(L-1),tf.float32),x<tf.cast(l/(L-1)+0.5/(L-1),tf.float32)),l*tf.ones_like(x),y)
    def grad(upstream):
        multiplier= (L-1)*tf.ones_like(x)
        return upstream*multiplier
    # y=tf.where(x>0.5,tf.ones_like(x),tf.zeros_like(x))
    return y, grad

@tf.custom_gradient
def Mask(x):
    n=256
    L=15.
    m=tf.zeros_like(x)
    m=tf.tile(m,[1, 1, 1, n])

    # @tf.function
    # def _inner_func():
    #     # Avoid exception during the forward pass
    #     return tf.stop_gradient(tf.shape(x))
    #     # return tf.random.shuffle(x)  # This will raise

    batch= tf.shape(x)[0]
    height = tf.shape(x)[1]
    width = tf.shape(x)[2]
    idx_value = tf.range(n)
    idx_value = tf.expand_dims(tf.expand_dims(tf.expand_dims(idx_value, axis=0),axis=0),axis=0)
    idx_value = tf.tile(idx_value, [batch, height, width, 1])
    idx_value = tf.cast(idx_value,tf.float32)

    m=tf.where(idx_value<(n/L)*x, tf.ones_like(m), tf.zeros_like(m))
    def grad(upstream):
        multiplier = tf.where(tf.math.logical_and((tf.math.ceil(idx_value*L/n)>=x-1),(tf.math.ceil(idx_value*L/n)< x+2)), (1/3.)*tf.ones_like(idx_value), tf.zeros_like(idx_value))
        return tf.reduce_sum(multiplier*upstream,keepdims=True)
    return m, grad

def real_awgn(x, stddev):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = x + awgn

    return y


def fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        h = tf.complex(
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
            tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2)),
        )

    # additive white gaussian noise
    awgn = tf.complex(
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
        tf.random.normal(tf.shape(x), 0, 1 / np.sqrt(2)),
    )

    return (h * x + stddev * awgn), h


def phase_invariant_fading(x, stddev, h=None):
    """Implements the fading channel with multiplicative fading and
    additive white gaussian noise. Also assumes that phase shift
    introduced by the fading channel is known at the receiver, making
    the model equivalent to a real slow fading channel.

    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # channel gain
    if h is None:
        n1 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)
        n2 = tf.random.normal([tf.shape(x)[0], 1], 0, 1 / np.sqrt(2), dtype=tf.float32)

        h = tf.sqrt(tf.square(n1) + tf.square(n2))

    # additive white gaussian noise
    awgn = tf.random.normal(tf.shape(x), 0, stddev / np.sqrt(2), dtype=tf.float32)

    return (h * x + awgn), h


class Channel(layers.Layer):
    def __init__(self, channel_type, channel_snr, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def call(self, inputs):
        (encoded_img, mask) = inputs

        inter_shape = tf.shape(encoded_img)
        # reshape array to [-1, dim_z]
        z = layers.Flatten()(encoded_img)
        z_mask = layers.Flatten()(mask)
        z = z*z_mask
        # convert from snr to std
        print("channel_snr: {}".format(self.channel_snr))
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        # Add channel noise
        if self.channel_type == "awgn":
            # dim_z = tf.shape(z)[1]
            dim_z = tf.reduce_sum(z_mask, axis=-1, keepdims=True)
            # normalize latent vector so that the average power is 1
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
                z, axis=1
            )
            z_out = real_awgn(z_in, noise_stddev)
            h = tf.ones_like(z_in)  # h just makes sense on fading channels

        # elif self.channel_type == "fading":
        #     dim_z = tf.shape(z)[1] // 2
        #     # convert z to complex representation
        #     z_in = tf.complex(z[:, :dim_z], z[:, dim_z:])
        #     # normalize the latent vector so that the average power is 1
        #     z_norm = tf.reduce_sum(
        #         tf.math.real(z_in * tf.math.conj(z_in)), axis=1, keepdims=True
        #     )
        #     z_in = z_in * tf.complex(
        #         tf.sqrt(tf.cast(dim_z, dtype=tf.float32) / z_norm), 0.0
        #     )
        #     z_out, h = fading(z_in, noise_stddev, prev_h)
        #     # convert back to real
        #     z_out = tf.concat([tf.math.real(z_out), tf.math.imag(z_out)], 1)
        #
        # elif self.channel_type == "fading-real":
        #     # half of the channels are I component and half Q
        #     dim_z = tf.shape(z)[1] // 2
        #     # normalization
        #     z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.nn.l2_normalize(
        #         z, axis=1
        #     )
        #     z_out, h = phase_invariant_fading(z_in, noise_stddev, prev_h)

        else:
            raise Exception("This option shouldn't be an option!")

        # convert signal back to intermediate shape
        z_out=z_out*z_mask
        z_out = tf.reshape(z_out, inter_shape)
        # compute average power
        avg_power = tf.reduce_mean(tf.math.real(z_in * tf.math.conj(z_in)))
        # add avg_power as layer's metric
        return z_out, avg_power, dim_z


class OutputsCombiner(layers.Layer):
    def __init__(self, name="out_combiner", **kwargs):
        super(OutputsCombiner, self).__init__(name=name, **kwargs)
        self.conv1 = layers.Conv2D(48, 3, 1, padding="same")
        self.prelu1 = layers.PReLU(shared_axes=[1, 2])
        self.conv2 = layers.Conv2D(3, 3, 1, padding="same", activation=tf.nn.sigmoid)

    def call(self, inputs):
        img_prev, residual = inputs

        reconst = tf.concat([img_prev, residual], axis=-1)
        reconst = self.conv1(reconst)
        reconst = self.prelu1(reconst)
        reconst = self.conv2(reconst)

        return reconst


class DeepJSCCF(layers.Layer):
    def __init__(
        self,
        channel_snr,
        conv_depth,
        channel_type,
        feedback_snr,
        refinement_layer,
        layer_id,
        target_analysis=False,
        name="deep_jscc_f",
        **kwargs
    ):
        super(DeepJSCCF, self).__init__(name=name, **kwargs)

        n_channels = 3  # change this if working with BW images
        self.refinement_layer = refinement_layer
        self.feedback_snr = feedback_snr
        self.layer = layer_id
        self.encoder = Encoder(conv_depth, name="encoder_output")
       
        self.encoder1 = Encoder1(conv_depth, name="encoder_output1")
        
        self.decoder = Decoder(n_channels, name="decoder_output")
        self.rate_fun_encoder = Rate_fun_encoder()
      
        self.rate_fun_decoder = Rate_fun_decoder()
        # self.rate_fun.trainable=False
        self.channel = Channel(channel_type, channel_snr, name="channel_output")
       
        if self.refinement_layer:
            self.image_combiner = OutputsCombiner(name="out_comb")
        self.target_analysis = target_analysis

    def call(self, inputs):
        if self.refinement_layer:
            (
                img,
                prev_img_out_fb,
                prev_chn_out_fb,
                prev_img_out_dec,
                prev_chn_out_dec,
                prev_chn_gain,
            ) = inputs

            img_in = tf.concat([prev_img_out_fb, img], axis=-1)

        else:  # base layer
            # inputs is just the original image
            img_in = img = inputs
            prev_chn_gain = None

     
        
      #   rate=self.rate_fun_decoder(rate_q/32.)
        
     #    rate_enlarge = tf.image.resize(rate, [tf.shape(img_in)[1], tf.shape(img_in)[2]],
     #                             tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        chn_in, encode_features = self.encoder(img_in)
        
        rate=self.rate_fun_encoder(tf.stop_gradient(chn_in))
        bit = Quantizer(rate)
        m = Mask(bit)
        rate_enlarge = tf.image.resize(rate, [tf.shape(chn_in)[1], tf.shape(chn_in)[2]],
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)    
        
        chn_in = self.encoder1(tf.concat([chn_in,tf.stop_gradient(rate_enlarge)],axis=-1))
       #  rate = tf.image.resize(rate, [tf.shape(chn_in)[1], tf.shape(chn_in)[2]],
       #                           tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        # rate=tf.stop_gradient(rate)

        
        batch = tf.shape(chn_in)[0]
        height = tf.shape(chn_in)[1]
        width = tf.shape(chn_in)[2]
        
        
        
      #  weights_enc = self.weight_adjust_enc(tf.stop_gradient(rate))
      #  weights_enc_weight1 = tf.reshape(weights_enc[:, :, :, :256 * 128], [batch, height, width, 256, 128])     
      #  weights_enc_bias = weights_enc[:, :, :, 256 * 128:]
      #  chn_in = tf.reshape(chn_in, [batch, height, width, 1, 256])
      #  chn_in = tf.matmul(chn_in, weights_enc_weight1)

      #  chn_in = tf.reshape(chn_in, [batch, height, width, 128]) + weights_enc_bias

        bit_mean, bit_var = tf.nn.moments(tf.reduce_mean(tf.reduce_sum(m, axis=-1)/16., axis=[1, 2]), axes=0)
        chn_in = chn_in * m
        chn_out, avg_power, dim_z = self.channel((chn_in, m))

      #  weights_dec = self.weight_adjust_dec(tf.stop_gradient(rate))
      #  weights_dec_weight1 = tf.reshape(weights_dec[:, :, :, :256 * 128], [batch, height, width, 128, 256])
        
      #  weights_dec_bias = weights_dec[:, :, :, 256 * 128:]

      #  chn_out = tf.reshape(chn_out, [batch, height, width, 1, 128])
      #  chn_out = tf.matmul(chn_out, weights_dec_weight1)
       
      #  chn_out = tf.reshape(chn_out, [batch, height, width, 256]) + weights_dec_bias

        # theta_tilde = self.decoder(chn_out, **kwargs)
        #
        # chn_out, avg_power, chn_gain = self.channel((chn_in, prev_chn_gain))

        # add feedback noise to chn_output
        if self.feedback_snr is None:  # No feedback noise
            chn_out_fb = chn_out
        else:
            fb_noise_stddev = np.sqrt(10 ** (-self.feedback_snr / 10))
            chn_out_fb = real_awgn(chn_out, fb_noise_stddev)

        if self.refinement_layer:
            # combine chn_output with previous stored chn_outs
            chn_out_exp = tf.concat([chn_out, prev_chn_out_dec], axis=-1)
            residual_img = self.decoder(chn_out_exp)
            # combine residual ith previous stored image reconstruction
            decoded_img = self.image_combiner((prev_img_out_dec, residual_img))

            # feedback estimation
            # Note: the ops below is just computed when this is not the last
            # layer (as this op is not included in the loss function when this
            # is the output), so decoder is just trained with actual chn_outs,
            # and the op below just happens when trainable=False
            chn_out_exp_fb = tf.concat([chn_out_fb, prev_chn_out_fb], axis=-1)
            residual_img_fb = self.decoder(chn_out_exp_fb)
            decoded_img_fb = self.image_combiner([prev_img_out_fb, residual_img_fb])
        else:
            chn_out_exp = chn_out
            decoded_img = self.decoder(tf.concat([chn_out_exp ,tf.stop_gradient(bit)/15.],axis=-1))

            chn_out_exp_fb = chn_out_fb
            decoded_img_fb = decoded_img

        # keep track of some metrics
        self.add_metric(
            tf.image.psnr(img, decoded_img, max_val=1.0),
            aggregation="mean",
            name="psnr{}".format(self.layer),
        )
        # self.add_metric(
        #     tf.image.psnr(img, decoded_img_fb, max_val=1.0),
        #     aggregation="mean",
        #     name="psnr_fb{}".format(self.layer),
        # )
        self.add_metric(
            tf.reduce_mean(tf.math.square(img - decoded_img)),
            aggregation="mean",
            name="mse{}".format(self.layer),
        )

        self.add_metric(
            bit_mean,
            aggregation="bit_mean",
            name="bit_mean{}".format(self.layer),
        )

        self.add_metric(
            bit_var,
            aggregation="bit_var",
            name="bit_var{}".format(self.layer),
        )

        # self.add_metric(
        #     avg_power, aggregation="mean", name="avg_pwr{}".format(self.layer)
        # )
        # rate_mean=tf.reduce_mean(rate)
        # bit_mean=tf.reduce_mean(tf.reduce_sum(m,axis=-1))

        return (decoded_img, encode_features, rate, bit, m, bit_mean, bit_var, decoded_img_fb, chn_out_exp, chn_out_exp_fb, chn_in)

    def change_channel_snr(self, channel_snr):
        self.channel.channel_snr = channel_snr

    def change_feedback_snr(self, feedback_snr):
        self.feedback_snr = feedback_snr


class Discriminator(layers.Layer):
    """Build encoder from specified arch"""

    def __init__(self, name="discriminator", **kwargs):
        super(Discriminator, self).__init__(name=name,
                **kwargs)
        self.data_format = "channels_last"
        self.num_filters_base=64
        self.num_latent_filters_base = 32
        # self.num_layers = 3
        self.latents_layers=[
            tfc.SignalConv2D(
                self.num_latent_filters_base,
                (9, 9),
                name="latent_layer_0",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="latent_gdn_0"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                self.num_latent_filters_base,
                (5, 5),
                name="latent_layer_1",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="latent_gdn_1"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                12,
                (5, 5),
                name="latent_layer_2",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="latent_gdn_2"),
            ),
        ]
        self.layers = [
            tfc.SignalConv2D(
                self.num_filters_base,
                (5, 5),
                name="layer_0",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_0"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                self.num_filters_base,
                (5, 5),
                name="layer_1",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_1"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                self.num_filters_base,
                (5, 5),
                name="layer_2",
                corr=True,
                strides_down=2,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_2"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                self.num_filters_base,
                (5, 5),
                name="layer_3",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=tfc.GDN(name="gdn_3"),
            ),
            layers.PReLU(shared_axes=[1, 2]),
            tfc.SignalConv2D(
                1,
                (5, 5),
                name="layer_out",
                corr=True,
                strides_down=1,
                padding="same_zeros",
                use_bias=True,
                activation=None,
            ),
        ]

    def call(self, x):
        x, latent = x
        x_shape = tf.shape(x)
        for sublayer in self.latents_layers:
            latent=sublayer(latent)
        latent = tf.image.resize(latent, [x_shape[1], x_shape[2]],
                                 tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        x = tf.concat([x, latent], axis=-1)
        for sublayer in self.layers:
            x = sublayer(x)
        out_logits = tf.reshape(x, [-1, 1])  # Reshape all into batch dimension.
        # out = tf.nn.sigmoid(out_logits)
        return out_logits


class LPIPSLoss(layers.Layer):
  """Calcualte LPIPS loss."""

  def __init__(self, weight_path, name="LPIPSLoss", **kwargs):
    # helpers.ensure_lpips_weights_exist(weight_path)
    super(LPIPSLoss, self).__init__(name=name,
                                        **kwargs)
    if not os.path.isfile(weight_path):

        print("Downloading LPIPS weights:", _LPIPS_URL, "->", weight_path)

        urllib.request.urlretrieve(_LPIPS_URL, weight_path)
        if not os.path.isfile(weight_path):
            raise ValueError(f"Failed to download LPIPS weights from {_LPIPS_URL} "
                             f"to {weight_path}. Please manually download!")


    def wrap_frozen_graph(graph_def, inputs, outputs):
      def _imports_graph_def():
        tf.graph_util.import_graph_def(graph_def, name="")
      wrapped_import = tf.compat.v1.wrap_function(_imports_graph_def, [])
      import_graph = wrapped_import.graph
      return wrapped_import.prune(
          tf.nest.map_structure(import_graph.as_graph_element, inputs),
          tf.nest.map_structure(import_graph.as_graph_element, outputs))

    # Pack LPIPS network into a tf function
    graph_def = tf.compat.v1.GraphDef()
    with open(weight_path, "rb") as f:
      graph_def.ParseFromString(f.read())
    self._lpips_func = tf.function(
        wrap_frozen_graph(
            graph_def, inputs=("0:0", "1:0"), outputs="Reshape_10:0"))

  def call(self, fake_image, real_image):
    """Assuming inputs are in [0, 1]."""
    # Move inputs to [-1, 1] and NCHW format.
    def _transpose_to_nchw(x):
      return tf.transpose(x, (0, 3, 1, 2))
    fake_image = _transpose_to_nchw(fake_image * 2 - 1.0)
    real_image = _transpose_to_nchw(real_image * 2 - 1.0)
    loss = self._lpips_func(fake_image, real_image)
    return tf.reduce_mean(loss)  # Loss is N111, take mean to get scalar.




dloss_tracker = tf.keras.metrics.Mean(name="dloss")
gloss_tracker = tf.keras.metrics.Mean(name="gloss")
distortion_tracker = tf.keras.metrics.Mean(name="distortion")
perceptual_loss_tracker = tf.keras.metrics.Mean(name="perceptual_loss")

class GAN(tf.keras.Model):
    def __init__(self, inputs, outputs, discriminator, generator, lpips_loss):
        super(GAN, self).__init__(inputs=inputs, outputs=outputs)
        self.discriminator = discriminator
        self.generator = generator
        self.lpips_loss = lpips_loss

    def compile(self, d_optimizer, g_rate_optimizer, g_optimizer,metrics):
        super(GAN, self).compile(metrics=metrics)
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.g_rate_optimizer=g_rate_optimizer
        # self.loss_fn = loss_fn
        # self._metrics=metrics

    def train_step(self, real_images):

        # Sample random points in the latent space
        # batch_size = tf.shape(real_images)[0]
        #
        # sub_batch_size = batch_size // 2
        # splits = [sub_batch_size, sub_batch_size]
        # input_image, input_images_d_steps = tf.split(real_images, splits)
        #
        #
        # # Decode them to fake images
        # generated_images = self.generator(input_images_d_steps)
        # (
        #     decoded_img,
        #     _decoded_img_fb,
        #     _chn_out_exp,
        #     _chn_out_exp_fb,
        #     _chn_gain,
        #     ch_in,
        # ) = generated_images
        # # Combine them with real images
        # discriminator_in = tf.concat([input_images_d_steps, decoded_img], axis=0)
        # latent = tf.stop_gradient(ch_in)
        # latent = tf.concat([latent, latent], axis=0)
        #
        # discriminator_in = (discriminator_in, latent)
        #
        #
        # # Train the discriminator
        # with tf.GradientTape() as tape:
        #     predictions = self.discriminator(discriminator_in)
        #     d_real_logits, d_fake_logits = tf.split(predictions, 2)
        #     d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=d_real_logits, labels=tf.ones_like(d_real_logits)+0.05 * tf.random.uniform(tf.shape(d_real_logits)),
        #         name="cross_entropy_d_real"))
        #     d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #         logits=d_fake_logits, labels=tf.zeros_like(d_fake_logits)+0.05 * tf.random.uniform(tf.shape(d_fake_logits)),
        #         name="cross_entropy_d_fake"))
        #     d_loss = d_loss_real + d_loss_fake
        #     # g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
        #     #     logits=d_fake_logits, labels=tf.ones_like(d_fake_logits),
        #     #     name="cross_entropy_g"))
        #
        # # d_w=self.discriminator.trainable_weights
        # # g_w=self.generator.trainable_weights
        # grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        # self.d_optimizer.apply_gradients(
        #     zip(grads, self.discriminator.trainable_weights)
        # )
        # dloss_tracker.update_state(d_loss)
        # # misleading_labels = tf.zeros((batch_size, 1))
        #
        # # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        input_image = real_images
        with tf.GradientTape() as tape:
            generated_images = self.generator(input_image)
            (
                decoded_img,
                encode_features,
                rate,
                bit,
                m,
                bit_mean, bit_var,
                _decoded_img_fb,
                _chn_out_exp,
                _chn_out_exp_fb,

                ch_in,
            ) = generated_images
            # discriminator_in = tf.concat([input_image, decoded_img], axis=0)
            # latent = tf.stop_gradient(ch_in)
            # latent = tf.concat([latent, latent], axis=0)
            # discriminator_in = (discriminator_in, latent)
            # predictions = self.discriminator(discriminator_in)
            # d_real_logits, d_fake_logits = tf.split(predictions, 2)
            #
            # g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            #     logits=d_fake_logits, labels=tf.ones_like(d_fake_logits)+0.05 * tf.random.uniform(tf.shape(d_fake_logits)),
            #     name="cross_entropy_g"))

            sq_err = tf.math.squared_difference(input_image, decoded_img)
            distortion_loss = tf.reduce_mean(sq_err)
            rate_loss=tf.reduce_mean(rate)
            # input_image_scaled=input_image*255.
            # decoded_img_scaled=decoded_img*255.
            perceptual_loss=self.lpips_loss(decoded_img, input_image)
            #
            # loss=distortion_loss+0.01*g_loss+0.01*perceptual_loss 2.1 (3.68)  1.2 (5.44) 0.8 ()//   8.0(4.0)   6.0(5)  4.0(6.2)  3.0(7)
            loss=distortion_loss+0.8*rate_loss

        # gloss_tracker.update_state(g_loss)
        perceptual_loss_tracker.update_state(perceptual_loss)
        distortion_tracker.update_state(distortion_loss)

        grads = tape.gradient(loss, self.generator.trainable_weights)
        vars_all=self.generator.trainable_weights
        var_gen=[]
        var_rate=[]
        grade_gen = []
        grade_rate = []
        for grade,var in zip(grads, self.generator.trainable_weights):
            if 'decoder' in var.name:
                var_rate.append(var)
                grade_rate.append(grade)
            else:
                var_gen.append(var)
                grade_gen.append(grade)



        # self.g_optimizer.apply_gradients(zip(grads, self.generator.trainable_weights))
        self.g_rate_optimizer.apply_gradients(zip(grade_rate, var_rate))
       # self.g_optimizer.apply_gradients(zip(grade_gen, var_gen))
        self.compiled_metrics.update_state(input_image, decoded_img)
        state_dict = {m.name: m.result() for m in self.metrics}
        # state_dict.update({"dloss": dloss_tracker.result()})
        # state_dict.update({"gloss": gloss_tracker.result()})
        state_dict.update({"perceptual_loss": perceptual_loss_tracker.result()})
        state_dict.update({"distortion_loss": distortion_tracker.result()})

        return state_dict

    def test_step(self, real_images):
        # Unpack the data
        input_image = real_images
        # Compute predictions
        generated_images = self.generator(input_image)
        (
            decoded_img,
            encode_features,
            rate,
            bit,
            m,
            bit_mean, bit_var,
            _decoded_img_fb,
            _chn_out_exp,
            _chn_out_exp_fb,

            ch_in,
        ) = generated_images
        sq_err = tf.math.squared_difference(input_image, decoded_img)
        distortion_loss = tf.reduce_mean(sq_err)
        perceptual_loss = self.lpips_loss(decoded_img, input_image)
        perceptual_loss_tracker.update_state(perceptual_loss)
        distortion_tracker.update_state(distortion_loss)
        self.compiled_metrics.update_state(input_image, decoded_img)
        state_dict = {m.name: m.result() for m in self.metrics}
        state_dict.update({"perceptual_loss": perceptual_loss_tracker.result()})
        state_dict.update({"distortion_loss": distortion_tracker.result()})
        return state_dict


def main(args):
    # get dataset
    x_train, x_val, x_tst, x_train_small = get_dataset(args)

    if args.delete_previous_model and tf.io.gfile.exists(args.model_dir):
        print("Deleting previous model files at {}".format(args.model_dir))
        tf.io.gfile.rmtree(args.model_dir)
        tf.io.gfile.makedirs(args.model_dir)
    else:
        print("Starting new model at {}".format(args.model_dir))
        tf.io.gfile.makedirs(args.model_dir)

    # load model
    prev_layer_out = None
    # add input placeholder to please keras
    img = tf.keras.Input(shape=(None, None, 3))

    if not args.run_eval_once:
        feedback_snr = None if not args.feedback_noise else args.feedback_snr_train
        channel_snr = args.channel_snr_train
    else:
        feedback_snr = None if not args.feedback_noise else args.feedback_snr_eval
        channel_snr = args.channel_snr_eval

    all_models = []
    for layer in range(args.n_layers):
        ckpt_file = os.path.join(args.model_dir, "ckpt_layer{}".format(layer))
        layer_name = "layer{}_generator".format(layer)
        ae_layer = DeepJSCCF(
            channel_snr,
            int(args.conv_depth),
            args.channel,
            feedback_snr,
            layer > 0,  # refinement or base?
            layer,
            args.target_analysis,
            name=layer_name,
        )
        generator=ae_layer

        # connect ae_layer to previous model, (if any)
        if layer == 0:  # base layer
            # model returns img and channel outputs
            # layer_output = ae_layer(img)
            layer_output=ae_layer(img)
        else:
            # add prev layer outputs as input for cur layer
            (
                prev_img_out_dec,
                prev_img_out_fb,
                prev_chn_out_dec,
                prev_chn_out_fb,
                prev_chn_gain,
                ch_in,
            ) = prev_layer_out
            layer_output = ae_layer(
                (   img,
                    prev_img_out_fb,
                    prev_chn_out_fb,
                    prev_img_out_dec,
                    prev_chn_out_dec,
                    prev_chn_gain,
                )
            )

        (
            decoded_img,
            encode_features,
            rate,
            bit,
            m,
            rate_mean,
            bit_mean,
            _decoded_img_fb,
            _chn_out_exp,
            _chn_out_exp_fb,

            ch_in,
        ) = layer_output
        # model = tf.keras.Model(inputs=img, outputs=decoded_img)
        # model=CustomModel(inputs=img, outputs=decoded_img)
        discriminator=Discriminator(name="discriminator{}".format(layer))
        lpips_loss=LPIPSLoss(weight_path='./lpips_weight__net-lin_alex_v0.1.pb',name="lpips{}".format(layer))
        model=GAN(inputs=img, outputs=[decoded_img,rate,bit,m,rate_mean,bit_mean], discriminator=discriminator,generator=generator,lpips_loss=lpips_loss)
        model_metrics = [
            tf.keras.metrics.MeanSquaredError(),
            psnr_metric,
            # PSNRsVar(name="psnr_var{}".format(layer)),
        ]
        if args.target_analysis:
            model_metrics.append(TargetPSNRsHistogram(name="target{}".format(layer)))
        model.compile(
            g_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learn_rate),
            g_rate_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learn_rate),
            d_optimizer=tf.keras.optimizers.Adam(learning_rate=args.learn_rate),
            metrics=model_metrics,
        )


        # check if checkpoint already exists and load it
        if args.pretrained_base_layer or args.previous_checkpoint:
            # trick to restore metrics too (see tensorflow guide on saving and
            # serializing subclassed models)
            # model.train_on_batch(x_train)
          #  model.fit(
          #      x_train_small,
          #      epochs=1,
          #      verbose=1
          #      ),

            if args.pretrained_base_layer:
                print("Using pre-trained base layer!")
                model.load_weights(args.pretrained_base_layer,by_name=True,skip_mismatch=True)
            else:
                print("Restoring weights from checkpoint!")
                print(args.previous_checkpoint)
                # ckpt_file=ckpt_file + "-01.hdf5"
                model.load_weights(args.previous_checkpoint)

        print(model.summary())

        # skip training if just running eval or if loading first layer from
        # pretrained ckpt
        if not args.run_eval_once:
            train_patience = 3 if args.dataset_train != "imagenet" else 2
            callbacks = [
                # tf.keras.callbacks.EarlyStopping(
                #     patience=train_patience,
                #     monitor="val_psnr_metric",
                #     min_delta=10e-3,
                #     verbose=1,
                #     mode="max",
                #     restore_best_weights=True,
                # ),
                # tf.keras.callbacks.TensorBoard(log_dir=args.eval_dir),
                # just save a single checkpoint with best. If more is wanted,
                # create a new callback
                tf.keras.callbacks.ModelCheckpoint(
                    filepath=ckpt_file+"-{epoch:02d}.hdf5",
                    # monitor="val_psnr_metric",
                    # mode="max",
                    save_best_only=False,
                    verbose=1,
                    save_weights_only=True,
                    save_freq=10*(1+DATASETS[args.dataset_train]._NUM_IMAGES["train"]// args.batch_size_train),

                ),
                tf.keras.callbacks.TerminateOnNaN(),
            ]

            if args.dataset_train == "imagenet":
                callbacks.append(NBatchLogger(100))

            model.fit(
                x_train,
                epochs=args.train_epochs,
                validation_data=x_val,
                callbacks=callbacks,
                verbose=1,
                validation_freq=args.epochs_between_evals,
                validation_steps=(
                    DATASETS[args.dataset_train]._NUM_IMAGES["validation"]
                    // args.batch_size_val
                ),
                initial_epoch=0

            )

        # freeze weights of already trained layers
        model.trainable = False
        # define model as prev_model
        prev_layer_out = layer_output
        all_models.append(model)

    print("EVALUATION!!!")
  #  models = [model] if not args.target_analysis else all_models
  #  for eval_model in models:
  #      dataset = x_tst.enumerate()
  #      bit_frequency=[]
  #      bit_all=[]
  #      for i, inputs in dataset.as_numpy_iterator():
  #          print(i)
  #          outs, rate,bit,m,bit_mean, bit_var = eval_model.predict(inputs, verbose=0)
  #          print(np.shape(bit))
  #          bit=bit[0]
  #          height=np.shape(bit)[0]
  #          width=np.shape(bit)[1]
  #          f=np.zeros(64)
   #         for j in range(64):
   #             all_zeors=np.zeros_like(bit)
   #             all_ones=np.ones_like(bit)
   #             bit_flatten=np.reshape(bit,-1)
   #             out=np.count_nonzero(bit_flatten==j)
   #             f[j]=out/(height*width)
   #         print('tmp_frequency:')
    #        print(np.asarray(f))
   #         bit_frequency.append(f)
   #         print('average_frequency:')
   #         print(np.mean(np.asarray(bit_frequency),axis=0))
            
    # normally we just eval the complete model, unless we are doing target_analysis
    models = [model] if not args.target_analysis else all_models
    for eval_model in models:
        dataset = x_tst.enumerate()
        psnr_all=[]
        bit_all=[]
        lpips_all=[]
        for i, inputs in dataset.as_numpy_iterator():
            print(i)
            outs, rate,bit,m,bit_mean, bit_var = eval_model.predict(inputs, verbose=0)
            psnr_=tf.image.psnr(inputs, outs, max_val=1.0)
            psnr_all.append(psnr_.numpy())
            bit_all.append(bit_mean)
            lpips_=lpips_loss(outs, inputs)
            lpips_=lpips_.numpy()
            lpips_all.append(lpips_)
            print('PSNR_mean:')
            print(np.mean(np.asarray(psnr_all)))
            print('PSNR_var:')
            print(np.std(np.asarray(psnr_all)))
            print('bit_mean:')
            print(np.mean(np.asarray(bit_all)))
            print('bit_var:')
            print(np.std(np.asarray(bit_all)))
            print('LPIPS_mean:')
            print(np.mean(np.asarray(lpips_all)))
            print('LPIPS_var:')
            print(np.std(np.asarray(lpips_all)))

    #         inputs = np.round(inputs * 255.)
    #         inputs = np.clip(inputs, 0, 255.)
    #         inputs = inputs.astype(np.uint8)
    #         outs = np.round(outs * 255.)
    #         outs = np.clip(outs, 0, 255.)
    #         outs = outs.astype(np.uint8)
    #         Image.fromarray(inputs[0]).save(
    #             os.path.join('./out', f'{i:010d}_inp.png'))
    #         Image.fromarray(outs[0]).save(
    #             os.path.join('./out', f'{i:010d}_out.png'))
   # models = [model] if not args.target_analysis else all_models
   # for eval_model in models:
   #     out_eval= eval_model.evaluate(x_tst, verbose=2)
   #     for m, v in zip(eval_model.metrics_names, out_eval):
   #         met_name = "_".join(["eval", m])
  #          print("{}={}".format(met_name, v), end=" ")
  #      print()
  #      print()


def get_dataset(args):
    data_options = tf.data.Options()
    data_options.experimental_deterministic = False
    data_options.experimental_optimization.apply_default_optimizations = True
    data_options.experimental_optimization.map_parallelization = True
    data_options.experimental_optimization.parallel_batch = True
    # data_options.experimental_optimization.autotune_buffers = True

    def prepare_dataset(dataset, mode, parse_record_fn, bs):
        dataset = dataset.with_options(data_options)
        if mode == "train":
            dataset = dataset.shuffle(buffer_size=dataset_obj.SHUFFLE_BUFFER)
        dataset = dataset.map(
            lambda v: parse_record_fn(v, mode, tf.float32),
            num_parallel_calls=tf.data.experimental.AUTOTUNE,
        )
        dataset=dataset.batch(bs)

        def _batch_to_dict(batch):
            return dict(input_image=batch)

        return dataset

    dataset_obj = DATASETS[args.dataset_train]
    parse_record_fn = dataset_obj.parse_record
    if args.dataset_train != "imagenet" and args.dataset_train != "coco2014":
        tr_val_dataset = dataset_obj.get_dataset(True, args.data_dir_train)
        tr_dataset = tr_val_dataset.take(dataset_obj._NUM_IMAGES["train"])
        val_dataset = tr_val_dataset.skip(dataset_obj._NUM_IMAGES["train"])
        tr_dataset_small=tr_dataset
    elif args.dataset_train == "coco2014":  # treat imagenet differently, as we usually dont use it for training
        tr_dataset,val_dataset, tr_dataset_small = dataset_obj.get_dataset(True, args.data_dir_train)
    else:
        tr_dataset = dataset_obj.get_dataset(True, args.data_dir_train)
        val_dataset = dataset_obj.get_dataset(False, args.data_dir_train)
    # Train
    x_train = prepare_dataset(
        tr_dataset, "train", parse_record_fn, args.batch_size_train
    )
    x_train_small = prepare_dataset(
        tr_dataset_small, "train", parse_record_fn, args.batch_size_train
    )
    # Validation

    x_val = prepare_dataset(val_dataset, "val", parse_record_fn, args.batch_size_val)


    # Test
    dataset_obj = DATASETS[args.dataset_eval]
    parse_record_fn = dataset_obj.parse_record
    tst_dataset = dataset_obj.get_dataset(False, args.data_dir_eval)
    x_tst = prepare_dataset(tst_dataset, "test", parse_record_fn, args.batch_size_eval)
    x_tst.repeat(10)  # number of realisations per image on evaluation

    return x_train, x_val, x_tst, x_train_small


if __name__ == "__main__":
    # parse args
    p = configargparse.ArgParser()
    p.add(
        "-c",
        "--my-config",
        required=False,
        is_config_file=True,
        help="config file path",
    )
    p.add(
        "--conv_depth",
        type=float,
        default=64,
        help=(
            "Number of channels of last conv layer, used to define the "
            "compression rate: k/n=c_out/(16*3)"
        ),
        required=False,
    )
    p.add(
        "--n_layers",
        type=int,
        default=1,
        help=("Number of layers/rounds used in the transmission"),
        required=False,
    )
    p.add(
        "--channel",
        type=str,
        default="awgn",
        choices=["awgn", "fading", "fading-real"],
        help="Model of channel used (awgn, fading)",
    )
    p.add(
        "--model_dir",
        type=str,
        default="./tmp/train_logs_c64_l1_s10_coco_adaptive_test5_c10_deeper_c4_smallL_7_tune_refined",
        help=("The location of the model checkpoint files."),
    )
    p.add(
        "--eval_dir",
        type=str,
        default="./tmp/train_logs_c64_l1_s10_coco_adaptive_test5_c10_deeper_c4_smallL_7_tune_refined/eval",
        help=("The location of eval files (tensorboard, etc)."),
    )
    p.add(
        "--delete_previous_model",
        # action="store_true",
        default=False,
        help=("If model_dir has checkpoints, delete it before" "starting new run"),
    )
    p.add(
        "--channel_snr_train",
        type=float,
        default=10,
        help="target SNR of channel during training (dB)",
    )
    p.add(
        "--channel_snr_eval",
        type=float,
        default=10,
        help="target SNR of channel during evaluation (dB)",
    )
    p.add(
        "--feedback_noise",
        action="store_true",
        default=False,
        help=("Apply (AWGN) noise to feedback channel"),
    )
    p.add(
        "--feedback_snr_train",
        type=float,
        default=20,
        help=(
            "SNR (dB) of the feedback channel "
            "(only applies when feedback_noise=True)"
        ),
    )
    p.add(
        "--feedback_snr_eval",
        type=float,
        default=20,
        help=(
            "SNR (dB) of the feedback channel (only applies when feedback_noise=True)"
        ),
    )
    p.add(
        "--learn_rate",
        type=float,
        default=0.0001,
        help="Learning rate for Adam optimizer",
    )
    p.add(
        "--run_eval_once",
        action="store_true",
        default=True,
        help="Skip train, run only eval and exit",
    )
    p.add(
        "--train_epochs",
        type=int,
        default=600,  #10000
        help=(
            "The number of epochs used to train (each epoch goes over the whole dataset)"
        ),
    )
    p.add("--batch_size_train", type=int, default=32, help="Batch size for training")
    p.add("--batch_size_val", type=int, default=1, help="Batch size for validation")
    p.add("--batch_size_eval", type=int, default=1, help="Batch size for evaluation")
    p.add(
        "--epochs_between_evals",
        type=int,
        default=1,
        help=("the number of training epochs to run between evaluations."),
    )
    p.add(
        "--dataset_train",
        type=str,
        default="coco2014",
        choices=DATASETS.keys(),
        help=("Choose image dataset. Options: {}".format(DATASETS.keys())),
    )
    p.add(
        "--dataset_eval",
        type=str,
        default="coco2014",
        choices=DATASETS.keys(),
        help=("Choose image dataset. Options: {}".format(DATASETS.keys())),
    )
    p.add(
        "--data_dir_train",
        type=str,
        default="./tmp",
        help="Directory where to store the training data set",
    )
    p.add(
        "--data_dir_eval",
        type=str,
        default="./tmp",
        help="Directory where to store the eval data set",
    )
    p.add(
        "--pretrained_base_layer",
        type=str,
        # default="/opt/project/tmp/train_logs_c64_l1_s10_cifar_google_conv2d_all/ckpt_layer0-300.hdf5",252
        default=None,
        help="Use existing checkpoints for base layer",
    )
    p.add(
        "--previous_checkpoint",
        type=str,
        default="./tmp/train_logs_c64_l1_s10_coco_adaptive_test5_c10_deeper_c4_smallL_7_tune/ckpt_layer0-600.hdf5",
      
        # default=None,
        help="Use existing checkpoints for base layer",
    )
    p.add(
        "--target_analysis",
        action="store_true",
        default=False,
        help="perform PSNR target analysis",
    )

    args = p.parse_args()

    print("#######################################")
    print("Current execution paramenters:")
    for arg, value in sorted(vars(args).items()):
        print("{}: {}".format(arg, value))
    print("#######################################")
    main(args)
