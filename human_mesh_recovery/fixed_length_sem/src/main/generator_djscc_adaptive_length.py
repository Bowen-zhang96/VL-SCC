import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.applications.resnet_v2 import ResNet50V2
import numpy as np
from main import model_util
from main.config import Config
from main.smpl import Smpl

@tf.custom_gradient
def real_awgn(x, m):
    """Implements the real additive white gaussian noise channel.
    Args:
        x: channel input symbols
        stddev: standard deviation of noise
    Returns:
        y: noisy channel output symbols
    """
    # additive white gaussian noise
    stddev = np.sqrt(10 ** (10 / 10))
    awgn = tf.random.normal(tf.shape(x), 0, stddev, dtype=tf.float32)
    y = (x + awgn)*m

    def grad(upstream):
        return upstream * m, upstream * x


    return y, grad



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

class Channel(tf.keras.Model):
    def __init__(self, channel_type, channel_snr, name="channel", **kwargs):
        super(Channel, self).__init__(name=name, **kwargs)
        self.channel_type = channel_type
        self.channel_snr = channel_snr

    def call(self, inputs, **kwargs):
        (encoded_img, mask) = inputs
        # inter_shape = tf.shape(encoded_img)
        # reshape array to [-1, dim_z]
        # z = layers.Flatten()(encoded_img, **kwargs)
        z=encoded_img
        # convert from snr to std
        # print("channel_snr: {}".format(self.channel_snr))
        noise_stddev = np.sqrt(10 ** (-self.channel_snr / 10))

        # Add channel noise
        if self.channel_type == "awgn":
            # dim_z = tf.shape(z)[1]
            dim_z = tf.reduce_sum(tf.stop_gradient(mask),axis=-1,keepdims=True)
            # normalize latent vector so that the average power is 1
            z_in = tf.sqrt(tf.cast(dim_z, dtype=tf.float32)) * tf.math.l2_normalize(
                z, axis=-1
            )
            z_out = real_awgn(z_in, mask)

            # h = tf.ones_like(z_in)  # h just makes sense on fading channels


        else:
            raise Exception("This option shouldn't be an option!")

        # convert signal back to intermediate shape
        # z_out = tf.reshape(z_out, inter_shape)
        # compute average power
        avg_power = tf.reduce_mean(tf.math.real(z_in * tf.math.conj(z_in)))
        # add avg_power as layer's metric
        return z_out, avg_power

class Regressor(tf.keras.Model):

    def __init__(self):
        super(Regressor, self).__init__(name='regressor')
        self.config = Config()

        self.mean_theta = tf.Variable(model_util.load_mean_theta(), name='mean_theta', trainable=True)

        self.fc_one = layers.Dense(1024, name='fc_0')
        self.dropout_one = layers.Dropout(0.5)
        self.fc_two = layers.Dense(1024, name='fc_1')
        self.dropout_two = layers.Dropout(0.5)
        variance_scaling = tf.keras.initializers.VarianceScaling(.01, mode='fan_avg', distribution='uniform')
        self.fc_out = layers.Dense(85, kernel_initializer=variance_scaling, name='fc_out')

    def call(self, inputs, **kwargs):
        batch_size = inputs.shape[0] or self.config.BATCH_SIZE
        shape = (batch_size, 2048)
        assert inputs.shape[1:] == shape[1:], 'shape mismatch: should be {} but is {}'.format(shape, inputs.shape)

        batch_theta = tf.tile(self.mean_theta, [batch_size, 1])
        thetas = tf.TensorArray(tf.float32, self.config.ITERATIONS)
        for i in range(self.config.ITERATIONS):
            # [batch x 2133] <- [batch x 2048] + [batch x 85]
            total_inputs = tf.concat([inputs, batch_theta], axis=1)
            batch_theta = batch_theta + self._fc_blocks(total_inputs, **kwargs)
            thetas = thetas.write(i, batch_theta)

        return thetas.stack()

    def _fc_blocks(self, inputs, **kwargs):
        x = self.fc_one(inputs, **kwargs)
        x = tf.nn.relu(x)
        x = self.dropout_one(x, **kwargs)
        x = self.fc_two(x, **kwargs)
        x = tf.nn.relu(x)
        x = self.dropout_two(x, **kwargs)
        x = self.fc_out(x, **kwargs)
        return x

@tf.custom_gradient
def Binarizer(x):
    def grad(upstream):
        return upstream
    y=tf.where(x>0.5,tf.ones_like(x),tf.zeros_like(x))
    return y, grad

@tf.custom_gradient
def Binarizer(x):
    def grad(upstream):
        return upstream
    y=tf.where(x>0.5,tf.ones_like(x),tf.zeros_like(x))
    return y, grad


@tf.custom_gradient
def Quantizer(x):
    L=64.
    y=tf.zeros_like(x)
    for l in range(64):
       l = tf.cast(l, tf.float32)
       y=tf.where(tf.math.logical_and(x>=tf.cast(l/(L-1)-0.5/(L-1),tf.float32),x<tf.cast(l/(L-1)+0.5/(L-1),tf.float32)),l*tf.ones_like(x),y)
    def grad(upstream):
        multiplier=(L-1)*tf.ones_like(x)
        return upstream*multiplier
    # y=tf.where(x>0.5,tf.ones_like(x),tf.zeros_like(x))
    return y, grad

@tf.custom_gradient
def Mask(x):
    n=4000
    L=63.
    m=tf.zeros_like(x)
    m=tf.tile(m,[1,n])
    batch= tf.shape(x)[0]
    idx_value = tf.range(n)
    idx_value = tf.expand_dims(idx_value, axis=0)
    idx_value = tf.tile(idx_value, [batch, 1])
    idx_value=tf.cast(idx_value,tf.float32)

    m=tf.where(idx_value<=(n/L)*x, tf.ones_like(m), tf.zeros_like(m))
    def grad(upstream):
        multiplier = tf.where(tf.math.logical_and((tf.math.ceil(idx_value*L/n)>=x-1),(tf.math.ceil(idx_value*L/n)< x+2)), (1/3.)*tf.ones_like(idx_value), tf.zeros_like(idx_value))
        return tf.reduce_sum(multiplier*upstream,keepdims=True)
    return m, grad

class Rate_fun(tf.keras.Model):
    """Build encoder from specified arch"""

    def __init__(self, name="Rate_fun", **kwargs):
        super(Rate_fun, self).__init__(name=name, **kwargs)

        self.sublayers = [
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_1"),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_2"),
            # layers.BatchNormalization(),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_3"),
            layers.Dense(
                1, use_bias=True, activation='sigmoid', name="fc_4")
        ]
        # self.build(input_shape=input_shape)

    def call(self, x, **kwargs):
        for sublayer in self.sublayers:
            x = sublayer(x, **kwargs)
        return x

class Encoder(tf.keras.Model):
    """Build encoder from specified arch"""

    def __init__(self, name="encoder", **kwargs):
        super(Encoder, self).__init__(name=name, **kwargs)

        self.sublayers = [
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_1"),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_2"),
            # layers.BatchNormalization(),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_3"),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_4"),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_5"),
            # layers.BatchNormalization(),
            layers.Dense(
                4000, use_bias=True, activation=None, name="fc_6")
        ]
        # self.build(input_shape=input_shape)

    def call(self, x, **kwargs):
        for sublayer in self.sublayers:
            x = sublayer(x, **kwargs)
        return x

class Decoder(tf.keras.Model):
    """Build encoder from specified arch"""

    def __init__(self, name="decoder", **kwargs):
        super(Decoder, self).__init__(name=name, **kwargs)

        self.sublayers = [
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_1"),
            # layers.BatchNormalization(),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_2"),
            # layers.BatchNormalization(),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_3"),
            # layers.BatchNormalization(),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_4"),
            layers.Dense(
                1024, use_bias=True, activation="leaky_relu", name="fc_5"),
            layers.Dense(
                85, use_bias=True, activation=None, name="fc_6")
        ]
        # self.build(input_shape=input_shape)

    def call(self, x, **kwargs):
        for sublayer in self.sublayers:
            x = sublayer(x, **kwargs)
        return x

class Generator(tf.keras.Model):

    def __init__(self,name='generator'):
        super(Generator, self).__init__(name=name)
        self.config = Config()

        self.enc_shape = self.config.ENCODER_INPUT_SHAPE
        self.resnet50V2 = ResNet50V2(include_top=False, weights=None, input_shape=self.enc_shape, pooling='avg')
        self._set_resnet_arg_scope()
        self.resnet50V2.summary()
        self.resnet50V2.trainable = False
        self.channel = Channel("awgn", -10, name="channel_output")

        self.encoder = Encoder()
        self.decoder = Decoder()
        self.rate_fun = Rate_fun()
        # self.encoder.summary()'imagenet'
        self.regressor = Regressor()
        self.regressor.trainable = False
        self.smpl = Smpl()

    def _set_resnet_arg_scope(self):
        """This method acts similar to TF 1.x contrib's slim `resnet_arg_scope()`.
            It overrides
        """
        vs_initializer = tf.keras.initializers.VarianceScaling(2.0)
        l2_regularizer = tf.keras.regularizers.l2(self.config.GENERATOR_WEIGHT_DECAY)
        for layer in self.resnet50V2.layers:
            if isinstance(layer, layers.Conv2D):
                # original implementations slim `resnet_arg_scope` additionally sets
                # `normalizer_fn` and `normalizer_params` which in TF 2.0 need to be implemented
                # as own layers. This is not possible using keras ResNet50V2 application.
                # Nevertheless this is not needed as training seems to be likely stable.
                # See https://www.tensorflow.org/guide/migrate#a_note_on_slim_contriblayers for more
                # migration insights
                # setattr(layer, 'padding', 'same')
                setattr(layer, 'kernel_initializer', vs_initializer)
                setattr(layer, 'kernel_regularizer', l2_regularizer)
            if isinstance(layer, layers.BatchNormalization):
                setattr(layer, 'momentum', 0.997)
                setattr(layer, 'epsilon', 1e-5)
            # if isinstance(layer, layers.MaxPooling2D):
            #     setattr(layer, 'padding', 'same')

    def call(self, inputs, **kwargs):
        check = inputs.shape[1:] == self.enc_shape
        assert check, 'shape mismatch: should be {} but is {}'.format(self.enc_shape, inputs.shape)

        features = self.resnet50V2(inputs, **kwargs)
        thetas = self.regressor(features, **kwargs)
        thetas = tf.stop_gradient(thetas)

        # self.encoder.summary()
        # prev_chn_gain = None

        outputs = []
        theta_losses = []
        bits = []
        bits_val = []
        rates = []

        for i in range(self.config.ITERATIONS):
            theta = thetas[i, :]

            rate=self.rate_fun(theta,**kwargs)
            bit=Quantizer(rate)
            m=Mask(bit)
            # m=tf.ones_like(m)
            theta_tilde = self.encoder(tf.concat([theta, tf.stop_gradient(rate)],axis=-1), **kwargs)
            # theta_tilde=theta_tilde*m
            theta_tilde, avg_power= self.channel((theta_tilde,m), **kwargs)
            theta_tilde = self.decoder(theta_tilde, **kwargs)

            m_mean, m_var = tf.nn.moments(tf.reduce_sum(m, axis=-1), axes=[0])

            bits.append(m_mean)
            bits_val.append(m_var)
            rates.append(tf.reduce_mean(rate))

            outputs.append(self._compute_output(theta_tilde, **kwargs))
            theta_losses.append(tf.reduce_mean(tf.reduce_sum(tf.square(theta - theta_tilde), axis=-1) / 85.))

        return outputs, rates, bits, bits_val, theta_losses

    def _compute_output(self, theta, **kwargs):
        cams = theta[:, :self.config.NUM_CAMERA_PARAMS]
        pose_and_shape = theta[:, self.config.NUM_CAMERA_PARAMS:]
        vertices, joints_3d, rotations = self.smpl(pose_and_shape, **kwargs)
        joints_2d = model_util.batch_orthographic_projection(joints_3d, cams)
        shapes = theta[:, -self.config.NUM_SHAPE_PARAMS:]

        return tf.tuple([vertices, joints_2d, joints_3d, rotations, shapes, cams])
