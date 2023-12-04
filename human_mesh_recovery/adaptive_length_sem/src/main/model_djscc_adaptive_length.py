import os
import sys
import time
import gc
# t=gc.isenabled()
# t=gc.get_threshold()
# t=gc.get_count()
import matplotlib.pyplot as plt
# to make run from console for module import
sys.path.append(os.path.abspath(".."))
import ctypes
libc = ctypes.CDLL("libc.so.6")
libc.malloc_trim(0)
# tf INFO and WARNING messages are not printed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["MALLOC_TRIM_THRESHOLD_"] = "0"
os.environ["MALLOC_MMAP_THRESHOLD_"] = "0"


import tensorflow as tf

try:
    from IPython import get_ipython

    ipy_str = str(type(get_ipython()))
    if 'zmqshell' in ipy_str:
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm
except:  # noqa E722
    from tqdm import tqdm

from main.config import Config
from main.dataset import Dataset
from main.discriminator import Discriminator
from main.generator_djscc_adaptive_length import Generator
from main.model_util import batch_align_by_pelvis, batch_compute_similarity_transform, batch_rodrigues

import tensorflow.compat.v1.losses as v1_loss


class ExceptionHandlingIterator:
    """This class was introduced to avoid tensorflow.python.framework.errors_impl.InvalidArgumentError
        thrown while iterating over the zipped datasets.

        One assumption is that the tf records contain one wrongly generated set due to following error message:
            Expected begin[1] in [0, 462], but got -11 [[{{node Slice}}]] [Op:IteratorGetNextSync]
    """

    def __init__(self, iterable):
        self._iter = iter(iterable)

    def __iter__(self):
        return self

    def __next__(self):
        try:
            return self._iter.__next__()
        except StopIteration as e:
            raise e
        except Exception as e:
            print(e)
            return self.__next__()


class Model:

    def __init__(self, display_config=True):
        self.config = Config()
        self.config.save_config()
        if display_config:
            self.config.display()

        self._build_model()
        self._setup_summary()

    def _build_model(self):
        print('building model...\n')

        physical_devices = tf.config.experimental.list_physical_devices('GPU')
        if physical_devices:
            tf.config.experimental.set_memory_growth(physical_devices[0], True)

        gen_input = ((self.config.BATCH_SIZE,) + self.config.ENCODER_INPUT_SHAPE)

        self.generator = Generator()
        self.generator.build(input_shape=gen_input)
        # self.generator.summary()
        self.generator_opt = tf.keras.optimizers.Adam(learning_rate=self.config.GENERATOR_LEARNING_RATE)

        if not self.config.ENCODER_ONLY:
            disc_input = (self.config.BATCH_SIZE, self.config.NUM_JOINTS * 9 + self.config.NUM_SHAPE_PARAMS)

            self.discriminator = Discriminator()
            self.discriminator.build(input_shape=disc_input)
            self.discriminator_opt = tf.keras.optimizers.Adam(learning_rate=self.config.DISCRIMINATOR_LEARNING_RATE)

        # setup checkpoint
        self.checkpoint_prefix = os.path.join(self.config.LOG_DIR, "ckpt")
        if not self.config.ENCODER_ONLY:
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                             discriminator=self.discriminator,
                                             generator_opt=self.generator_opt,
                                             discriminator_opt=self.discriminator_opt)
        else:
            checkpoint = tf.train.Checkpoint(generator=self.generator,
                                             generator_opt=self.generator_opt)

        self.checkpoint_manager = tf.train.CheckpointManager(checkpoint, self.config.LOG_DIR, max_to_keep=100)

        # if a checkpoint exists, restore the latest checkpoint.
        self.restore_check = None
        if self.checkpoint_manager.latest_checkpoint:
            restore_path = self.config.RESTORE_PATH
            if restore_path is None:
                restore_path = self.checkpoint_manager.latest_checkpoint

            self.restore_check = checkpoint.restore(restore_path).expect_partial()
            print('Checkpoint restored from {}'.format(restore_path))

    def _setup_summary(self):
        # self.summary_path = os.path.join(self.config.LOG_DIR, 'hmr2.0', '3D_{}'.format(self.config.USE_3D))
        # self.summary_writer = tf.summary.create_file_writer(self.summary_path)

        self.generator_loss_log = tf.keras.metrics.Mean('generator_loss', dtype=tf.float32)
        self.kp2d_loss_log = tf.keras.metrics.Mean('kp2d_loss', dtype=tf.float32)
        self.gen_disc_loss_log = tf.keras.metrics.Mean('gen_disc_loss', dtype=tf.float32)

        self.theta_loss_log = tf.keras.metrics.Mean('theta_loss', dtype=tf.float32)
        self.theta_loss_log_test = tf.keras.metrics.Mean('theta_loss_test', dtype=tf.float32)
        if self.config.USE_3D:
            self.kp3d_loss_log = tf.keras.metrics.Mean('kp3d_loss', dtype=tf.float32)
            self.pose_shape_loss_log = tf.keras.metrics.Mean('pose_shape_loss', dtype=tf.float32)

        self.discriminator_loss_log = tf.keras.metrics.Mean('discriminator_loss', dtype=tf.float32)
        self.disc_real_loss_log = tf.keras.metrics.Mean('disc_real_loss', dtype=tf.float32)
        self.disc_fake_loss_log = tf.keras.metrics.Mean('disc_fake_loss', dtype=tf.float32)

        self.kp2d_mpjpe_log = tf.keras.metrics.Mean('kp2d_mpjpe', dtype=tf.float32)
        self.kp3d_mpjpe_log = tf.keras.metrics.Mean('kp3d_mpjpe', dtype=tf.float32)
        self.kp3d_mpjpe_aligned_log = tf.keras.metrics.Mean('kp3d_mpjpe_aligned', dtype=tf.float32)

        self.rate_loss_log = tf.keras.metrics.Mean('rate_loss', dtype=tf.float32)

        self.bit_train_log = tf.keras.metrics.Mean('bit_train', dtype=tf.float32)
        self.bit_test_log = tf.keras.metrics.Mean('bit_test', dtype=tf.float32)

        self.bit_var_train_log = tf.keras.metrics.Mean('bit_var_train', dtype=tf.float32)
        self.bit_var_test_log = tf.keras.metrics.Mean('bit_var_test', dtype=tf.float32)

    ############################################################
    #  Train/Val
    ############################################################

    def train(self):
        # Place tensors on the CPU
        # with tf.device('/CPU:0'):
        dataset = Dataset()

        with tf.device('/CPU:0'):
            ds_train = dataset.get_train()
            ds_smpl = dataset.get_smpl()
            ds_val = dataset.get_val()

        start = 1
        if self.config.RESTORE_EPOCH:
            start = self.config.RESTORE_EPOCH

        for epoch in range(start, self.config.EPOCHS + 1):
            # gc.get_threshold()
            # t = gc.get_count()
            gc.collect()
            start = time.time()
            print('Start of Epoch {}'.format(epoch))

            dataset_train = ExceptionHandlingIterator(tf.data.Dataset.zip((ds_train, ds_smpl)))
            total = int(self.config.NUM_TRAINING_SAMPLES / self.config.BATCH_SIZE)
            count=0

            for image_data, theta in tqdm(dataset_train, total=total, position=0, desc='training'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._train_step(images, kp2d, kp3d, has3d, theta)
                count=count+1
                if count%100==0:
                    print(self.bit_train_log.result())
                # t = gc.get_threshold()
                # t = gc.get_count()
                # gc.collect()

            self._log_train(epoch=epoch)

            total = int(self.config.NUM_VALIDATION_SAMPLES / self.config.BATCH_SIZE)

            for image_data in tqdm(ds_val, total=total, position=0, desc='validate'):
                images, kp2d, kp3d, has3d = image_data[0], image_data[1], image_data[2], image_data[3]
                self._val_step(images, kp2d, kp3d, has3d)
                # t = gc.get_threshold()
                # t = gc.get_count()
                # gc.collect()

            self._log_val(epoch=epoch)

            print('Time taken for epoch {} is {} sec\n'.format(epoch, time.time() - start))

            # saving (checkpoint) the model every 5 epochs
            if epoch % 1 == 0:
                print('saving checkpoint\n')
                self.checkpoint_manager.save(epoch)
            # gc.collect()
            # K.clear_session()

        # self.summary_writer.flush()
        self.checkpoint_manager.save(self.config.EPOCHS + 1)

    @tf.function
    def _train_step(self, images, kp2d, kp3d, has3d, theta):
        tf.keras.backend.set_learning_phase(1)
        batch_size = images.shape[0]

        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            # generator_outputs, rate = self.generator(images, training=True)
            generator_outputs, rates, bits, bits_var, theta_losses = self.generator(images, training=True)
            # only use last computed theta (from iterative feedback loop)

            _, kp2d_pred, kp3d_pred, pose_pred, shape_pred, _ = generator_outputs[-1]

            vis = tf.expand_dims(kp2d[:, :, 2], -1)
            kp2d_loss = v1_loss.absolute_difference(kp2d[:, :, :2], kp2d_pred, weights=vis)
            kp2d_loss = kp2d_loss * self.config.GENERATOR_2D_LOSS_WEIGHT
            rate_loss= rates[-1]*self.config.Rate_Lambda*self.config.GENERATOR_RATE_LOSS_WEIGHT
            theta_loss=theta_losses[-1]*self.config.GENERATOR_theta_LOSS_WEIGHT

            if self.config.USE_3D:
                has3d = tf.expand_dims(has3d, -1)

                kp3d_real = batch_align_by_pelvis(kp3d)
                kp3d_pred = batch_align_by_pelvis(kp3d_pred[:, :self.config.NUM_KP3D, :])

                kp3d_real = tf.reshape(kp3d_real, [batch_size, -1])
                kp3d_pred = tf.reshape(kp3d_pred, [batch_size, -1])

                kp3d_loss = v1_loss.mean_squared_error(kp3d_real, kp3d_pred, weights=has3d) * 0.5
                kp3d_loss = kp3d_loss * self.config.GENERATOR_3D_LOSS_WEIGHT

                """Calculating pose and shape loss basically makes no sense
                    due to missing paired 3d and mosh ground truth data.
                    The original implementation has paired data for Human 3.6 M dataset
                    which was not published due to licence conflict.
                    Nevertheless with SMPLify paired data can be generated
                    (see http://smplify.is.tue.mpg.de/ for more information)
                """
                pose_pred = tf.reshape(pose_pred, [batch_size, -1])
                shape_pred = tf.reshape(shape_pred, [batch_size, -1])
                pose_shape_pred = tf.concat([pose_pred, shape_pred], 1)

                # fake ground truth
                has_smpl=tf.zeros(batch_size,tf.float32)
                has_smpl=tf.expand_dims(has_smpl,-1)
                pose_shape_real=tf.zeros(pose_shape_pred.shape)

                # pose_shape_real = tf.zeros(pose_shape_pred.shape)

                ps_loss = v1_loss.mean_squared_error(pose_shape_real, pose_shape_pred, weights=has_smpl) * 0.5
                ps_loss = ps_loss * self.config.GENERATOR_3D_LOSS_WEIGHT

            # use all poses and shapes from iterative feedback loop
            fake_disc_input = self.accumulate_fake_disc_input(generator_outputs)
            fake_disc_output = self.discriminator(fake_disc_input, training=True)

            real_disc_input = self.accumulate_real_disc_input(theta)
            real_disc_output = self.discriminator(real_disc_input, training=True)

            gen_disc_loss = tf.reduce_mean(tf.reduce_sum((fake_disc_output - 1) ** 2, axis=1))
            gen_disc_loss = gen_disc_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT

            generator_loss = tf.reduce_sum([kp2d_loss, gen_disc_loss, rate_loss, theta_loss])
            if self.config.USE_3D:
                generator_loss = tf.reduce_sum([generator_loss, kp3d_loss, ps_loss])

            disc_real_loss = tf.reduce_mean(tf.reduce_sum((real_disc_output - 1) ** 2, axis=1))
            disc_fake_loss = tf.reduce_mean(tf.reduce_sum(fake_disc_output ** 2, axis=1))

            discriminator_loss = tf.reduce_sum([disc_real_loss, disc_fake_loss])
            discriminator_loss = discriminator_loss * self.config.DISCRIMINATOR_LOSS_WEIGHT

        generator_grads = gen_tape.gradient(generator_loss, self.generator.trainable_variables)
        discriminator_grads = disc_tape.gradient(discriminator_loss, self.discriminator.trainable_variables)
        self.generator_opt.apply_gradients(zip(generator_grads, self.generator.trainable_variables))
        self.discriminator_opt.apply_gradients(zip(discriminator_grads, self.discriminator.trainable_variables))

        self.generator_loss_log.update_state(generator_loss)
        self.kp2d_loss_log.update_state(kp2d_loss)
        self.gen_disc_loss_log.update_state(gen_disc_loss)
        self.rate_loss_log.update_state(rate_loss)
        self.bit_train_log.update_state(tf.reduce_mean(bits[-1]))
        self.bit_var_train_log.update_state(tf.reduce_mean(bits_var[-1]))
        self.theta_loss_log.update_state(theta_loss)

        if self.config.USE_3D:
            self.kp3d_loss_log.update_state(kp3d_loss)
            self.pose_shape_loss_log.update_state(ps_loss)

        self.discriminator_loss_log.update_state(discriminator_loss)
        self.disc_real_loss_log.update_state(disc_real_loss)
        self.disc_fake_loss_log.update_state(disc_fake_loss)

    def accumulate_fake_disc_input(self, generator_outputs):
        fake_poses, fake_shapes = [], []
        for output in generator_outputs:
            fake_poses.append(output[3])
            fake_shapes.append(output[4])
        # ignore global rotation
        fake_poses = tf.reshape(tf.convert_to_tensor(fake_poses), [-1, self.config.NUM_JOINTS_GLOBAL, 9])[:, 1:, :]
        fake_poses = tf.reshape(fake_poses, [-1, self.config.NUM_JOINTS * 9])
        fake_shapes = tf.reshape(tf.convert_to_tensor(fake_shapes), [-1, self.config.NUM_SHAPE_PARAMS])

        fake_disc_input = tf.concat([fake_poses, fake_shapes], 1)
        return fake_disc_input

    def accumulate_real_disc_input(self, theta):
        real_poses = theta[:, :self.config.NUM_POSE_PARAMS]
        # compute rotations matrices for [batch x K x 9] - ignore global rotation
        real_poses = batch_rodrigues(real_poses)[:, 1:, :]
        real_poses = tf.reshape(real_poses, [-1, self.config.NUM_JOINTS * 9])
        real_shapes = theta[:, -self.config.NUM_SHAPE_PARAMS:]

        real_disc_input = tf.concat([real_poses, real_shapes], 1)
        return real_disc_input

    def _log_train(self, epoch):
        template = 'Generator Loss: {}, Discriminator Loss: {}'
        print(template.format(self.generator_loss_log.result(), self.discriminator_loss_log.result()))
        if self.config.USE_3D:
            template = 'theta_loss: {}, kp2d_loss: {}, rate_loss: {}, kp3d_loss: {}'
            print(template.format(self.theta_loss_log.result(), self.kp2d_loss_log.result(), self.rate_loss_log.result(), self.kp3d_loss_log.result()))
        else:
            template = 'kp2d_loss: {}, rate_loss: {}'
            print(
                template.format(self.kp2d_loss_log.result(), self.rate_loss_log.result()))
        print('bit: {}, bit_var:{}'.format(self.bit_train_log.result(),self.bit_var_train_log.result()))

        # with self.summary_writer.as_default():
        #     tf.summary.scalar('generator_loss', self.generator_loss_log.result(), step=epoch)
        #     tf.summary.scalar('kp2d_loss', self.kp2d_loss_log.result(), step=epoch)
        #     tf.summary.scalar('gen_disc_loss', self.gen_disc_loss_log.result(), step=epoch)
        #     tf.summary.scalar('rate_loss', self.rate_loss_log.result(), step=epoch)
        #     tf.summary.scalar('theta_loss', self.theta_loss_log.result(), step=epoch)
        #
        #     if self.config.USE_3D:
        #         tf.summary.scalar('kp3d_loss', self.kp3d_loss_log.result(), step=epoch)
        #         tf.summary.scalar('pose_shape_loss', self.pose_shape_loss_log.result(), step=epoch)
        #
        #     tf.summary.scalar('discriminator_loss', self.discriminator_loss_log.result(), step=epoch)
        #     tf.summary.scalar('disc_real_loss', self.disc_real_loss_log.result(), step=epoch)
        #     tf.summary.scalar('disc_fake_loss', self.disc_fake_loss_log.result(), step=epoch)

        self.generator_loss_log.reset_states()
        self.kp2d_loss_log.reset_states()
        self.gen_disc_loss_log.reset_states()
        self.rate_loss_log.reset_states()
        self.theta_loss_log.reset_states()

        if self.config.USE_3D:
            self.kp3d_loss_log.reset_states()
            self.pose_shape_loss_log.reset_states()

        self.discriminator_loss_log.reset_states()
        self.disc_real_loss_log.reset_states()
        self.disc_fake_loss_log.reset_states()
        self.bit_train_log.reset_states()
        self.bit_var_train_log.reset_states()

    @tf.function
    def _val_step(self, images, kp2d, kp3d, has3d):
        tf.keras.backend.set_learning_phase(0)


        results, rates, bits, bits_var, theta_losses = self.generator(images, training=False)
        # only use last computed theta (from accumulated iterative feedback loop)
        _, kp2d_pred, kp3d_pred, _, _, _ = results[-1]
        theta_loss_test = theta_losses[-1]
        vis = kp2d[:, :, 2]
        kp2d_norm = tf.norm(kp2d_pred[:, :self.config.NUM_KP2D, :] - kp2d[:, :, :2], axis=2) * vis
        kp2d_mpjpe = tf.reduce_sum(kp2d_norm) / tf.reduce_sum(vis)
        self.theta_loss_log_test(theta_loss_test)
        self.kp2d_mpjpe_log(kp2d_mpjpe)

        self.bit_test_log(tf.reduce_mean(bits[-1]))
        self.bit_var_test_log(tf.reduce_mean(bits_var[-1]))

        if self.config.USE_3D:
            # check if at least one 3d sample available
            if tf.reduce_sum(has3d) > 0:
                kp3d_real = tf.boolean_mask(kp3d, has3d)
                kp3d_predict = tf.boolean_mask(kp3d_pred, has3d)
                kp3d_predict = kp3d_predict[:, :self.config.NUM_KP3D, :]

                kp3d_real = batch_align_by_pelvis(kp3d_real)
                kp3d_predict = batch_align_by_pelvis(kp3d_predict)

                kp3d_mpjpe = tf.norm(kp3d_predict - kp3d_real, axis=2)
                kp3d_mpjpe = tf.reduce_mean(kp3d_mpjpe)

                aligned_kp3d = batch_compute_similarity_transform(kp3d_real, kp3d_predict)
                kp3d_mpjpe_aligned = tf.norm(aligned_kp3d - kp3d_real, axis=2)
                kp3d_mpjpe_aligned = tf.reduce_mean(kp3d_mpjpe_aligned)

                self.kp3d_mpjpe_log.update_state(kp3d_mpjpe)
                self.kp3d_mpjpe_aligned_log.update_state(kp3d_mpjpe_aligned)

    def _log_val(self, epoch):
        print('MPJPE kp2d: {}'.format(self.kp2d_mpjpe_log.result()))
        print('bit: {}, bit_var:{}'.format(self.bit_test_log.result(),self.bit_var_test_log.result()))
        print('theta_loss: {}'.format(self.theta_loss_log_test.result()))
        if self.config.USE_3D:
            print('MPJPE kp3d: {}, MPJPE kp3d aligned: {}'.format(self.kp3d_mpjpe_log.result(),
                                                                  self.kp3d_mpjpe_aligned_log.result()))

        # with self.summary_writer.as_default():
        #     tf.summary.scalar('kp2d_mpjpe', self.kp2d_mpjpe_log.result(), step=epoch)
        #     if self.config.USE_3D:
        #         tf.summary.scalar('kp3d_mpjpe', self.kp3d_mpjpe_log.result(), step=epoch)
        #         tf.summary.scalar('kp3d_mpjpe_aligned', self.kp3d_mpjpe_aligned_log.result(), step=epoch)

        self.kp2d_mpjpe_log.reset_states()
        self.bit_test_log.reset_states()
        self.bit_var_test_log.reset_states()
        if self.config.USE_3D:
            self.kp3d_mpjpe_log.reset_states()
            self.kp3d_mpjpe_aligned_log.reset_states()

    ############################################################
    #  Test
    ############################################################

    def test(self, return_kps=True):
        """Run evaluation of the model
        Specify LOG_DIR to point to the saved checkpoint directory

        Args:
            return_kps: set to return keypoints - default = False
        """

        if self.restore_check is None:
            raise RuntimeError('restore did not succeed, pleas check if you set config.LOG_DIR correctly')

        if self.config.INITIALIZE_CUSTOM_REGRESSOR:
            self.restore_check.assert_nontrivial_match()
        else:
            self.restore_check.assert_existing_objects_matched().assert_nontrivial_match()

        # Place tensors on the CPU
        with tf.device('/CPU:0'):
            dataset = Dataset()
            ds_test = dataset.get_test()

        start = time.time()
        # print('Start of Testing')
        # prior_learnt=self.generator.compression_trainer.prior
        # _ = tf.linspace(-6., 6., 501)[:, None]
        # plt.plot(_, prior_learnt.prob(_))
        # plt.savefig('learnt_prior_latent10.png')
        # # plt.show()

        # mpjpe, mpjpe_aligned, kps2d_pred, kps2d_real = [], [], [], []
        mpjpe, kps2d_pred, kps2d_real,mpjpe_kp3d,mpjpe_kp3d_aligned, kps3d_pred, kps3d_real, bit_length = [], [], [],[], [], [], [], []

        total = int(self.config.NUM_TEST_SAMPLES / self.config.BATCH_SIZE)
        for image_data in tqdm(ds_test, total=total, position=0, desc='testing'):
            image, kp2d, kp3d, has_3d = image_data[0], image_data[1], image_data[2], image_data[3]
            # kp2d_mpjpe, kp2d_mpjpe_aligned, predict_kp2d, real_kp2d = self._test_step(image, kp2d[:,:,:2], return_kps=return_kps)
            kp2d_mpjpe, predict_kp2d, real_kp2d,kp3d_mpjpe,kp3d_mpjpe_aligned, predict_kp3d, real_kp3d, bit = self._test_step(image, kp2d, kp3d, has_3d, return_kps=return_kps)
            if return_kps:
                kps2d_pred.append(predict_kp2d)
                kps2d_real.append(real_kp2d)
                kps3d_pred.append(predict_kp3d)
                kps3d_real.append(real_kp3d)

            mpjpe.append(kp2d_mpjpe)
            mpjpe_kp3d.append(kp3d_mpjpe)
            mpjpe_kp3d_aligned.append(kp3d_mpjpe_aligned)
            bit_length.append(bit)
            # sequences.append(sequence)

        print('Time taken for testing {} sec\n'.format(time.time() - start))

        def convert(tensor, num=None, is_kp=False):
            if num is None:
                num = self.config.NUM_KP2D
            if is_kp:
                return tf.squeeze(tf.reshape(tf.stack(tensor), [-1, num, 3]))

            return tf.squeeze(tf.reshape(tf.stack(tensor), [-1, num]))

        # mpjpe, mpjpe_aligned= convert(mpjpe), convert(mpjpe_aligned)
        # result_dict = {"kp2d_mpjpe": convert(mpjpe), "kp2d_mpjpe_aligned": mpjpe_aligned}
        result_dict = {"kp2d_mpjpe": tf.reduce_mean(tf.stack(mpjpe)),
                       "kp3d_mpjpe": tf.reduce_mean(tf.stack(mpjpe_kp3d)),
                       "kp3d_mpjpe_aligned": tf.reduce_mean(tf.stack(mpjpe_kp3d_aligned)),
                       "bits": tf.reduce_mean(tf.stack(bit_length))}

        if return_kps:
            kps2d_pred=tf.squeeze(tf.reshape(tf.stack(kps2d_pred), [-1, self.config.NUM_KP2D, 2]))
            kps2d_real=tf.squeeze(tf.reshape(tf.stack(kps2d_real), [-1, self.config.NUM_KP2D, 3]))
            kps3d_pred = tf.squeeze(tf.reshape(tf.stack(kps3d_pred), [-1, self.config.NUM_KP2D, 3]))
            kps3d_real = tf.squeeze(tf.reshape(tf.stack(kps3d_real), [-1, self.config.NUM_KP2D, 3]))
            # kps2d_pred, kps2d_real = convert(kps2d_pred, is_kp=True), convert(kps2d_real, is_kp=True)
            result_dict.update({'kps2d_pred': kps2d_pred, 'kps2d_real': kps2d_real,'kps3d_pred': kps3d_pred, 'kps3d_real': kps3d_real})

        return result_dict

    @tf.function
    def _test_step(self, image, kp2d, kp3d, has3d, return_kps=False):
        tf.keras.backend.set_learning_phase(0)

        if len(tf.shape(image)) != 4:
            image = tf.expand_dims(image, 0)
            kp2d = tf.expand_dims(kp2d, 0)

        results, rates, bits, bits_var, theta_losses = self.generator(image, training=False)

        # only use last computed theta (from accumulated iterative feedback loop)
        _, kp2d_pred, kp3d_pred, _, _, _ = results[-1]
        vis = kp2d[:, :, 2]
        kp2d_norm = tf.norm(kp2d_pred[:, :self.config.NUM_KP2D, :] - kp2d[:, :, :2], axis=2) * vis
        kp2d_mpjpe = tf.reduce_sum(kp2d_norm) / tf.reduce_sum(vis)
        kp3d_mpjpe=0.
        kp3d_mpjpe_aligned=0.
        if self.config.USE_3D:
            # check if at least one 3d sample available
            if tf.reduce_sum(has3d) > 0:
                kp3d_real = tf.boolean_mask(kp3d, has3d)
                kp3d_predict = tf.boolean_mask(kp3d_pred, has3d)
                kp3d_predict = kp3d_predict[:, :self.config.NUM_KP3D, :]

                kp3d_real = batch_align_by_pelvis(kp3d_real)
                kp3d_predict = batch_align_by_pelvis(kp3d_predict)

                kp3d_mpjpe = tf.norm(kp3d_predict - kp3d_real, axis=2)
                kp3d_mpjpe = tf.reduce_mean(kp3d_mpjpe)

                aligned_kp3d = batch_compute_similarity_transform(kp3d_real, kp3d_predict)
                kp3d_mpjpe_aligned = tf.norm(aligned_kp3d - kp3d_real, axis=2)
                kp3d_mpjpe_aligned = tf.reduce_mean(kp3d_mpjpe_aligned)

                # self.kp3d_mpjpe_log.update_state(kp3d_mpjpe)
                # self.kp3d_mpjpe_aligned_log.update_state(kp3d_mpjpe_aligned)

        if return_kps:
            return kp2d_mpjpe,  kp2d_pred,  kp2d, kp3d_mpjpe, kp3d_mpjpe_aligned, kp3d_pred,  kp3d, bits[-1]

        return kp2d_mpjpe,  None, None, kp3d_mpjpe, kp3d_mpjpe_aligned, None, None, bits[-1]

        # factor = tf.constant(1000, tf.float32)
        # kp2d, kp2d_predict = kp2d * factor, kp2d_pred * factor  # convert back from m -> mm
        # kp2d_predict = kp2d_predict[:, :self.config.NUM_KP2D, :]
        #
        # real_kp2d = batch_align_by_pelvis(kp2d)
        # predict_kp2d = batch_align_by_pelvis(kp2d_predict)
        #
        # kp2d_mpjpe = tf.norm(real_kp2d - predict_kp2d, axis=2)
        #
        # aligned_kp2d = batch_compute_similarity_transform(real_kp2d, predict_kp2d)
        # kp2d_mpjpe_aligned = tf.norm(real_kp2d - aligned_kp2d, axis=2)

        # if return_kps:
        #     return kp2d_mpjpe, kp2d_mpjpe_aligned, predict_kp2d, real_kp2d
        #
        # return kp2d_mpjpe, kp2d_mpjpe_aligned, None, None

    ############################################################
    #  Detect/Single Inference
    ############################################################


    def detect(self, image):
        tf.keras.backend.set_learning_phase(0)

        if self.restore_check is None:
            raise RuntimeError('restore did not succeed, pleas check if you set config.LOG_DIR correctly')

        if self.config.INITIALIZE_CUSTOM_REGRESSOR:
            self.restore_check.assert_nontrivial_match()
        else:
            self.restore_check.assert_existing_objects_matched().assert_nontrivial_match()

        if len(tf.shape(image)) != 4:
            image = tf.expand_dims(image, 0)

        result = self.generator(image, training=False)

        vertices_pred, kp2d_pred, kp3d_pred, pose_pred, shape_pred, cam_pred = result[-1]
        result_dict = {
            "vertices": tf.squeeze(vertices_pred),
            "kp2d": tf.squeeze(kp2d_pred),
            "kp3d": tf.squeeze(kp3d_pred),
            "pose": tf.squeeze(pose_pred),
            "shape": tf.squeeze(shape_pred),
            "cam": tf.squeeze(cam_pred)
        }
        return result_dict


if __name__ == '__main__':
    model = Model()
    result=model.test()
    print(result["kp2d_mpjpe"])
    print(result["kp3d_mpjpe"])
    print(result["kp3d_mpjpe_aligned"])
    print(result["bits"])
    # model.train()
