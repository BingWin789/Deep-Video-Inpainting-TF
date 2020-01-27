import os
import cv2
import numpy as np
from PIL import Image
import tensorflow as tf
import argparse

from model.vinet import VINet_final
from utils import preprocess_image, preprocess_mask, postprocess_output

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', help='which gpu used.', default=0, type=int)
parser.add_argument('--imagesdir', help='directory to pth image frames.', default='./examples/images', type=str)
parser.add_argument('--masksdir', help='directory to pth masks.', default='./examples/masks', type=str)
parser.add_argument('--resultdir', help='directory to result.', default='./examples/results', type=str)
parser.add_argument('--cropsize', help='crop size, 512 or 256.', default=512, type=int)
args = parser.parse_args()


class Object():
    pass
opt = Object()
opt.crop_size = args.cropsize
opt.double_size = True if opt.crop_size == 512 else False
opt.search_range = 4 # fixed as 4: search range for flow subnetworks
opt.batch_norm = False
opt.t_stride = 3
opt.prev_warp = True
opt.hidden_size = 128  # used in conv lstm
opt.T = 5  # 5 frames needed once
opt.loss_on_raw = False
opt.gpuid = args.gpuid
opt.warm_up_frame_num = 30

ts = opt.t_stride

frames_dir = args.imagesdir
masks_dir = args.masksdir   
result_dir = args.resultdir
ckpt_path = './weights/tensorflow/save_agg_rec_512.ckpt' if opt.crop_size == 512 else './weights/tensorflow/save_agg_rec.ckpt'

frame_names_list = sorted(os.listdir(frames_dir))
frame_names_list = [i.split('.')[0] for i in frame_names_list]
mirror_frames_num = opt.T // 2 * opt.t_stride
frame_names_list = frame_names_list[mirror_frames_num + opt.warm_up_frame_num:0:-1] + frame_names_list + frame_names_list[-mirror_frames_num:]

os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpuid)
# ## build model
# placeholders
masked_image = tf.placeholder(tf.float32, shape=[1, 3, opt.T, opt.crop_size, opt.crop_size])
mask = tf.placeholder(tf.float32, shape=[1, 1, opt.T, opt.crop_size, opt.crop_size])
prev_state_0 = tf.placeholder(tf.float32, shape=[1, opt.hidden_size, opt.crop_size // (8 * 2 if opt.double_size else 1), opt.crop_size // (8 * 2 if opt.double_size else 1)])
prev_state_1 = tf.placeholder(tf.float32, shape=[1, opt.hidden_size, opt.crop_size // (8 * 2 if opt.double_size else 1), opt.crop_size // (8 * 2 if opt.double_size else 1)])
prev_state = (prev_state_0, prev_state_1)
prev_feed = tf.placeholder(tf.float32, shape=[1, 5, opt.crop_size, opt.crop_size])
# model
output, state_0, state_1 = VINet_final(opt, masked_image, mask, prev_state, prev_feed)  # , trainable=False

init_op = tf.group(
    tf.global_variables_initializer(),
    tf.local_variables_initializer()
)

restorer = tf.train.Saver()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True

with tf.Session(config=config) as sess:
    sess.run(init_op)
    restorer.restore(sess, ckpt_path)

    last_mask_np, last_output_np, last_state_0_np, last_state_1_np = None, None, None, None

    print('warming up...')
    for idx in range(mirror_frames_num, (len(frame_names_list) - mirror_frames_num)):
        # read images
        img_prev_2 = cv2.resize(cv2.imread(os.path.join(frames_dir, frame_names_list[idx - 2 * ts] + '.jpg')), (opt.crop_size, opt.crop_size), cv2.INTER_CUBIC)
        img_prev_1 = cv2.resize(cv2.imread(os.path.join(frames_dir, frame_names_list[idx - 1 * ts] + '.jpg')), (opt.crop_size, opt.crop_size), cv2.INTER_CUBIC)
        img_curr = cv2.resize(cv2.imread(os.path.join(frames_dir, frame_names_list[idx] + '.jpg')), (opt.crop_size, opt.crop_size), cv2.INTER_CUBIC)
        img_next_1 = cv2.resize(cv2.imread(os.path.join(frames_dir, frame_names_list[idx + 1 * ts] + '.jpg')), (opt.crop_size, opt.crop_size), cv2.INTER_CUBIC)
        img_next_2 = cv2.resize(cv2.imread(os.path.join(frames_dir, frame_names_list[idx + 2 * ts] + '.jpg')), (opt.crop_size, opt.crop_size), cv2.INTER_CUBIC)
        # read masks
        msk_prev_2 = cv2.resize(cv2.imread(os.path.join(masks_dir, frame_names_list[idx - 2 * ts] + '.png')), (opt.crop_size, opt.crop_size), cv2.INTER_NEAREST)[:, :, 0]
        msk_prev_1 = cv2.resize(cv2.imread(os.path.join(masks_dir, frame_names_list[idx - 1 * ts] + '.png')), (opt.crop_size, opt.crop_size), cv2.INTER_NEAREST)[:, :, 0]
        msk_curr = cv2.resize(cv2.imread(os.path.join(masks_dir, frame_names_list[idx] + '.png')), (opt.crop_size, opt.crop_size), cv2.INTER_NEAREST)[:, :, 0]
        msk_next_1 = cv2.resize(cv2.imread(os.path.join(masks_dir, frame_names_list[idx + 1 * ts] + '.png')), (opt.crop_size, opt.crop_size), cv2.INTER_NEAREST)[:, :, 0]
        msk_next_2 = cv2.resize(cv2.imread(os.path.join(masks_dir, frame_names_list[idx + 2 * ts] + '.png')), (opt.crop_size, opt.crop_size), cv2.INTER_NEAREST)[:, :, 0]

        img_list = [img_prev_2, img_prev_1, img_curr, img_next_1, img_next_2]
        img_input = np.stack(img_list, axis=0) # T*H*W*C
        img_input = np.transpose(img_input, [3, 0, 1, 2])
        img_input = np.expand_dims(img_input, axis=0)
        img_input = preprocess_image(img_input)

        msk_list = [msk_prev_2, msk_prev_1, msk_curr, msk_next_1, msk_next_2]
        msk_list = [preprocess_mask(i) for i in msk_list]
        msk_input = np.stack(msk_list, axis=0) # T*H*W*C
        msk_input = np.transpose(msk_input, [3, 0, 1, 2])
        msk_input_np = np.expand_dims(msk_input, axis=0)

        msk_3 = np.concatenate([msk_input_np] * 3, axis=1)
        masked_image_np = img_input * (1.0 - msk_3)

        if idx == mirror_frames_num:  # for first frame case
            prev_img_np = masked_image_np[:, :, 2, :, :]
            prev_msk_np = msk_input_np[:, :, 2, :, :]
            last_mask_np = prev_msk_np * 0.5  # * 0.5 if opt.double_size
            prev_ones = np.ones_like(prev_msk_np, np.float32)
            prev_feed_np = np.concatenate([prev_img_np, prev_ones, prev_ones*prev_msk_np], axis=1)
            prev_state_0_np = np.zeros([1, opt.hidden_size, opt.crop_size // (8 * 2 if opt.double_size else 1), opt.crop_size // (8 * 2 if opt.double_size else 1)], np.float32)
            prev_state_1_np = np.zeros([1, opt.hidden_size, opt.crop_size // (8 * 2 if opt.double_size else 1), opt.crop_size // (8 * 2 if opt.double_size else 1)], np.float32)
        else:
            prev_img_np = last_output_np
            prev_msk_np = last_mask_np
            if not opt.double_size:
                prev_msk_np = np.zeros_like(prev_msk_np, np.float32)
            last_mask_np = msk_input_np[:, :, 2, :, :] * 0.5  # * 0.5 if opt.double_size
            prev_ones = np.ones_like(prev_msk_np, np.float32)
            prev_feed_np = np.concatenate([prev_img_np, prev_ones, prev_ones*prev_msk_np], axis=1)
            prev_state_0_np = last_state_0_np
            prev_state_1_np = last_state_1_np
        
        last_output_np, last_state_0_np, last_state_1_np = sess.run([output, state_0, state_1],
                        feed_dict={
                            masked_image: masked_image_np,
                            mask: msk_input_np,
                            prev_state_0: prev_state_0_np,
                            prev_state_1: prev_state_1_np,
                            prev_feed: prev_feed_np
                        })
        
        resimg = postprocess_output(last_output_np)
        if idx >= opt.warm_up_frame_num + mirror_frames_num:
            print(frame_names_list[idx])
            cv2.imwrite(os.path.join(result_dir, frame_names_list[idx] + '.png'), resimg)

