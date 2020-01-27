import numpy as np
import tensorflow as tf
import torch
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--gpuid', help='which gpu used.', default=0, type=int)
parser.add_argument('--pthpath', help='path to pth file.', default='', type=str)
parser.add_argument('--ckptpath', help='path to ckpt file.', default='', type=str)
args = parser.parse_args()

def pth_to_ckpt(pthpath, ckptpath, gpuid=0):
    # reference: https://www.zhihu.com/question/317353538/answer/924705912
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpuid)
    var_value = torch.load(pthpath)['state_dict']
    def pthnm_to_tfnm(pthnm, value):
        scopes = pthnm.split('.')
        if scopes[-1] == 'weight':
            tfnm = '/'.join(scopes[:-1])+'/'+'kernel'
            if len(value.shape) == 4:
                value = np.transpose(value, [2, 3, 1, 0])
            elif len(value.shape) == 5:
                value = np.transpose(value, [2, 3, 4, 1, 0])
        elif scopes[-1] == 'bias':
            tfnm = '/'.join(scopes)
        else:
            raise NotImplementedError
        return tfnm, value
    with tf.device('/gpu:0'):
        for k in var_value:
            v = var_value[k].cpu().numpy()
            nm, v = pthnm_to_tfnm(k, v)
            tf.Variable(initial_value=v, name=nm)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        saver.save(sess, ckptpath)

if __name__ == '__main__':
    if not os.path.exists(args.pthpath):
        print('spcify .pth file path first.')
    else:
        pth_to_ckpt(args.pthpath, args.ckptpath, args.gpuid)
        print('convert done.\nckpt file is saved in %s.' % args.ckptpath)
    
