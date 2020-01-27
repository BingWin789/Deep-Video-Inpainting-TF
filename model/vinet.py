import tensorflow as tf
from .gated_conv import conv2d, GatedConvolution, GatedUpConvolution2D, DATA_FORMAT, Upsample, resize_bilinear, resize_nearest
from .flow_modules import LongFlowNetCorr, WarpingLayer, MaskEstimator_
from .ConvLSTM import ConvLSTM


def VI_2D_Encoder_3(inputs, opt, reuse=False, trainable=True, use_bias=False, namescope='VI_2D_Encoder_3'):
    with tf.variable_scope(namescope, reuse=reuse):
        st = 2 if opt.double_size else 1
        out_1 = GatedConvolution(inputs, 32, k_size=(3, 3), strides=(st, st), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec0')
        ec1 = GatedConvolution(out_1, 64, k_size=(3, 3), strides=(2, 2), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec1')
        out_2 = GatedConvolution(ec1, 64, k_size=(3, 3), strides=(1, 1), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec2')
        ec3_1 = GatedConvolution(out_2, 96, k_size=(3, 3), strides=(2, 2), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec3_1')
        out_4 = GatedConvolution(ec3_1, 96, k_size=(3, 3), strides=(1, 1), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec3_2')
        ec4_1 = GatedConvolution(out_4, 128, k_size=(3, 3), strides=(2, 2), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec4_1')
        ec4 = GatedConvolution(ec4_1, 128, k_size=(3, 3), strides=(1, 1), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec4')
        out = GatedConvolution(ec4, 128, k_size=(3, 3), strides=(1, 1), trainable=trainable, 
                                 padding='SAME', use_bias=use_bias, convtype='2d', namescope='ec5')
    return out, out_4, out_2, out_1

def VI_2D_Decoder_3(inputs, opt, x2_64_warp=None, x2_128_warp=None, reuse=False, trainable=True, use_bias=False, namescope='VI_2D_Decoder_3'):
    with tf.variable_scope(namescope, reuse=reuse):
        dv = 2 if opt.double_size else 1
        dc0 = GatedConvolution(inputs, 128, k_size=(1, 3, 3), strides=(1, 1, 1), convtype='3d', 
                               trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc0')
        dc1 = GatedConvolution(dc0, 128, k_size=(1, 3, 3), strides=(1, 1, 1),  convtype='3d', 
                               trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc1')
        # Upconv
        dc1_1 = GatedUpConvolution2D(dc1, 96, (opt.crop_size // 4 // dv, opt.crop_size // 4 // dv), k_size=(1, 3, 3), trainable=trainable, 
                                   convtype='3d', strides=(1, 1, 1), padding='SAME', use_bias=use_bias, namescope='dc1_1')
        
        x1_x2_64 = tf.concat([dc1_1, x2_64_warp], axis=1) if x2_64_warp is not None else dc1_1
        
        dc2_1 = GatedConvolution(x1_x2_64, 96, k_size=(1, 3, 3), strides=(1, 1, 1),  convtype='3d', 
                                 trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc2_1')
        dc2_bt1 = GatedConvolution(dc2_1, 96, k_size=(1, 3, 3), strides=(1, 1, 1), dilation=(1, 2, 2),  convtype='3d', 
                                 trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc2_bt1')
        dc2_bt2 = GatedConvolution(dc2_bt1, 96, k_size=(1, 3, 3), strides=(1, 1, 1), dilation=(1, 4, 4),  convtype='3d', 
                                 trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc2_bt2')
        dc2_bt3 = GatedConvolution(dc2_bt2, 96, k_size=(1, 3, 3), strides=(1, 1, 1), dilation=(1, 8, 8),  convtype='3d', 
                                 trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc2_bt3')

        # Upconv
        dc2_2 = GatedUpConvolution2D(dc2_bt3, 64, (opt.crop_size//2//dv, opt.crop_size//2//dv), k_size=(1, 3, 3), trainable=trainable, 
                                   convtype='3d',  strides=(1, 1, 1), padding='SAME', use_bias=use_bias, namescope='dc2_2')

        x1_x2_128 = tf.concat([dc2_2, x2_128_warp], axis=1) if x2_128_warp is not None else dc2_2
        dc3_1 = GatedConvolution(x1_x2_128, 64, k_size=(1, 3, 3), strides=(1, 1, 1),  convtype='3d', 
                                trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc3_1')
        dc3_2 = GatedConvolution(dc3_1, 64, k_size=(1, 3, 3), strides=(1, 1, 1),  convtype='3d', 
                                trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc3_2')

        # Upconv
        dc4 = GatedUpConvolution2D(dc3_2, 32, (opt.crop_size//dv, opt.crop_size//dv), k_size=(1, 3, 3), strides=(1, 1, 1),
                                convtype='3d', trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc4')
        if opt.double_size:
            dc4 = Upsample(dc4, (opt.crop_size, opt.crop_size), is3d=True)

        dc5 = GatedConvolution(dc4, 16, k_size=(1, 3, 3), strides=(1, 1, 1),  convtype='3d', 
                                trainable=trainable, padding='SAME', use_bias=use_bias, namescope='dc5')
        dc6 = tf.layers.conv3d(dc5, 3, kernel_size=(1, 3, 3), strides=(1, 1, 1), data_format=DATA_FORMAT, 
                                trainable=trainable, padding='SAME', use_bias=use_bias, kernel_initializer=tf.keras.initializers.he_normal(), name='dc6')
    return dc6, None

def VI_2D_BottleNeck(inputs, reuse=False, trainable=True, use_bias=False, namescope='VI_2D_Decoder_3'):
    with tf.variable_scope(namescope, reuse=reuse):
        bt0 = GatedConvolution(inputs, 128, k_size=(3, 3), strides=(1, 1), dilation=(1, 1), 
                                trainable=trainable, padding='SAME', use_bias=use_bias, convtype='2d', namescope='bt0')
        bt1 = GatedConvolution(bt0, 128, k_size=(3, 3), strides=(1, 1), dilation=(2, 2), 
                                trainable=trainable, padding='SAME', use_bias=use_bias, convtype='2d', namescope='bt1')
        bt2 = GatedConvolution(bt1, 128, k_size=(3, 3), strides=(1, 1), dilation=(4, 4), 
                                trainable=trainable, padding='SAME', use_bias=use_bias, convtype='2d', namescope='bt2')
        bt3 = GatedConvolution(bt2, 128, k_size=(3, 3), strides=(1, 1), dilation=(8, 8), 
                                trainable=trainable, padding='SAME', use_bias=use_bias, convtype='2d', namescope='bt3')
    return bt3

def VI_Aggregator(inputs, depth, reuse=False, trainable=True, use_bias=False,  namescope='VI_Aggregator'):
    with tf.variable_scope(namescope, reuse=reuse):
        ch = inputs.get_shape().as_list()[1]
        # depth = inputs.get_shape().as_list()[2]
        stagg = GatedConvolution(inputs, ch, k_size=(depth, 3, 3), strides=(depth, 1, 1), use_bias=use_bias, padding='SAME',
                                 trainable=trainable, convtype='3d', namescope='stAgg')
    return stagg


def VINet_final(opt, masked_img, mask, prev_state=None, prev_feed=None, idx=0, trainable=True, namescope='module'):
    with tf.variable_scope(namescope):
        # masked_img: b x C x TxHxW
        T = masked_img.get_shape().as_list()[2]
        ref_idx = (T-1)//2
        ones = tf.ones_like(mask)
        # encoder
        enc_output = []
        enc_input = tf.concat([masked_img, ones, ones*mask], axis=1)
        # print(type(enc_input[:,:,ref_idx,:,:]), type(mask))
        # input('hereeeee')

        f1, f1_64, f1_128, f1_256 = VI_2D_Encoder_3(enc_input[:,:,ref_idx,:,:], opt, reuse=False, trainable=trainable, namescope='encoder1')
        
        f2, f2_64, f2_128, _ = VI_2D_Encoder_3(enc_input[:,:,ref_idx-2,:,:], opt, reuse=False, trainable=trainable, namescope='encoder2')
        f3, f3_64, f3_128, _ = VI_2D_Encoder_3(enc_input[:,:,ref_idx-1,:,:], opt, reuse=True, trainable=trainable, namescope='encoder2')
        f4, f4_64, f4_128, _ = VI_2D_Encoder_3(enc_input[:,:,ref_idx+1,:,:], opt, reuse=True, trainable=trainable, namescope='encoder2')
        f5, f5_64, f5_128, _ = VI_2D_Encoder_3(enc_input[:,:,ref_idx+2,:,:], opt, reuse=True, trainable=trainable, namescope='encoder2')
        f6, f6_64, f6_128, f6_256 = VI_2D_Encoder_3(prev_feed, opt, reuse=True, trainable=trainable, namescope='encoder2')

        flow2 = LongFlowNetCorr(f1, f2, opt, trainable=trainable, reuse=False, namescope='flownet')
        flow3 = LongFlowNetCorr(f1, f3, opt, trainable=trainable, reuse=True, namescope='flownet')
        flow4 = LongFlowNetCorr(f1, f4, opt, trainable=trainable, reuse=True, namescope='flownet')
        flow5 = LongFlowNetCorr(f1, f5, opt, trainable=trainable, reuse=True, namescope='flownet')
        flow6 = LongFlowNetCorr(f1, f6, opt, trainable=trainable, reuse=True, namescope='flownet')

        f2_warp = WarpingLayer(f2, flow2)
        f3_warp = WarpingLayer(f3, flow3)
        f4_warp = WarpingLayer(f4, flow4)
        f5_warp = WarpingLayer(f5, flow5)
        f6_warp = WarpingLayer(f6, flow6)

        f_stack_oth = tf.stack([f2_warp, f3_warp, f4_warp, f5_warp, f6_warp], axis=2)
        f_agg = tf.squeeze(VI_Aggregator(f_stack_oth, depth=5, trainable=trainable, namescope='st_agg'), axis=2)
        occlusion_mask = MaskEstimator_(tf.abs(f1 - f_agg), trainable=trainable, namescope='masknet')
        f_syn = (1-occlusion_mask) * f1 + occlusion_mask * f_agg

        bott_input = tf.concat([f1, f_syn], axis=1)
        output = VI_2D_BottleNeck(bott_input, trainable=trainable, namescope='bottleneck')

        # CONV LSTM
        state = ConvLSTM(output, prev_state, trainable=trainable, namescope='convlstm')

        bot_h, bot_w = output.get_shape().as_list()[2:4]
        # ============================ SCALE - 1/4 : 64 =============================
        # align_corners=True
        flow2_64 = resize_bilinear(flow2, (bot_h * 2, bot_w * 2), align_corners=True) * 2
        flow3_64 = resize_bilinear(flow3, (bot_h * 2, bot_w * 2), align_corners=True) * 2
        flow4_64 = resize_bilinear(flow4, (bot_h * 2, bot_w * 2), align_corners=True) * 2
        flow5_64 = resize_bilinear(flow5, (bot_h * 2, bot_w * 2), align_corners=True) * 2
        flow6_64 = resize_bilinear(flow6, (bot_h * 2, bot_w * 2), align_corners=True) * 2

        f2_64_warp = WarpingLayer(f2_64, flow2_64)
        f3_64_warp = WarpingLayer(f3_64, flow3_64)
        f4_64_warp = WarpingLayer(f4_64, flow4_64)
        f5_64_warp = WarpingLayer(f5_64, flow5_64)
        f6_64_warp = WarpingLayer(f6_64, flow6_64)

        flow2_64 = LongFlowNetCorr(f1_64, f2_64_warp, opt, flow2_64, trainable=trainable, reuse=False, namescope='flownet_64') + flow2_64
        flow3_64 = LongFlowNetCorr(f1_64, f3_64_warp, opt, flow3_64, trainable=trainable, reuse=True, namescope='flownet_64') + flow3_64
        flow4_64 = LongFlowNetCorr(f1_64, f4_64_warp, opt, flow4_64, trainable=trainable, reuse=True, namescope='flownet_64') + flow4_64
        flow5_64 = LongFlowNetCorr(f1_64, f5_64_warp, opt, flow5_64, trainable=trainable, reuse=True, namescope='flownet_64') + flow5_64
        flow6_64 = LongFlowNetCorr(f1_64, f6_64_warp, opt, flow6_64, trainable=trainable, reuse=True, namescope='flownet_64') + flow6_64

        f2_64_warp = WarpingLayer(f2_64, flow2_64)
        f3_64_warp = WarpingLayer(f3_64, flow3_64)
        f4_64_warp = WarpingLayer(f4_64, flow4_64)
        f5_64_warp = WarpingLayer(f5_64, flow5_64)
        f6_64_warp = WarpingLayer(f6_64, flow6_64)

        f_stack_64_oth = tf.stack([f2_64_warp, f3_64_warp, f4_64_warp, f5_64_warp, f6_64_warp], axis=2)
        f_agg_64 = tf.squeeze(VI_Aggregator(f_stack_64_oth, depth=5, trainable=trainable, namescope='st_agg_64'), axis=2)
        occlusion_mask_64 = MaskEstimator_(tf.abs(f1_64 - f_agg_64), trainable=trainable, namescope='masknet_64')
        f_syn_64 = (1-occlusion_mask_64) * f1_64 + occlusion_mask_64 * f_agg_64

        # ============================= SCALE - 1/2 : 128 ===============================
        flow2_128 = resize_bilinear(flow2_64, (bot_h * 4, bot_w * 4), align_corners=True) * 2
        flow3_128 = resize_bilinear(flow3_64, (bot_h * 4, bot_w * 4), align_corners=True) * 2
        flow4_128 = resize_bilinear(flow4_64, (bot_h * 4, bot_w * 4), align_corners=True) * 2
        flow5_128 = resize_bilinear(flow5_64, (bot_h * 4, bot_w * 4), align_corners=True) * 2
        flow6_128 = resize_bilinear(flow6_64, (bot_h * 4, bot_w * 4), align_corners=True) * 2

        f2_128_warp = WarpingLayer(f2_128, flow2_128)
        f3_128_warp = WarpingLayer(f3_128, flow3_128)
        f4_128_warp = WarpingLayer(f4_128, flow4_128)
        f5_128_warp = WarpingLayer(f5_128, flow5_128)
        f6_128_warp = WarpingLayer(f6_128, flow6_128)

        flow2_128 = LongFlowNetCorr(f1_128, f2_128_warp, opt, flow2_128, trainable=trainable, reuse=False, namescope='flownet_128') + flow2_128
        flow3_128 = LongFlowNetCorr(f1_128, f3_128_warp, opt, flow3_128, trainable=trainable, reuse=True, namescope='flownet_128') + flow3_128
        flow4_128 = LongFlowNetCorr(f1_128, f4_128_warp, opt, flow4_128, trainable=trainable, reuse=True, namescope='flownet_128') + flow4_128
        flow5_128 = LongFlowNetCorr(f1_128, f5_128_warp, opt, flow5_128, trainable=trainable, reuse=True, namescope='flownet_128') + flow5_128
        flow6_128 = LongFlowNetCorr(f1_128, f6_128_warp, opt, flow6_128, trainable=trainable, reuse=True, namescope='flownet_128') + flow6_128

        f2_128_warp = WarpingLayer(f2_128, flow2_128)
        f3_128_warp = WarpingLayer(f3_128, flow3_128)
        f4_128_warp = WarpingLayer(f4_128, flow4_128)
        f5_128_warp = WarpingLayer(f5_128, flow5_128)
        f6_128_warp = WarpingLayer(f6_128, flow6_128)

        f_stack_128_oth = tf.stack([f2_128_warp, f3_128_warp, f4_128_warp, f5_128_warp, f6_128_warp], axis=2)
        f_agg_128 = tf.squeeze(VI_Aggregator(f_stack_128_oth, depth=5, trainable=trainable, namescope='st_agg_128'), axis=2)
        occlusion_mask_128 = MaskEstimator_(tf.abs(f1_128 - f_agg_128), trainable=trainable, namescope='masknet_128')
        f_syn_128 = (1-occlusion_mask_128) * f1_128 + occlusion_mask_128 * f_agg_128

        output, _ = VI_2D_Decoder_3(tf.expand_dims(state[0], axis=2), opt, x2_64_warp=tf.expand_dims(f_syn_64, axis=2),
                                    x2_128_warp=tf.expand_dims(f_syn_128, axis=2),
                                    trainable=trainable, namescope='decoder')
        occ_mask = resize_bilinear(occlusion_mask, (bot_h * 8, bot_w * 8), align_corners=True)
        occ_mask_64 = resize_bilinear(occlusion_mask_64, (bot_h * 4, bot_w * 4), align_corners=True)
        occ_mask_128 = resize_bilinear(occlusion_mask_128, (bot_h * 2, bot_w * 2), align_corners=True)

        flow6_256, flow6_512 = None, None
        # NOTE: TF is static graph.
        if opt.prev_warp:
            if prev_state is not None or idx != 0:
                h, w = flow6_128.get_shape().as_list()[2:4]
                flow6_256 = resize_bilinear(flow6_128, (h * 2, w * 2)) * 2
                flow6_512 = resize_bilinear(flow6_128, (h * 4, w * 4)) * 4
                f6_256_warp = WarpingLayer(f6_256, flow6_256)
                flow6_256 = LongFlowNetCorr(f1_256, f6_256_warp, opt, flow6_256, trainable=trainable, namescope='flownet_256')
                occlusion_mask_256 = MaskEstimator_(tf.abs(f1_256 - f6_256_warp), trainable=trainable, namescope='masknet_256')
                output_ = output
                if opt.double_size:
                    prev_feed_warp = WarpingLayer(prev_feed[:, :3], flow6_512)
                    h, w = occlusion_mask_256.get_shape().as_list()[2:4]
                    occlusion_mask_512 = resize_nearest(occlusion_mask_256, (h * 2, w * 2))
                    output = tf.squeeze((1-tf.expand_dims(occlusion_mask_512, axis=2)) * output + occlusion_mask_512 * tf.expand_dims(prev_feed_warp, axis=2), axis=2)
                    flow6_256=flow6_512
                else:
                    prev_feed_warp = WarpingLayer(prev_feed[:,:3], flow6_256)
                    output = tf.squeeze((1-tf.expand_dims(occlusion_mask_256, axis=2)) * output + occlusion_mask_256 * tf.expand_dims(prev_feed_warp, axis=2), axis=2)
                if opt.loss_on_raw:
                    output = (output, output_)
        # return output, tf.stack([flow2_128,flow3_128,flow4_128,flow5_128, flow6_128],axis=2), state, tf.stack([occ_mask, 1-occ_mask, occ_mask_64, 1-occ_mask_64, occ_mask_128, 1-occ_mask_128], axis=2), flow6_256
        return output, state[0], state[1]
