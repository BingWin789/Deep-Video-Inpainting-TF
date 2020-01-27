import tensorflow as tf
from .gated_conv import conv2d, to_channels_first, to_channels_last
from lib.correlation import correlation

def conv(inputs, 
        filters, 
        k_size=3, 
        strides=1, 
        dilation_rate=1, 
        batch_norm=False, 
        trainable=True, 
        namescope='conv'):
    use_bias = False if batch_norm else True
    x = conv2d(inputs, filters, k_size=k_size, strides=strides, padding='SAME', 
                dilation_rate=dilation_rate, use_bias=use_bias, trainable=trainable, namescope=namescope)
    if batch_norm:
        x = tf.contrib.layers.batch_norm(x, is_training=trainable, 
                trainable=trainable, data_format='NCHW')
    x = tf.nn.leaky_relu(x, 0.1)
    return x

def MaskEstimator_(inputs, trainable=True, reuse=False, namescope='MaskEstimator_'):
    with tf.variable_scope(namescope, reuse=reuse):
        ch = inputs.get_shape().as_list()[1]
        x = conv(inputs, ch // 2, batch_norm=False, trainable=trainable, namescope='convs/0/0')
        x = conv(x, ch // 2, batch_norm=False, trainable=trainable, namescope='convs/1/0')
        x = conv2d(x, 1, use_bias=True, trainable=trainable, namescope='convs/2')
        x = tf.sigmoid(x)
    return x

def get_grid(inputs):
    # NOTE: inputs channels last required
    H, W = inputs.get_shape().as_list()[1:3]
    x = tf.linspace(-1.0, 1.0, W)
    y = tf.linspace(-1.0, 1.0, H)
    x_t, y_t = tf.meshgrid(x, y)
    x_t = tf.expand_dims(x_t, axis=-1)
    x_t = tf.expand_dims(x_t, axis=0)
    y_t = tf.expand_dims(y_t, axis=-1)
    y_t = tf.expand_dims(y_t, axis=0)
    N = inputs.get_shape().as_list()[0]
    grid_0 = tf.tile(x_t, [N, 1, 1, 1])
    grid_1 = tf.tile(y_t, [N, 1, 1, 1])
    return grid_0, grid_1

def WarpingLayer(inputs, flow): 
    inp = to_channels_last(inputs)
    flo = to_channels_last(flow)
    H, W = flo.get_shape().as_list()[1:3]
    flo_0 = flo[:, :, :, 0:1] / ((W - 1.0) / 2.0)
    flo_1 = flo[:, :, :, 1:2] / ((H - 1.0) / 2.0)
    grid_0, grid_1 = get_grid(inp)
    x_t = grid_0 + flo_0
    y_t = grid_1 + flo_1
    x_t = x_t[:, :, :, 0]
    y_t = y_t[:, :, :, 0]
    res = bilinear_sampler(inp, x_t, y_t)
    res = to_channels_first(res)
    return res

def ContextNetwork(inputs, args, trainable=True, namescope='ContextNetwork'):
    with tf.variable_scope(namescope):
        x = conv(inputs, 128, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/0/0')
        x = conv(x, 128, batch_norm=args.batch_norm, dilation_rate=2, trainable=trainable, namescope='convs/1/0')
        x = conv(x, 128, batch_norm=args.batch_norm, dilation_rate=4, trainable=trainable, namescope='convs/2/0')
        x = conv(x, 96, batch_norm=args.batch_norm, dilation_rate=8, trainable=trainable, namescope='convs/3/0')
        x = conv(x, 64, batch_norm=args.batch_norm, dilation_rate=16, trainable=trainable, namescope='convs/4/0')
        x = conv(x, 32, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/5/0')
        x = conv(x, 2, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/6/0')
    return x

def LongFlowEstimatorCorr(inputs, args, trainable=True, namescope='LongFlowEstimatorCorr'):
    with tf.variable_scope(namescope):
        x = conv(inputs, 128, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/0/0')
        x = conv(x, 128, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/1/0')
        x = conv(x, 96, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/2/0')
        x = conv(x, 64, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/3/0')
        x = conv(x, 32, batch_norm=args.batch_norm, trainable=trainable, namescope='convs/4/0')
        flo_coarse = conv2d(x, 2, use_bias=True, trainable=trainable, namescope='conv1')
        flo_tmp = tf.concat([x, flo_coarse], axis=1)
        flo_fine = ContextNetwork(flo_tmp, args, trainable=trainable, namescope='convs_fine')
        flo = flo_coarse + flo_fine
    return flo

def LongFlowNetCorr(inputs1, inputs2, args, upflow=None, trainable=True, reuse=False, namescope='LongFlowNetCorr'):
    with tf.variable_scope(namescope, reuse=reuse):
        inp1 = to_channels_last(inputs1)
        inp2 = to_channels_last(inputs2)
        corr = correlation(inp1, inp2, kernel_size=1, max_displacement=args.search_range, stride_1=1, stride_2=1, padding=args.search_range)
        corr = to_channels_first(corr)

        inp = tf.concat([inputs1, corr], axis=1)
        if upflow is not None:
            inp = tf.concat([inp, upflow], axis=1)
        flow = LongFlowEstimatorCorr(inp, args, trainable=trainable, namescope='flow_estimator')
    return flow

# bilinear sample equals to pytorch F.grid_sample
# reference: https://github.com/kevinzakka/spatial-transformer-network/blob/master/stn/transformer.py#L159
def get_pixel_value(img, x, y):
    """
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)

def bilinear_sampler(img, x, y):
    """
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32')
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0)
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out
