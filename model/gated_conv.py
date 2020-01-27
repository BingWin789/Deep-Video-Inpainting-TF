import tensorflow as tf

DATA_FORMAT = 'channels_first'

def conv2d(inputs, filters, k_size=3, strides=1, dilation_rate=1, use_bias=False, padding='SAME', trainable=True, namescope='conv2d'):
    # with tf.variable_scope(namescope):
    x = tf.layers.conv2d(inputs, filters, k_size, strides, padding=padding, 
                dilation_rate=dilation_rate, use_bias=use_bias, trainable=trainable, data_format=DATA_FORMAT, 
                kernel_initializer=tf.keras.initializers.he_normal(), name=namescope)
    return x

def to_channels_first(inputs):
    return tf.transpose(inputs, [0, 3, 1, 2])

def to_channels_last(inputs):
    return tf.transpose(inputs, [0, 2, 3, 1])

def resize_bilinear(inputs, size, align_corners=True):
    x = to_channels_last(inputs)
    x = tf.image.resize_bilinear(x, size, align_corners=align_corners)
    x = to_channels_first(x)
    return x

def resize_nearest(inputs, size, align_corners=True):
    x = to_channels_last(inputs)
    x = tf.image.resize_nearest_neighbor(x, size, align_corners=align_corners)
    x = to_channels_first(x)
    return x

def Upsample(inputs, size, is3d=False):
    if is3d:
        # ref: https://stackoverflow.com/questions/43814367/resize-3d-data-in-tensorflow-like-tf-image-resize-images
        x= tf.unstack(inputs, axis=2)
        x = [resize_bilinear(i, size) for i in x]
        x = tf.stack(x, axis=2)
        return x
    return resize_bilinear(inputs, size)

def GatedConvolution(inputs, 
                    filters, 
                    k_size, 
                    strides, 
                    dilation=1, 
                    padding='SAME',
                    use_bias=False, 
                    convtype='3d', 
                    trainable=True, 
                    namescope='gatedconv'):
    with tf.variable_scope(namescope):
        # print(convtype)
        convfcn = tf.layers.conv3d if convtype == '3d' else tf.layers.conv2d
        x = convfcn(inputs, filters * 2, k_size, strides, padding=padding, 
                    dilation_rate=dilation, use_bias=use_bias, trainable=trainable, data_format=DATA_FORMAT, 
                    kernel_initializer=tf.keras.initializers.he_normal(), name='conv')    # 
        phi, gate = tf.split(x, 2, axis=1)
        gate = tf.sigmoid(gate)
        phi = tf.nn.relu(phi)
        result = gate * phi
    if trainable:
        return result
    else:
        return result, gate


def GatedUpConvolution2D(inputs,
                       filters, 
                       newsize,
                       k_size, 
                       strides, 
                       padding, 
                       use_bias=False, 
                       convtype='3d',
                       trainable=True, 
                       namescope='gatedupconv'):
    with tf.variable_scope(namescope):
        convfcn = tf.layers.conv3d if convtype == '3d' else tf.layers.conv2d
        x = Upsample(inputs, newsize, convtype)
        x = convfcn(x, filters * 2, k_size, strides, padding=padding, use_bias=use_bias, data_format=DATA_FORMAT, 
                    trainable=trainable, kernel_initializer=tf.keras.initializers.he_normal(), name='conv/1')
        phi, gate = tf.split(x, 2, axis=1)
        gate = tf.sigmoid(gate)
        phi = tf.nn.leaky_relu(phi, 0.2)
        result = gate * phi
    if trainable:
        return result
    else:
        return result, gate
