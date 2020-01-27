import tensorflow as tf
from .gated_conv import conv2d


def ConvLSTM(inputs, prev_state, hidden_size=128, k_size=3, trainable=True, namescope='convlstm'):
    # prev_state:([N, H, W, hidden_size], [N, H, W, hidden_size])
    # prev_state = (tf.zeros([N, H, W, hidden_size], tf.float32), tf.zeros([N, H, W, hidden_size], tf.float32)) for
    # first frame.
    # NOTE: THERE is no 'type' params.
    with tf.variable_scope(namescope):
        prev_hidden, prev_cell = prev_state
        stacked_inputs = tf.concat([inputs, prev_hidden], axis=1)

        gates = conv2d(stacked_inputs, hidden_size * 4, k_size=k_size, trainable=trainable, use_bias=True, namescope='Gates')
        # split across channel dimension
        in_gate, remember_gate, out_gate, cell_gate = tf.split(gates, 4, axis=1)
        # apply sigmoid non linearity
        in_gate = tf.sigmoid(in_gate)
        remember_gate = tf.sigmoid(remember_gate)
        out_gate = tf.sigmoid(out_gate)

        # apply tanh non linearity
        cell_gate = tf.tanh(cell_gate)

        # compute current cell and hidden state
        cell = (remember_gate * prev_cell) + (in_gate * cell_gate)
        hidden = out_gate * tf.tanh(cell)

    return hidden, cell