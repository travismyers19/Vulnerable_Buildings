from tensorflow.python.ops import math_ops
import tensorflow.keras.backend as K
from tensorflow.keras.backend import epsilon
from tensorflow.python.framework import constant_op
from tensorflow.python.ops import clip_ops

def _constant_to_tensor(x, dtype):
  """Convert the input `x` to a tensor of type `dtype`.
  This is slightly faster than the _to_tensor function, at the cost of
  handling fewer cases.
  Arguments:
      x: An object to be converted (numpy arrays, floats, ints and lists of
        them).
      dtype: The destination type.
  Returns:
      A tensor.
  """
  return constant_op.constant(x, dtype=dtype)

def focal_binary_crossentropy(target, output, from_logits=False):
    """Binary crossentropy between an output tensor and a target tensor.
    Arguments:
    target: A tensor with the same shape as `output`.
    output: A tensor.
    from_logits: Whether `output` is expected to be a logits tensor.
        By default, we consider that `output`
        encodes a probability distribution.
    Returns:
    A tensor.
    """
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)

    # Compute cross entropy from probabilities.
    bce = target * math_ops.square(1 - output) * math_ops.log(output + epsilon())
    bce += (1 - target) * math_ops.square(output) * math_ops.log(1 - output + epsilon())
    return K.mean(-bce)

def focal_categorical_crossentropy(target, output, from_logits=False, axis=-1):
    """Categorical crossentropy between an output tensor and a target tensor.
    Arguments:
        target: A tensor of the same shape as `output`.
        output: A tensor resulting from a softmax
            (unless `from_logits` is True, in which
            case `output` is expected to be the logits).
        from_logits: Boolean, whether `output` is the
            result of a softmax, or is a tensor of logits.
        axis: Int specifying the channels axis. `axis=-1` corresponds to data
            format `channels_last', and `axis=1` corresponds to data format
            `channels_first`.
    Returns:
        Output tensor.
    Raises:
        ValueError: if `axis` is neither -1 nor one of the axes of `output`.
    Example:
    >>> a = tf.constant([1., 0., 0., 0., 1., 0., 0., 0., 1.], shape=[3,3])
    >>> print(a)
    tf.Tensor(
    [[1. 0. 0.]
        [0. 1. 0.]
        [0. 0. 1.]], shape=(3, 3), dtype=float32)
    >>> b = tf.constant([.9, .05, .05, .5, .89, .6, .05, .01, .94], shape=[3,3])
    >>> print(b)
    tf.Tensor(
    [[0.9  0.05 0.05]
        [0.5  0.89 0.6 ]
        [0.05 0.01 0.94]], shape=(3, 3), dtype=float32)
    >>> loss = tf.keras.backend.categorical_crossentropy(a, b)
    >>> print(loss)
    tf.Tensor([0.10536055 0.8046684  0.06187541], shape=(3,), dtype=float32)
    >>> loss = tf.keras.backend.categorical_crossentropy(a, a)
    >>> print(loss)
    tf.Tensor([1.1920929e-07 1.1920929e-07 1.19...e-07], shape=(3,),
    dtype=float32)
    """
    # scale preds so that the class probas of each sample sum to 1
    output = output / math_ops.reduce_sum(output, axis, True)
    # Compute cross entropy from probabilities.
    epsilon_ = _constant_to_tensor(epsilon(), output.dtype.base_dtype)
    output = clip_ops.clip_by_value(output, epsilon_, 1. - epsilon_)
    return K.mean(-math_ops.reduce_sum(target * math_ops.square(1 - output) * math_ops.log(output), axis))