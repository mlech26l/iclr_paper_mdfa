import tensorflow as tf
import numpy as np

class AbsRandomNormal(tf.keras.initializers.Initializer):
  """Initializer that generates tensors with a normal distribution.
  Args:
    mean: a python scalar or a scalar tensor. Mean of the random values to
      generate.
    stddev: a python scalar or a scalar tensor. Standard deviation of the random
      values to generate.
    seed: A Python integer. Used to create random seeds. See
      `tf.compat.v1.set_random_seed` for behavior.
    dtype: Default data type, used if no `dtype` argument is provided when
      calling the initializer. Only floating point types are supported.
  """

  def __init__(self, mean=0.0, stddev=1.0, seed=None, dtype=tf.float32):
    self.mean = mean
    self.stddev = stddev
    self.seed = seed
    self.dtype = dtype

  def __call__(self, shape, dtype=None, partition_info=None):
    if dtype is None:
      dtype = self.dtype
    return tf.abs(tf.random_normal(
        shape, self.mean, self.stddev, dtype, seed=self.seed))

  def get_config(self):
    return {
        "mean": self.mean,
        "stddev": self.stddev,
        "seed": self.seed,
        "dtype": self.dtype.name
    }


def parse_initializer(text,in_size):
    fanin_init = 1.0/np.sqrt(in_size)

    text = text.upper()
    if("U" in text):
        text = text.replace("U","")
        v = float(text) * fanin_init
        return tf.random_uniform_initializer(-v,v)
    elif("N" in text):
        text = text.replace("N","")
        v = float(text)
        return tf.random_normal_initializer(stddev=v)
    elif("A" in text):
        text = text.replace("A","")
        v = float(text)
        return AbsRandomNormal(stddev=v)
    elif("P" in text):
        text = text.replace("P","")
        v = float(text) * fanin_init
        return tf.random_uniform_initializer(0,v)

    v = float(text)
    return tf.constant_initializer(v)
