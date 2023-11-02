"""Low Rank DCN Module."""

from typing import Optional, Text, Union

import tensorflow as tf


# @tf.keras.utils.register_keras_serializable()
class LowRankDCN(tf.keras.layers.Layer):
  """Cross Layer in Deep & Cross Network to learn explicit feature interactions.

  A layer that creates explicit and bounded-degree feature interactions
  efficiently. The `call` method accepts `inputs` as a tuple of size 2 tensors.
  The first input `x0` is the base layer that contains the original features
  (usually the embedding layer); the second input `xi` is the output of the
  previous `Cross` layer in the stack, i.e., the i-th `Cross` layer. For the
  first `Cross` layer in the stack, x0 = xi. The output is x_{i+1} = x0 .* (W *
  xi + bias + diag_scale * xi) + xi, where .* designates elementwise
  multiplication, W could be a full-rank matrix, or a low-rank matrix U*V to
  reduce the computational cost, and diag_scale increases the diagonal of W to
  improve training stability ( especially for the low-rank case). References:

      1. [R. Wang et al.](https://arxiv.org/pdf/2008.13535.pdf)
        See Eq. (1) for full-rank and Eq. (2) for low-rank version.
      2. [R. Wang et al.](https://arxiv.org/pdf/1708.05123.pdf)
  Example:
      ```python
      # after embedding layer in a functional model:
      input = tf.keras.Input(shape=(None,), name='index', dtype=tf.int64)
      x0 = tf.keras.layers.Embedding(input_dim=32, output_dim=6)
      x1 = Cross()(x0, x0)
      x2 = Cross()(x0, x1)
      logits = tf.keras.layers.Dense(units=10)(x2)
      model = tf.keras.Model(input, logits)
      ```
  Args: num_layers = number of layers of Cross Netwrok with Low Rank
      projection_dim: project dimension to reduce the computational cost.
      Default is `None` such that a full (`input_dim` by `input_dim`) matrix W
      is used. If enabled, a low-rank matrix W = U*V will be used, where U is of
      size `input_dim` by `projection_dim` and V is of size `projection_dim` by
      `input_dim`. `projection_dim` need to be smaller than `input_dim`/2 to
      improve the model efficiency. In practice, we've observed that
      `projection_dim` = d/4 consistently preserved the accuracy of a full-rank
      version.
      diag_scale: a non-negative float used to increase the diagonal of the
      kernel W by `diag_scale`, that is, W + diag_scale * I, where I is an
      identity matrix.
      use_bias: whether to add a bias term for this layer. If set to False, no
      bias term will be used.
      preactivation: Activation applied to output matrix of the layer, before
      multiplication with the input. Can be used to control the scale of the
      layer's outputs and improve stability.
      kernel_initializer: Initializer to use on the kernel matrix.
      bias_initializer: Initializer to use on the bias vector.
      kernel_regularizer: Regularizer to use on the kernel matrix.
      bias_regularizer: Regularizer to use on bias vector. Input shape: A tuple
      of 2 (batch_size, `input_dim`) dimensional inputs. Output shape: A single
      (batch_size, `input_dim`) dimensional output.
  """

  def __init__(
      self,
      num_layers: int,
      projection_dim: Optional[int] = 1,
      diag_scale: Optional[float] = 0.0,
      use_bias: bool = True,
      preactivation: Optional[Union[str, tf.keras.layers.Activation]] = None,
      kernel_initializer: Union[
          Text, tf.keras.initializers.Initializer
      ] = "glorot_normal",
      bias_initializer: Union[
          Text, tf.keras.initializers.Initializer
      ] = "zeros",
      kernel_regularizer: Union[
          Text, None, tf.keras.regularizers.Regularizer
      ] = None,
      bias_regularizer: Union[
          Text, None, tf.keras.regularizers.Regularizer
      ] = None,
      **kwargs
  ):
    super(LowRankDCN, self).__init__(**kwargs)
    self.built = False
    self._num_layers = num_layers
    self._projection_dim = projection_dim
    self._diag_scale = diag_scale
    self._use_bias = use_bias
    self._preactivation = tf.keras.activations.get(preactivation)
    self._kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self._bias_initializer = tf.keras.initializers.get(bias_initializer)
    self._kernel_regularizer = tf.keras.regularizers.get(kernel_regularizer)
    self._bias_regularizer = tf.keras.regularizers.get(bias_regularizer)
    self._input_dim = None

    self._supports_masking = True

    if self._diag_scale < 0:  # pytype: disable=unsupported-operands
      raise ValueError(
          "`diag_scale` should be non-negative. Got `diag_scale` = {}".format(
              self._diag_scale
          )
      )

  def _construct_dense_u_kernels(self):
    self._dense_u_kernels = []
    for _ in range(self._num_layers):
      self._dense_u_kernels.append(
          tf.keras.layers.Dense(
              self._projection_dim,
              kernel_initializer=_clone_initializer(self._kernel_initializer),
              kernel_regularizer=self._kernel_regularizer,
              use_bias=False,
              dtype=self.dtype,
          )
      )

  def _construct_dense_v_kernels(self, last_dim: int):
    self._dense_v_kernels = []
    for _ in range(self._num_layers):
      self._dense_v_kernels.append(
          tf.keras.layers.Dense(
              last_dim,
              kernel_initializer=_clone_initializer(self._kernel_initializer),
              bias_initializer=self._bias_initializer,
              kernel_regularizer=self._kernel_regularizer,
              bias_regularizer=self._bias_regularizer,
              use_bias=self._use_bias,
              dtype=self.dtype,
              activation=self._preactivation,
          )
      )

  def build(self, input_shape):
    """Function to build all the layers of low rank dcn.
    
    Args:
      input_shape: shape of input tensor
    """
    # Gagik - this does not seem so right, need you help here.
    # As per the layer - 
    # https://github.com/pytorch/torchrec/blob/main/torchrec/modules/crossnet.py#L92
    last_dim = input_shape[-1][-1]
    self._construct_dense_u_kernels()
    self._construct_dense_v_kernels(last_dim)
    self.built = True

  def call(self, x0_l: tf.Tensor, x: Optional[tf.Tensor] = None) -> tf.Tensor:
    """Computes the feature cross.

    Args:
      x0_l: The input tensor
      x: Optional second input tensor. If provided, the layer will compute
        crosses between x0 and x; if not provided, the layer will compute
        crosses between x0 and itself.

    Returns:
     Tensor of crosses.
    """
    # print(x0.shape, self.built)
    # pass
    # print("x0 and x1", x0, x)
    x0 = x0_l[0]
    x = x0_l[1]

    if not self.built:
      print("calling build ", x0.shape)
      self.build(x0.shape)

    if x is None:
      x = x0

    if x0.shape[-1] != x.shape[-1]:
      raise ValueError(
          "`x0` and `x` dimension mismatch! Got `x0` dimension {}, and x "
          "dimension {}. This case is not supported yet.".format(
              x0.shape[-1], x.shape[-1]
          )
      )
    print(self._num_layers)
    print("dense u kernels", len(self._dense_u_kernels))
    print("dense v kernels", len(self._dense_v_kernels))
    for i in range(self._num_layers):
      prod_output = self._dense_v_kernels[i](self._dense_u_kernels[i](x))
      # prod_output = tf.cast(prod_output, self.compute_dtype)
      # if self._diag_scale:
      #   prod_output = prod_output + self._diag_scale * x
      x = x0 * prod_output + x
      print("layer and prod output ", i, prod_output)
      print("layer and x ", i, x)
    return x

  def get_config(self):
    config = {
        "projection_dim": self._projection_dim,
        "diag_scale": self._diag_scale,
        "use_bias": self._use_bias,
        "preactivation": tf.keras.activations.serialize(self._preactivation),
        "kernel_initializer": tf.keras.initializers.serialize(
            self._kernel_initializer
        ),
        "bias_initializer": tf.keras.initializers.serialize(
            self._bias_initializer
        ),
        "kernel_regularizer": tf.keras.regularizers.serialize(
            self._kernel_regularizer
        ),
        "bias_regularizer": tf.keras.regularizers.serialize(
            self._bias_regularizer
        ),
    }
    base_config = super().get_config()
    return dict(list(base_config.items()) + list(config.items()))


def _clone_initializer(initializer):
  return initializer.__class__.from_config(initializer.get_config())
