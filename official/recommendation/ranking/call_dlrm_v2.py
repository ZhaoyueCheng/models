"""Example calling DLRMv2 implementation."""

import math
from typing import List
import tensorflow as tf
import tensorflow_recommenders as tfrs
from low_rank_dcn import LowRankDCN
from recommendation_v2 import Recommendation

print(tf.__version__)

cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=f'local')
# print("Running on TPU ", cluster_resolver.cluster_spec().as_dict()["worker"])

tf.config.experimental_connect_to_cluster(cluster_resolver)
tf.tpu.experimental.initialize_tpu_system(cluster_resolver)
strategy = tf.distribute.experimental.TPUStrategy(cluster_resolver)


# GLOBAL_BATCH_SIZE = 65536
# NUM_DENSE = 13
# VOCAB_SIZES = [
#     40000000,
#     39060,
#     17295,
#     7424,
#     20265,
#     3,
#     7122,
#     1543,
#     63,
#     40000000,
#     3067956,
#     405282,
#     10,
#     2209,
#     11938,
#     155,
#     4,
#     976,
#     14,
#     40000000,
#     40000000,
#     40000000,
#     590152,
#     12973,
#     108,
#     36,
# ]
# MULTI_HOT_SIZES = [
#     3,
#     2,
#     1,
#     2,
#     6,
#     1,
#     1,
#     1,
#     1,
#     7,
#     3,
#     8,
#     1,
#     6,
#     9,
#     5,
#     1,
#     1,
#     1,
#     12,
#     100,
#     27,
#     10,
#     3,
#     1,
#     1,
# ]

# EMBEDDING_DIM = 128
# DATASET_SIZE_TRAIN = 262144
# DATASET_SIZE_EVAL = 65536
# BATCH_SIZE = 65536
# BOTTOM_MLP_LIST = [512, 256, EMBEDDING_DIM]
# TOP_MLP_LIST = [1024, 1024, 512, 256, 1]
# PROJECTION_DIM = 512
# NUM_LDCN_LAYERS = 3
# NUM_WORKERS = 8
# NUM_DEVICES_PER_WORKER = 4
# NUM_SAMPLES_PER_CORE = int(BATCH_SIZE / (NUM_WORKERS * NUM_DEVICES_PER_WORKER))

GLOBAL_BATCH_SIZE = 1024
NUM_DENSE =  3
VOCAB_SIZES = [6, 7, 6, 7]
MULTI_HOT_SIZES = [2, 2, 3, 1]
EMBEDDING_DIM = 2
DATASET_SIZE_TRAIN = 256
DATASET_SIZE_EVAL = 256
# if batch size is 128, per host is 128/8 and then per core is 128/8/4
BATCH_SIZE = 128
BOTTOM_MLP_LIST = [256, 64, EMBEDDING_DIM]
TOP_MLP_LIST = [512, 256, 1]
PROJECTION_DIM = 2
NUM_LDCN_LAYERS = 3
NUM_WORKERS = 4
NUM_DEVICES_PER_WORKER = 1
NUM_SAMPLES_PER_CORE = int(BATCH_SIZE/(NUM_WORKERS*NUM_DEVICES_PER_WORKER))


def _generate_synthetic_data_train(
    context: tf.distribute.InputContext,
) -> tf.data.Dataset:
  """Function to generate synthetic data.

  Args:
    context:

  Returns:

  """
  dense_tensor = tf.random.uniform(
      shape=(DATASET_SIZE_TRAIN, NUM_DENSE), maxval=1.0, dtype=tf.float32
  )

  sparse_tensors = []
  for i, size in enumerate(VOCAB_SIZES):
    temp_dense_tensor = tf.random.uniform(
        shape=(DATASET_SIZE_TRAIN, MULTI_HOT_SIZES[i]),
        maxval=int(size),
        dtype=tf.int32,
    )
    temp_sparse_tensor = tf.sparse.from_dense(temp_dense_tensor)
    sparse_tensors.append(temp_sparse_tensor)

  sparse_tensor_elements = {
      str(i): sparse_tensors[i] for i in range(len(sparse_tensors))
  }

  # The mean is in [0, 1] interval.
  dense_tensor_mean = tf.math.reduce_mean(dense_tensor, axis=1)
  label_tensor = (dense_tensor_mean) / 2.0
  label_tensor = tf.cast(label_tensor + 0.5, tf.int32)

  input_elem = {
      "dense_features": dense_tensor,
      "sparse_features": sparse_tensor_elements,
  }, label_tensor
  
  dataset = tf.data.Dataset.from_tensor_slices(input_elem)
  dataset = dataset.repeat()
  dataset = dataset.batch(
      context.get_per_replica_batch_size(BATCH_SIZE), drop_remainder=True
  )
  return dataset.prefetch(tf.data.experimental.AUTOTUNE)

#   return dataset.batch(BATCH_SIZE, drop_remainder=True)


def _generate_synthetic_data_eval(
    context: tf.distribute.InputContext,
) -> tf.data.Dataset:
  """Function to generate synthetic data.

  Args:
    context:

  Returns:

  """
  dense_tensor = tf.random.uniform(
      shape=(DATASET_SIZE_EVAL, NUM_DENSE), maxval=1.0, dtype=tf.float32
  )

  sparse_tensors = []
  for i, size in enumerate(VOCAB_SIZES):
    temp_dense_tensor = tf.random.uniform(
        shape=(DATASET_SIZE_EVAL, MULTI_HOT_SIZES[i]),
        maxval=int(size),
        dtype=tf.int32,
    )
    temp_sparse_tensor = tf.sparse.from_dense(temp_dense_tensor)
    sparse_tensors.append(temp_sparse_tensor)

  sparse_tensor_elements = {
      str(i): sparse_tensors[i] for i in range(len(sparse_tensors))
  }

  # The mean is in [0, 1] interval.
  dense_tensor_mean = tf.math.reduce_mean(dense_tensor, axis=1)
  label_tensor = (dense_tensor_mean)/2.0
  label_tensor = tf.cast(label_tensor + 0.5, tf.int32)

  input_elem = {
      "dense_features": dense_tensor,
      "sparse_features": sparse_tensor_elements
  }, label_tensor

  dataset = tf.data.Dataset.from_tensor_slices(input_elem)
  dataset = dataset.batch(
      context.get_per_replica_batch_size(BATCH_SIZE),
      drop_remainder=True)
  return dataset.prefetch(tf.data.experimental.AUTOTUNE)


def _get_tpu_embedding_layer(
    vocab_sizes: List[int],
    embedding_dim: int,
    embedding_optimizer: tf.keras.optimizers.Optimizer,
    table_name_prefix: str = "tpu_embedding"
) -> tfrs.layers.embedding.TPUEmbedding:
  """Returns TPU embedding layer with given `vocab_sizes` and `embedding_dim`.

  Args:
    vocab_sizes: List of sizes of categories/id's in the table.
    embedding_dim: Embedding dimension.
    embedding_optimizer: optimizer used for embeddings
    table_name_prefix: prefix added to table names
  Returns:
    A TPU embedding layer.
  """
  feature_config = {}

  for i, vocab_size in enumerate(vocab_sizes):
    table_config = tf.tpu.experimental.embedding.TableConfig(
        vocabulary_size=vocab_size,
        dim=embedding_dim,
        initializer=tf.initializers.TruncatedNormal(
            mean=0.0, stddev=1 / math.sqrt(embedding_dim)
        ),
        name=table_name_prefix + "_%s" % i,
    )
    feature_config[str(i)] = tf.tpu.experimental.embedding.FeatureConfig(
        table=table_config, output_shape=[NUM_SAMPLES_PER_CORE, embedding_dim]
    )

  tpu_embedding = tfrs.layers.embedding.TPUEmbedding(
      feature_config, embedding_optimizer
  )

  return tpu_embedding


def build_model(model_strategy):
  """Function to build model.

  Args:
    model_strategy:

  Returns:

  """
  with model_strategy.scope():
    # optimizer = tf.tpu.experimental.embedding.Adam()
    optimizer = tf.keras.optimizers.legacy.Adam()

    tpu_embedding = _get_tpu_embedding_layer(
        vocab_sizes=VOCAB_SIZES,
        embedding_dim=EMBEDDING_DIM,
        embedding_optimizer=tf.tpu.experimental.embedding.Adagrad(),
    )

    # To freeze the embedding weights set `tpu_embedding.trainable = False`
    recommendation_model = Recommendation(
        embedding_layer=tpu_embedding,
        bottom_stack=tfrs.layers.blocks.MLP(
            units=BOTTOM_MLP_LIST, final_activation="relu"
        ),
        # feature_interaction=tfrs.layers.feature_interaction.DotInteraction(),
        feature_interaction=LowRankDCN(
            num_layers=NUM_LDCN_LAYERS, projection_dim=PROJECTION_DIM
        ),
        top_stack=tfrs.layers.blocks.MLP(
            units=TOP_MLP_LIST, final_activation="sigmoid"
        ),
    )

    recommendation_model.compile(optimizer, steps_per_execution=100)

    return recommendation_model


with strategy.scope():
  input_options = tf.distribute.InputOptions(experimental_fetch_to_device=False)
  eval_dataset = tf.keras.utils.experimental.DatasetCreator(
      _generate_synthetic_data_eval, input_options=input_options
  )
  train_dataset = tf.keras.utils.experimental.DatasetCreator(
      _generate_synthetic_data_train, input_options=input_options
  )
  model = build_model(strategy)
  model.fit(
      train_dataset,
      steps_per_epoch=DATASET_SIZE_TRAIN // BATCH_SIZE,
      validation_data=eval_dataset,
      validation_steps=1,
  )
