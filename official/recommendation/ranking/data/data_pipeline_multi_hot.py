# Copyright 2024 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Data pipeline for the Ranking model.

This module defines various input datasets for the Ranking model.
"""

from typing import List
import tensorflow as tf, tf_keras

from official.recommendation.ranking.configs import config
class CriteoTFRecordReader(object):
  """Input reader fn for TFRecords that have been serialized in batched form."""

  def __init__(self,
               file_pattern: str,
               params: config.DataConfig,
               num_dense_features: int,
               vocab_sizes: List[int],
               multi_hot_sizes: List[int],
               use_cached_data: bool = False,
               embedding_threshold: int = 0):
    self._file_pattern = file_pattern
    self._params = params
    self._num_dense_features = num_dense_features
    self._vocab_sizes = vocab_sizes
    self._multi_hot_sizes = multi_hot_sizes
    self._use_cached_data = use_cached_data
    self._embedding_threshold = embedding_threshold

    self.label_features = 'clicked'
    self.dense_features = ['int-feature-%d' % x for x in range(1, 14)]
    self.sparse_features = ['categorical-feature-%d' % x for x in range(14, 40)]

  def __call__(self, ctx: tf.distribute.InputContext):
    params = self._params
    # Per replica batch size.
    batch_size = (
        ctx.get_per_replica_batch_size(params.global_batch_size)
        if ctx
        else params.global_batch_size
    )

    def _get_feature_spec():
      feature_spec = {}

      feature_spec[self.label_features] = tf.io.FixedLenFeature(
          [
              batch_size,
          ],
          dtype=tf.int64,
      )

      for dense_feat in self.dense_features:
        feature_spec[dense_feat] = tf.io.FixedLenFeature(
            [
                batch_size,
            ],
            dtype=tf.float32,
        )
      for sparse_feat in self.sparse_features:
        feature_spec[sparse_feat] = tf.io.FixedLenFeature(
            [
                batch_size,
            ],
            dtype=tf.string,
        )
      return feature_spec



    def _parse_fn(serialized_example):
      feature_spec = _get_feature_spec()
      parsed_features = tf.io.parse_single_example(
          serialized_example, feature_spec
      )
      label = parsed_features[self.label_features]
      features = {}
      features['clicked'] = tf.reshape(label, [batch_size,])
      int_features = []
      for dense_ft in self.dense_features:
        cur_feature = tf.reshape(parsed_features[dense_ft], [batch_size, 1])
        int_features.append(cur_feature)
      features['dense_features'] = tf.concat(int_features, axis=-1)
      features['sparse_features'] = {}

      for i, sparse_ft in enumerate(self.sparse_features):
        cat_ft_int64 = tf.io.decode_raw(parsed_features[sparse_ft], tf.int64)
        cat_ft_int64 = tf.reshape(
            cat_ft_int64, [batch_size, self._multi_hot_sizes[i]]
        )
        if self._vocab_sizes[i] > self._embedding_threshold:
          features['sparse_features'][str(i)] = tf.sparse.from_dense(
              cat_ft_int64
          )
        else:
          features['sparse_features'][str(i)] = cat_ft_int64

      return features
    if not params.is_training:
      parallelism = 4
    else:
      parallelism = 32
    dataset = tf.data.Dataset.list_files(self._file_pattern, shuffle=False)

    # tf._logging.info(f'num files: {tf.io.gfile.Glob(self._file_pattern)}')
    # Shard the full dataset according to host number.
    # Each host will get 1 / num_of_hosts portion of the data.

    dataset = dataset.shard(ctx.num_input_pipelines, ctx.input_pipeline_id)

    status = 'train' if params.is_training else 'eval'
    tf._logging.info(
        f'Current pipeline: {status}, ctx: {ctx}, num_input_pipelines:'
        f' {ctx.num_input_pipelines}, input_pipeline_id:'
        f' {ctx.input_pipeline_id}'
    )

    if params.is_training:
      dataset = dataset.shuffle(parallelism)
      dataset = dataset.repeat()

    dataset = tf.data.TFRecordDataset(
        dataset,
        buffer_size=256 * 1024 * 1024,
        num_parallel_reads=parallelism,
    )

    dataset = dataset.map(_parse_fn, num_parallel_calls=parallelism)
    if params.is_training:
      dataset = dataset.shuffle(256)
      
    # dataset = dataset.unbatch().batch(1056)

    if not params.is_training:
      num_dataset_samples = 89137319
      num_samples_per_device = num_dataset_samples // ctx.num_input_pipelines
      num_dataset_batches = (num_samples_per_device + batch_size - 1) // batch_size
      def _mark_as_padding(features):
        """Padding will be denoted with a label value of -1."""
        features['clicked'] = -1 * tf.ones(
            [
                batch_size,
            ],
            dtype=tf.int64,
        )
        return features

      # 100 steps worth of padding.
      padding_ds = dataset.take(1)
      padding_ds = padding_ds.map(_mark_as_padding).repeat(200)
      dataset = dataset.concatenate(padding_ds).take(660).cache().repeat()
      # dataset = dataset.cache()

    dataset = dataset.prefetch(buffer_size=256)

    if self._use_cached_data:
      dataset = dataset.take(1).cache().repeat()

    # dataset = dataset.unbatch().batch(1052)


    options = tf.data.Options()
    options.threading.private_threadpool_size = 96
    dataset = dataset.with_options(options)
    return dataset