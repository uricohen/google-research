# coding=utf-8
# Copyright 2019 The Google Research Authors.
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

"""Utilities for extracting layers activation of a model specified by a model config.

  Typical usage example:

  root_dir = # directory where the models are
  data_dir = # directory where the dataset is
  model_config = ModelConfig(...)
  input_fn = data_util.get_input(data_dir,
      data=model_config.dataset, data_format=model_config.data_format)
  activations = extract_layers(input_fn, root_dir, model_config)
  # a map with keys which correspond to layers, i.e. inputs, h1, h2, h3
"""

from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf


IMG_HEIGHT = 32
IMG_WIDTH = 32
IMG_CHANNELS = 3


def extract_layers(
    input_fn, root_dir, model_config, sess=None,
    batchsize=50, dataset_size=50000):
  """Extract activation at different layers of a model on all input data.

  Loads a given model from given directory and load the parameters in the given
  scope. Iterates over the entire training dataset and extracts activations and 
  labels.

  Args:
    input_fn:  function that produces the input and label tensors
    root_dir:  the directory containing the dataset
    model_config: a ModelConfig object that specifies the model
    sess: optional tensorflow session
    batchsize: batch size with which the margin is computed
    dataset_size: number of data points in the dataset

  Returns:
    A dictionary that maps each layer's name to neuron activations at that layer
    over the entire training set.
    A list of labels
  """
  #param_path = model_config.get_model_dir_name(root_dir)
  param_path = model_config.get_checkpoint_path(root_dir)
  model_fn = model_config.get_model_fn()

  if not sess:
    sess = tf.Session()

  data_format = model_config.data_format
  image_iter, label_iter = input_fn()
  if data_format == 'HWC':
    img_dim = [None, IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS]
  else:
    img_dim = [None, IMG_CHANNELS, IMG_HEIGHT, IMG_WIDTH]
  image = tf.placeholder(tf.float32, shape=img_dim, name='image')
  label = tf.placeholder(
      tf.float32, shape=[None, model_config.num_class], name='label')

  layer_names = ['inputs', 'h1', 'h2', 'h3']
  N_LAYERS = len(layer_names)
  N_OBJECTS = model_config.num_class
  MAX_NEURONS = 3072
  N_SAMPLES = int(np.ceil(dataset_size/N_OBJECTS)*1.005)
  print('Collecting %d neurons from %d layers (%d samples, %d objects)' % (MAX_NEURONS, N_LAYERS, N_SAMPLES, N_OBJECTS))

  end_points_collection = {}
  logits = model_fn(image, is_training=False, 
                    end_points_collection=end_points_collection)

  # set up the graph for computing activations
  layer_activations = [end_points_collection[l] for l in layer_names]
  layer_indices = [[] for l in layer_names]
  layer_n_neurons = np.zeros(N_LAYERS,dtype='int32')

  # load model parameters
  sess.run(tf.global_variables_initializer())
  model_config.load_parameters(param_path, sess)
  all_activations = np.empty([N_LAYERS, MAX_NEURONS, N_SAMPLES, N_OBJECTS], dtype='float32')
  all_activations[:] = np.NaN
  samples_per_object = np.zeros([N_LAYERS, N_OBJECTS], dtype='int32')

  count = 0
  activation_values = []
  while count < dataset_size:
    try:
      count += batchsize
      image_batch, label_batch = sess.run([image_iter, label_iter])
      label_batch = np.reshape(label_batch, [-1, model_config.num_class])
      fd = {image: image_batch, label: label_batch.astype(np.float32)}
      activations = sess.run(layer_activations, feed_dict=fd)
      assert(len(activations) == N_LAYERS)
      for il in range(N_LAYERS):
        # Initialize layer's indices if haven't done so yet
        if layer_n_neurons[il] == 0:
          assert(layer_indices[il] == [])
          assert(activations[il].shape[0] == batchsize)
          layer_n_neurons[il] = activations[il].size/batchsize
          if layer_n_neurons[il]>MAX_NEURONS:
            layer_indices[il] = np.random.choice(layer_n_neurons[il], MAX_NEURONS, replace=False) 
          else:
            layer_indices[il][:] = range(layer_n_neurons[il])

        # Collect layer's data
        data = np.reshape(activations[il],[batchsize,-1])
        data = data[:,layer_indices[il]]
        J = data.shape[1]
        for ik in range(batchsize):
          io = np.where(label_batch[ik,:]==1)[0]
          all_activations[il, 0:J, samples_per_object[il,io], io] = data[ik,:]
          samples_per_object[il,io] += 1
    except tf.errors.OutOfRangeError:
      print('reached the end of the data (%d)'%count)
      break
  assert(np.all(np.var(samples_per_object,axis=0) == 0))
  samples_per_object = np.mean(samples_per_object,axis=0)

  return all_activations, samples_per_object, layer_names, layer_indices, layer_n_neurons

