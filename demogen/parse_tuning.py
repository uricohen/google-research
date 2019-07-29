import tensorflow as tf
import demogen.data_util as data_util
import demogen.model_config as mc
import demogen.extract_layers_util as elu

import json
import os
import scipy.io
import numpy as np

root_dir = '../concept/data/demogen_models/'
data_dir = '../t2t_data/'
spec = os.path.join(root_dir, 'all_model_spec.json')

model_types = ['nin', 'resnet', 'resnet']
datasets = ['cifar10', 'cifar10', 'cifar100']
assert(len(datasets) == len(model_types))

# Read spec file
with tf._api.v1.io.gfile.GFile(spec, 'r') as f:
  d = json.load(f)

for i in range(len(model_types)):
  model_type = model_types[i]
  dataset = datasets[i]
  key = '%s_%s' % (model_type.upper(), dataset.upper())
  # Add defaults from model_config.py
  if not d[key].has_key('normalization'):
    d[key]['normalization'] = ['batch']
  if not d[key].has_key('batchnorm'):
    d[key]['batchnorm'] = [False]
  if not d[key].has_key('dropout_prob'):
    d[key]['dropout_prob'] = [0.0]
  if not d[key].has_key('learning_rate'):
    d[key]['learning_rate'] = [0.01]

  print(d[key])
  count = 0
  failures = []

  for wide_multiplier in d[key]['wide_multiplier']:
    for batchnorm in d[key]['batchnorm']:
      for dropout_prob in d[key]['dropout_prob']:
        for augmentation in d[key]['augmentation']:
          for decay_fac in d[key]['decay_fac']:
            for copy in d[key]['copy']:
              for normalization in d[key]['normalization']:
                for learning_rate in d[key]['learning_rate']:
                  model_config = mc.ModelConfig(model_type=model_type, dataset=dataset,
                                            wide_multiplier=wide_multiplier,
                                            batchnorm=batchnorm,
                                            dropout_prob=dropout_prob,
                                            data_augmentation=augmentation,
                                            l2_decay_factor=decay_fac,
                                            normalization=normalization,
                                            learning_rate=learning_rate,
                                            copy=copy,
                                            root_dir=root_dir)
                  count+=1
                  filename = os.path.basename(model_config.get_model_dir_name())
                  name = '%s_%s_%s_tuning.mat'%(model_type,dataset,filename)
                  if os.path.isfile(name):
                    continue
                  try:
                    # Training metadata
                    results = os.path.join(model_config.get_model_dir_name(),'train.json')
                    with tf._api.v1.io.gfile.GFile(results, 'r') as f:
                      dd = json.load(f)
                    train_loss = dd['loss']
                    train_cross_entropy = dd['CrossEntropy']
                    train_global_step = dd['global_step']
                    train_accuracy = dd['Accuracy']
                    results = os.path.join(model_config.get_model_dir_name(),'eval.json')
                    # Validation metadata
                    with tf._api.v1.io.gfile.GFile(results, 'r') as f:
                      dd = json.load(f)
                    eval_loss = dd['loss']
                    eval_cross_entropy = dd['CrossEntropy']
                    eval_global_step = dd['global_step']
                    eval_accuracy = dd['Accuracy']
                    # Other information
                    input_fn = data_util.get_input(data_dir, data=model_config.dataset, data_format=model_config.data_format, repeat_num=1)
                    all_activations, samples_per_object, layer_names, layer_indices, layer_n_neurons = elu.extract_layers(input_fn, root_dir, model_config)
                  except tf.errors.InvalidArgumentError:
                    failures += [filename]
                    continue
                  except tf.errors.NotFoundError:
                    failures += [filename]
                    continue
                  except ValueError:
                    failures += [filename]
                    continue
                  data_titles = np.zeros(len(layer_names), dtype=np.object)
                  data_titles[:] = layer_names
                  [MAX_NEURONS, N_LAYERS, N_SAMPLES, N_OBJECTS] = all_activations.shape
                  fd = {'model_type':model_type, 'dataset':dataset, 
                        'wide_multiplier':wide_multiplier, 'batchnorm':batchnorm, 
                        'dropout_prob':dropout_prob, 'data_augmentation':augmentation, 
                        'l2_decay_factor':decay_fac, 'normalization':normalization, 
                        'learning_rate':learning_rate, 'copy':copy,
                        'tuning_function':all_activations, 'samples_per_object':samples_per_object, 
                        'data_titles':data_titles, 'layer_indices':layer_indices, 'layer_n_neurons':layer_n_neurons,
                        'MAX_NEURONS':MAX_NEURONS, 'N_LAYERS':N_LAYERS, 
                        'N_SAMPLES':N_SAMPLES, 'N_OBJECTS':N_OBJECTS,
                        'eval_loss':eval_loss, 'eval_cross_entropy':eval_cross_entropy,
                        'eval_global_step':eval_global_step, 'eval_accuracy':eval_accuracy,
                        'train_loss':train_loss, 'train_cross_entropy':train_cross_entropy,
                        'train_global_step':train_global_step, 'train_accuracy':train_accuracy,
                        'filename':filename}
                  print('%s %s %s' % (model_type, dataset, filename))
                  scipy.io.savemat(name, fd);
                  
  #print('Found %d models'%count);
  if len(failures)>0:
    print('%d models failed'%len(failures))
    print(failures)
