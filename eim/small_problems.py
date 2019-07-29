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

"""Synthetic data experiments."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
from matplotlib import cm
import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


from eim.models import his
from eim.models import lars
from eim.models import nis
from eim.models import rejection_sampling
import eim.small_problems_dists as dists

tfd = tfp.distributions

tf.logging.set_verbosity(tf.logging.INFO)
tf.app.flags.DEFINE_enum("algo", "lars",
                         ["lars", "nis", "his", "rejection_sampling"],
                         "The algorithm to run.")
tf.app.flags.DEFINE_boolean(
    "lars_allow_eval_target", False,
    "Whether LARS is allowed to evaluate the target density.")
tf.app.flags.DEFINE_enum("target", dists.NINE_GAUSSIANS_DIST,
                         dists.TARGET_DISTS, "Distribution to draw data from.")
tf.app.flags.DEFINE_float(
    "nine_gaussians_variance", 0.01,
    "Variance for the mixture components in the nine gaussians.")
tf.app.flags.DEFINE_string(
    "energy_fn_sizes", "20,20",
    "List of hidden layer sizes for energy function as as comma "
    "separated list.")
tf.app.flags.DEFINE_integer(
    "his_t", 5, "Number of steps for hamiltonian importance sampling.")
tf.app.flags.DEFINE_float("his_stepsize", 1e-2,
                          "Stepsize for hamiltonian importance sampling.")
tf.app.flags.DEFINE_float("his_alpha", 0.995,
                          "Alpha for hamiltonian importance sampling.")
tf.app.flags.DEFINE_boolean("his_learn_stepsize", False,
                            "Allow HIS to learn the stepsize")
tf.app.flags.DEFINE_boolean("his_learn_alpha", False,
                            "Allow HIS to learn alpha.")
tf.app.flags.DEFINE_float("learning_rate", 3e-4,
                          "The learning rate to use for ADAM or SGD.")
tf.app.flags.DEFINE_integer("batch_size", 128,
                            "The number of examples per batch.")
tf.app.flags.DEFINE_integer("density_num_bins", 100,
                            "Number of points per axis when plotting density.")
tf.app.flags.DEFINE_integer("density_num_samples", 100000,
                            "Number of samples to use when plotting density.")
tf.app.flags.DEFINE_integer("eval_batch_size", 1000,
                            "The number of examples per eval batch.")
tf.app.flags.DEFINE_integer("K", 1024,
                            "The number of samples for NIS and LARS.")
tf.app.flags.DEFINE_string("logdir", "/tmp/lars",
                           "Directory for summaries and checkpoints.")
tf.app.flags.DEFINE_integer("max_steps", int(1e6),
                            "The number of steps to run training for.")
tf.app.flags.DEFINE_integer("summarize_every", int(1e3),
                            "The number of steps between each evaluation.")
tf.app.flags.DEFINE_integer(
    "run", 0,
    ("A number to distinguish which run this is. This allows us to run ",
     "multiple trials with the same params."))

FLAGS = tf.app.flags.FLAGS


def exp_name():
  return "target-%s.algo-%s.K-%d.run-%d" % (
      FLAGS.target,
      FLAGS.algo,
      FLAGS.K,
      FLAGS.run,
  )


tf_viridis = lambda x: tf.py_func(cm.get_cmap("viridis"), [x], [tf.float64])


def density_image_summary(log_density, num_points, title):
  """Plot density.

  Args:
    log_density: log_density up to a constant.
    num_points: Number of points in the grid.
    title: Title of the summary.
  """
  if (FLAGS.target == dists.NINE_GAUSSIANS_DIST or
      FLAGS.target == dists.TWO_RINGS_DIST):
    bounds = (-2, 2)
  elif FLAGS.target == dists.CHECKERBOARD_DIST:
    bounds = (0, 1)

  x = tf.range(
      bounds[0], bounds[1], delta=(bounds[1] - bounds[0]) / float(num_points))
  # pylint: disable=invalid-name
  X, Y = tf.meshgrid(x, x, indexing="ij")
  XY = tf.stack([X, Y], axis=-1)

  log_z = log_density(XY)
  log_Z = reduce_logavgexp(log_z)
  log_z_centered = log_z - log_Z
  z = tf.exp(log_z_centered)
  # pylint: enable=invalid-name

  plot = tf.reshape(z, [1, num_points, num_points, 1])
  tf.summary.image(
      title, plot, max_outputs=1, collections=["infrequent_summaries"])
  log_plot = tf.reshape(log_z_centered, [1, num_points, num_points, 1])
  tf.summary.image(
      "log_%s" % title,
      log_plot,
      max_outputs=1,
      collections=["infrequent_summaries"])


def sample_image_summary(model, title, num_samples=100000, num_bins=50):
  """Creates a summary plot approximating the density with samples."""
  if FLAGS.target == dists.NINE_GAUSSIANS_DIST or FLAGS.target == dists.TWO_RINGS_DIST:
    bounds = (-2, 2)
  elif FLAGS.target == dists.CHECKERBOARD_DIST:
    bounds = (0, 1)
  data = model.sample([num_samples])

  def _hist2d(x, y):
    return np.histogram2d(x, y, bins=num_bins, range=[bounds, bounds])[0]

  tf_hist2d = lambda x, y: tf.py_func(_hist2d, [x, y], [tf.float64])
  plot = tf.expand_dims(tf_hist2d(data[:, 0], data[:, 1]), -1)
  tf.summary.image(
      title, plot, max_outputs=1, collections=["infrequent_summaries"])


def reduce_logavgexp(input_tensor, axis=None, keepdims=None, name=None):
  dims = tf.shape(input_tensor)
  if axis is not None:
    dims = tf.gather(dims, axis)
  denominator = tf.reduce_prod(dims)
  return (tf.reduce_logsumexp(
      input_tensor, axis=axis, keepdims=keepdims, name=name) -
          tf.log(tf.to_float(denominator)))


def make_lars_graph(target_dist,  # pylint: disable=invalid-name
                    K,
                    batch_size,
                    eval_batch_size,
                    lr,
                    mlp_layers,
                    dtype=tf.float32):
  """Construct the training graph for LARS."""
  model = lars.SimpleLARS(
      K=K, data_dim=2, accept_fn_layers=mlp_layers, dtype=dtype)

  train_data = target_dist.sample(batch_size)
  log_p, ema_op = model.log_prob(train_data)
  test_data = target_dist.sample(eval_batch_size)
  eval_log_p, eval_ema_op = model.log_prob(test_data)

  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(lr)
  grads = opt.compute_gradients(-tf.reduce_mean(log_p))
  with tf.control_dependencies([ema_op, eval_ema_op]):
    apply_grads_op = opt.apply_gradients(grads, global_step=global_step)

  density_image_summary(
      lambda x: tf.squeeze(model.accept_fn(x)) + model.proposal.log_prob(x),
      FLAGS.density_num_bins, "energy/lars")
  tf.summary.scalar("elbo", tf.reduce_mean(log_p))
  tf.summary.scalar("eval_elbo", tf.reduce_mean(eval_log_p))
  return -tf.reduce_mean(log_p), apply_grads_op, global_step


def make_train_graph(target_dist,
                     model,
                     batch_size,
                     eval_batch_size,
                     lr,
                     eval_num_samples=1000):
  """Code for the TRS, SNIS, and HIS training loops."""
  train_batch = target_dist.sample(batch_size)
  eval_batch = target_dist.sample(eval_batch_size)

  train_elbo = tf.reduce_mean(model.log_prob(train_batch))
  eval_elbo = tf.reduce_mean(
      model.log_prob(eval_batch, num_samples=eval_num_samples))

  tf.summary.scalar("elbo", train_elbo)
  tf.summary.scalar(
      "eval_elbo", eval_elbo, collections=["infrequent_summaries"])

  global_step = tf.train.get_or_create_global_step()
  opt = tf.train.AdamOptimizer(learning_rate=lr)
  grads = opt.compute_gradients(-train_elbo)
  train_op = opt.apply_gradients(grads, global_step=global_step)
  return train_elbo, train_op, global_step


def make_log_hooks(global_step, loss):
  """Create logging summary hooks."""
  hooks = []

  def summ_formatter(d):
    return "Step {step}, loss: {loss:.5f}".format(**d)

  loss_hook = tf.train.LoggingTensorHook({
      "step": global_step,
      "loss": loss
  },
                                         every_n_iter=FLAGS.summarize_every,
                                         formatter=summ_formatter)
  hooks.append(loss_hook)
  if tf.get_collection("infrequent_summaries"):
    infrequent_summary_hook = tf.train.SummarySaverHook(
        save_steps=1000,
        output_dir=os.path.join(FLAGS.logdir, exp_name()),
        summary_op=tf.summary.merge_all(key="infrequent_summaries"))
    hooks.append(infrequent_summary_hook)
  return hooks


def main(unused_argv):
  g = tf.Graph()
  with g.as_default():
    target = dists.get_target_distribution(
        FLAGS.target, nine_gaussians_variance=FLAGS.nine_gaussians_variance)
    energy_fn_layers = [
        int(x.strip()) for x in FLAGS.energy_fn_sizes.split(",")
    ]
    if FLAGS.algo == "lars":
      print("Running LARS")
      loss, train_op, global_step = make_lars_graph(
          target_dist=target,
          K=FLAGS.K,
          batch_size=FLAGS.batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          lr=FLAGS.learning_rate,
          mlp_layers=energy_fn_layers,
          dtype=tf.float32)
    else:
      if FLAGS.algo == "nis":
        print("Running NIS")
        model = nis.NIS(
            K=FLAGS.K, data_dim=2, energy_hidden_sizes=energy_fn_layers)
        density_image_summary(
            lambda x:  # pylint: disable=g-long-lambda
            (tf.squeeze(model.energy_fn(x)) + model.proposal.log_prob(x)),
            FLAGS.density_num_bins, "energy/nis")
      elif FLAGS.algo == "rejection_sampling":
        print("Running Rejection Sampling")
        logit_accept_fn = tf.keras.Sequential([
            tf.keras.layers.Dense(layer_size, activation="tanh")
            for layer_size in energy_fn_layers
        ] + [tf.keras.layers.Dense(1, activation=None)])
        model = rejection_sampling.RejectionSampling(
            T=FLAGS.K, data_dim=[2], logit_accept_fn=logit_accept_fn)
        density_image_summary(
            lambda x: tf.squeeze(  # pylint: disable=g-long-lambda
                tf.log_sigmoid(model.logit_accept_fn(x)), axis=-1) + model.
            proposal.log_prob(x), FLAGS.density_num_bins, "energy/trs")
      elif FLAGS.algo == "his":
        print("Running HIS")
        model = his.FullyConnectedHIS(
            T=FLAGS.his_t,
            data_dim=2,
            energy_hidden_sizes=energy_fn_layers,
            q_hidden_sizes=energy_fn_layers,
            init_step_size=FLAGS.his_stepsize,
            learn_stepsize=FLAGS.his_learn_stepsize,
            init_alpha=FLAGS.his_alpha,
            learn_temps=FLAGS.his_learn_alpha)
        density_image_summary(lambda x: -model.hamiltonian_potential(x),
                              FLAGS.density_num_bins, "energy/his")
        sample_image_summary(
            model, "density/his", num_samples=100000, num_bins=50)

      loss, train_op, global_step = make_train_graph(
          target_dist=target,
          model=model,
          batch_size=FLAGS.batch_size,
          eval_batch_size=FLAGS.eval_batch_size,
          lr=FLAGS.learning_rate)

    log_hooks = make_log_hooks(global_step, loss)
    with tf.train.MonitoredTrainingSession(
        master="",
        is_chief=True,
        hooks=log_hooks,
        checkpoint_dir=os.path.join(FLAGS.logdir, exp_name()),
        save_checkpoint_secs=120,
        save_summaries_steps=FLAGS.summarize_every,
        log_step_count_steps=FLAGS.summarize_every) as sess:
      cur_step = -1
      while True:
        if sess.should_stop() or cur_step > FLAGS.max_steps:
          break
        # run a step
        _, cur_step = sess.run([train_op, global_step])


if __name__ == "__main__":
  tf.app.run(main)
