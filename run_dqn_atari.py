import argparse
import gym
from gym import wrappers
import os.path as osp
import random
import numpy as np
import tensorflow as tf
import tensorflow.contrib.layers as layers

import dqn
from dqn_utils import *
from atari_wrappers import *
from policy import LinearAnnealedPolicy


def atari_model(img_in, num_actions, scope, reuse=False):
    # as described in https://storage.googleapis.com/deepmind-data/assets/papers/DeepMindNature14236Paper.pdf
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope("convnet"):
            # original architecture
            out = layers.convolution2d(out, num_outputs=32, kernel_size=8, stride=4, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=4, stride=2, activation_fn=tf.nn.relu)
            out = layers.convolution2d(out, num_outputs=64, kernel_size=3, stride=1, activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope("action_value"):
            out = layers.fully_connected(out, num_outputs=512,         activation_fn=tf.nn.relu)
            out = layers.fully_connected(out, num_outputs=num_actions, activation_fn=None)

        return out

def atari_learn(env, session, diff_argument, num_timesteps):
    # This is just a rough estimate
    num_iterations = float(num_timesteps) / 4.0

    lr_multiplier = 1.0
    lr_schedule = PiecewiseSchedule([(0, 1e-4 * lr_multiplier),
                                     (num_iterations / 5, 1e-4 * lr_multiplier),
                                     (num_iterations / 2,  5e-5 * lr_multiplier)],
                                      outside_value=5e-5 * lr_multiplier)
    optimizer = dqn.OptimizerSpec(
        constructor=tf.train.AdamOptimizer,
        kwargs=dict(epsilon=1e-4),
        lr_schedule=lr_schedule
    )

    def stopping_criterion(env, t):
        # notice that here t is the number of steps of the wrapped env,
        # which is different from the number of steps in the underlying env
        return get_wrapper_by_name(env, "Monitor").get_total_steps() >= num_timesteps

    # Create action exploration/exploitation policy
    policy = LinearAnnealedPolicy(session=session, env=env, num_iterations=num_iterations)

    dqn.learn(
        env,
        diff_argument,
        policy = policy,
        q_func=atari_model,
        optimizer_spec=optimizer,
        session=session,
        #exploration=exploration_schedule,
        stopping_criterion=stopping_criterion,
        replay_buffer_size=1000000,
        batch_size=32,
        gamma=0.99,
        learning_starts=50000,
        learning_freq=6,
        frame_history_len=1,
        target_update_freq=10000,
        grad_norm_clipping=10
    )
    env.close()


def set_global_seeds(i):
    try:
        import tensorflow as tf
    except ImportError:
        pass
    else:
        tf.set_random_seed(i)
    np.random.seed(i)
    random.seed(i)

def get_session():
    tf.reset_default_graph()
    tf_config = tf.ConfigProto(
        inter_op_parallelism_threads=1,
        intra_op_parallelism_threads=1)
    session = tf.Session(config=tf_config)
    # print("AVAILABLE GPUS: ", get_available_gpus())
    return session

def get_env(task, seed):
    env_id = task.env_id

    env = gym.make(env_id)

    set_global_seeds(seed)
    env.seed(seed)

    #expt_dir = '/home/jsenec/repos/DQN-XPL/vid'
    expt_dir = '/Users/jsen/Repos/DQN-XPL/vid'

    # Some code to change the rate at which videos are recorded
    env = wrappers.Monitor(env, osp.join(expt_dir, "gym"), force=True, video_callable=lambda episode_id: episode_id%500==0)
    env = wrap_deepmind(env)

    return env

def main(diff_argument):
    # Get Atari games.
    benchmark = gym.benchmark_spec('Atari40M')

    # Change the index to select a different game. 3 is pong, 4 is Q*bert, 1 is breakout, 5 is seaquest, 6 is space invaders
    task = benchmark.tasks[6]

    # Run training
    seed = 0 #random.randint(0, 5) # Use a seed of zero (you may want to randomize the seed!)
    env = get_env(task, seed)
    session = get_session()
    atari_learn(env, session, diff_argument, num_timesteps=task.max_timesteps)
    # atari_learn(env, session, diff_argument, num_timesteps=10000000)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-d", "--difference", help="use difference frames", action="store_true")
    args = parser.parse_args()
    main(args.difference)
