import sys
import gym.spaces
import itertools
import numpy as np
import random
import tensorflow                as tf
import tensorflow.contrib.layers as layers
from collections import namedtuple
from dqn_utils import *
from memory import ReplayBuffer
import pickle

OptimizerSpec = namedtuple("OptimizerSpec", ["constructor", "kwargs", "lr_schedule"])

def learn(env,
          policy,
          q_func,
          optimizer_spec,
          session,
          stopping_criterion=None,
          replay_buffer_size=1000000,
          batch_size=32,
          gamma=0.99,
          learning_starts=50000,
          learning_freq=4,
          frame_history_len=4,
          target_update_freq=10000,
          grad_norm_clipping=10,
          lr_multiplier=1.0):
    """Run Deep Q-learning algorithm.
    You can specify your own convnet using q_func.
    All schedules are w.r.t. total number of steps taken in the environment.
    Parameters
    ----------
    env: gym.Env
        gym environment to train on.
    q_func: function
        Model to use for computing the q function. It should accept the
        following named arguments:
            img_in: tf.Tensor
                tensorflow tensor representing the input image
            num_actions: int
                number of actions
            scope: str
                scope in which all the model related variables
                should be created
            reuse: bool
                whether previously created variables should be reused.
    optimizer_spec: OptimizerSpec
        Specifying the constructor and kwargs, as well as learning rate schedule
        for the optimizer
    session: tf.Session
        tensorflow session to use.
    exploration: rl_algs.deepq.utils.schedules.Schedule
        schedule for probability of chosing random action.
    stopping_criterion: (env, t) -> bool
        should return true when it's ok for the RL algorithm to stop.
        takes in env and the number of steps executed so far.
    replay_buffer_size: int
        How many memories to store in the replay buffer.
    batch_size: int
        How many transitions to sample each time experience is replayed.
    gamma: float
        Discount Factor
    learning_starts: int
        After how many environment steps to start replaying experiences
    learning_freq: int
        How many steps of environment to take between every experience replay
    frame_history_len: int
        How many past frames to include as input to the model.
    target_update_freq: int
        How many experience replay rounds (not steps!) to perform between
        each update to the target Q network
    grad_norm_clipping: float or None
        If not None gradients' norms are clipped to this value.
    """
    assert type(env.observation_space) == gym.spaces.Box
    assert type(env.action_space)      == gym.spaces.Discrete

    ###############
    # BUILD MODEL #
    ###############

    if len(env.observation_space.shape) == 1:
        # This means we are running on low-dimensional observations (e.g. RAM)
        input_shape = env.observation_space.shape
    else:
        img_h, img_w, img_c = env.observation_space.shape
        input_shape = (img_h, img_w, frame_history_len * img_c)
    num_actions = env.action_space.n

    # set up placeholders
    # placeholder for current observation (or state)
    obs_t_ph              = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for current action
    act_t_ph              = tf.placeholder(tf.int32,   [None])
    # placeholder for current reward
    rew_t_ph              = tf.placeholder(tf.float32, [None])
    # placeholder for next observation (or state)
    obs_tp1_ph            = tf.placeholder(tf.uint8, [None] + list(input_shape))
    # placeholder for end of episode mask
    # this value is 1 if the next state corresponds to the end of an episode,
    # in which case there is no Q-value at the next state; at the end of an
    # episode, only the current state reward contributes to the target, not the
    # next state Q-value (i.e. target is just rew_t_ph, not rew_t_ph + gamma * q_tp1)
    done_mask_ph          = tf.placeholder(tf.float32, [None])

    # casting to float on GPU ensures lower data transfer times.
    obs_t_float   = tf.cast(obs_t_ph,   tf.float32) / 255.0
    obs_tp1_float = tf.cast(obs_tp1_ph, tf.float32) / 255.0

    # Declare variables for logging
    t_log = []
    mean_reward_log = []
    best_mean_log = []
    episodes_log = []
    exploration_log = []
    learning_rate_log = []

    # Create a network to produce the current q values for each possible action
    current_q_func = q_func(obs_t_float, num_actions, scope="q_func", reuse=False) # Current Q-Value Function
    q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='q_func')

    # Creat the target q function network
    target_q_func = q_func(obs_tp1_float, num_actions, scope="target_q_func", reuse=False) # Target Q-Value Function
    target_q_func_vars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='target_q_func')

    # Encode actions as as a one hot vector, based on the action that was chosen
    act_t = tf.one_hot(act_t_ph, depth=num_actions, dtype=tf.float32, name="action_one_hot")
    q_act_t = tf.reduce_sum(act_t*current_q_func, axis=1)

    # Calculate the current reward, and use that to get the loss function
    y = rew_t_ph + gamma * tf.reduce_max(target_q_func, reduction_indices=[1])
    total_error = tf.square(tf.subtract(y, q_act_t)) #(reward + gamma*V(s') - Q(s, a))**2

    # construct optimization op (with gradient clipping)
    learning_rate = tf.placeholder(tf.float32, (), name="learning_rate")
    optimizer = optimizer_spec.constructor(learning_rate=learning_rate, **optimizer_spec.kwargs)
    train_fn = minimize_and_clip(optimizer, total_error,
                 var_list=q_func_vars, clip_val=grad_norm_clipping)

    # update_target_fn will be called periodically to copy Q network to target Q network
    update_target_fn = []
    for var, var_target in zip(sorted(q_func_vars,        key=lambda v: v.name),
                               sorted(target_q_func_vars, key=lambda v: v.name)):
        update_target_fn.append(var_target.assign(var))
    update_target_fn = tf.group(*update_target_fn)

    # construct the replay buffer
    replay_buffer = ReplayBuffer(replay_buffer_size, frame_history_len)

    ###############
    # RUN ENV     #
    ###############
    model_initialized = False
    num_param_updates = 0
    mean_episode_reward      = -float('nan')
    best_mean_episode_reward = -float('inf')
    last_obs = env.reset()
    LOG_EVERY_N_STEPS = 10000
    SAVE_EVERY_N_STEPS = 200000

    for t in itertools.count():
        ### 1. Check stopping criterion
        if stopping_criterion is not None and stopping_criterion(env, t):
            break

        ### 2. Step the env and store the transition
        # At this point, "last_obs" contains the latest observation that was
        # recorded from the simulator.
        # Note that you cannot use "last_obs" directly as input
        # into your network, since it needs to be processed to include context
        # from previous frames. The replay buffer has a function called
        # encode_recent_observation that will take the latest observation
        # that you pushed into the buffer and compute the corresponding
        # input that should be given to a Q network by appending some
        # previous frames.

        # Store last_obs into replay buffer
        idx = replay_buffer.store_frame(last_obs)

        if t == 0:
            act, reward, done = env.action_space.sample(), 0, False

        # Choose action
        if not model_initialized:
            # choose random action
            act = env.action_space.sample()
        else:
            input_batch = replay_buffer.encode_recent_observation()
            act = policy.select_action(current_q_func, input_batch, obs_t_ph)

        # Step simulator forward one step
        last_obs, reward, done, info = env.step(act)
        replay_buffer.store_effect(idx, act, reward, done) # Store action taken after last_obs and corresponding reward

        if done == True: # done was True in latest transition; we have already stored that
            last_obs = env.reset() # Reset observation
            done = False

        #####

        # at this point, the environment should have been advanced one step (and
        # reset if done was true), and last_obs should point to the new latest
        # observation

        ### 3. Perform experience replay and train the network.
        # note that this is only done if the replay buffer contains enough samples
        # for us to learn something useful -- until then, the model will not be
        # initialized and random actions should be taken
        if (t > learning_starts and
                t % learning_freq == 0 and
                replay_buffer.can_sample(batch_size)):
            # Here, you should perform training. Training consists of four steps:
            # 3.a: use the replay buffer to sample a batch of transitions (see the
            # replay buffer code for function definition, each batch that you sample
            # should consist of current observations, current actions, rewards,
            # next observations, and done indicator).
            # 3.b: initialize the model if it has not been initialized yet; to do
            # that, call
            #    initialize_interdependent_variables(session, tf.global_variables(), {
            #        obs_t_ph: obs_t_batch,
            #        obs_tp1_ph: obs_tp1_batch,
            #    })
            # where obs_t_batch and obs_tp1_batch are the batches of observations at
            # the current and next time step. The boolean variable model_initialized
            # indicates whether or not the model has been initialized.
            # Remember that you have to update the target network too (see 3.d)!
            # 3.c: train the model. To do this, you'll need to use the train_fn and
            # total_error ops that were created earlier: total_error is what you
            # created to compute the total Bellman error in a batch, and train_fn
            # will actually perform a gradient step and update the network parameters
            # to reduce total_error. When calling session.run on these you'll need to
            # populate the following placeholders:
            # obs_t_ph
            # act_t_ph
            # rew_t_ph
            # obs_tp1_ph
            # done_mask_ph
            # (this is needed for computing total_error)
            # learning_rate -- you can get this from optimizer_spec.lr_schedule.value(t)
            # (this is needed by the optimizer to choose the learning rate)
            # 3.d: periodically update the target network by calling
            # session.run(update_target_fn)
            # you should update every target_update_freq steps, and you may find the
            # variable num_param_updates useful for this (it was initialized to 0)
            #####

            # 3.a Sample a batch of transitions
            obs_t_batch, act_batch, rew_batch, obs_tp1_batch, done_mask = replay_buffer.sample(batch_size)

            # 3.b Initialize model if not initialized yet
            if not model_initialized:
                initialize_interdependent_variables(session, tf.global_variables(), {
                   obs_t_ph: obs_t_batch,
                   obs_tp1_ph: obs_tp1_batch,
                })
                session.run(update_target_fn)
                model_initialized = True

            # 3.c Train the model using train_fn and total_error
            session.run(train_fn, {obs_t_ph: obs_t_batch, act_t_ph: act_batch, rew_t_ph: rew_batch, obs_tp1_ph: obs_tp1_batch,
                done_mask_ph: done_mask, learning_rate: optimizer_spec.lr_schedule.value(t)})

            # 3.d Update target network every target_update_freq steps
            if t % target_update_freq == 0:
                session.run(update_target_fn)
                num_param_updates += 1
            #####

        ### 4. Log progress
        episode_rewards = get_wrapper_by_name(env, "Monitor").get_episode_rewards()
        if len(episode_rewards) > 0:
            mean_episode_reward = np.mean(episode_rewards[-100:])
        if len(episode_rewards) > 100:
            best_mean_episode_reward = max(best_mean_episode_reward, mean_episode_reward)
        if t % LOG_EVERY_N_STEPS == 0 and model_initialized:
            print("Timestep %d" % (t,))
            t_log.append(t)
            print("mean reward (100 episodes) %f" % mean_episode_reward)
            mean_reward_log.append(mean_episode_reward)
            print("best mean reward %f" % best_mean_episode_reward)
            best_mean_log.append(best_mean_episode_reward)
            print("episodes %d" % len(episode_rewards))
            episodes_log.append(len(episode_rewards))
            print("exploration %f" % policy.current_eps)
            exploration_log.append(policy.current_eps)
            print("learning_rate %f" % optimizer_spec.lr_schedule.value(t))
            learning_rate_log.append(optimizer_spec.lr_schedule.value(t))
            sys.stdout.flush()

        if t % SAVE_EVERY_N_STEPS == 0 and model_initialized:
            training_log = ({'t_log': t_log, 'mean_reward_log': mean_reward_log, 'best_mean_log': best_mean_log, 'episodes_log': episodes_log,
                'exploration_log': exploration_log, 'learning_rate_log': learning_rate_log})
            output_file_name = 'ram_lr'+str(lr_multiplier)+'_' + str(t) + '_data.pkl'
            with open(output_file_name, 'wb') as f:
                pickle.dump(training_log, f)
