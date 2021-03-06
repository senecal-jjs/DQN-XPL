import random
import numpy as np


class LinearAnnealedPolicy():
    def __init__(self, session, env, num_iterations, eps_init=1, eps_final=0.01):
        self.session = session
        self.env = env
        self.eps_final = eps_final
        self.eps_decrement = (eps_init - eps_final) / num_iterations
        self.current_eps = eps_init

    def select_action(self, current_q_func, input_batch, current_observation):
        if random.random() < self.current_eps:
            action = self.env.action_space.sample()
        else:
            q_vals = self.session.run(current_q_func, {current_observation: input_batch[None, :]})
            action = np.argmax(q_vals)

        # Update epsilon
        if self.current_eps > self.eps_final:
            self.current_eps -=  1.5*self.eps_decrement
        return action
