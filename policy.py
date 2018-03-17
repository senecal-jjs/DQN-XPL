import random
import numpy as np

# Epsilon greedy policy
class LinearAnnealedPolicy():
    def __init__(self, session, env, num_iterations, eps_init=1, eps_final=0.1):
        self.session = session
        self.env = env
        self.eps_final = eps_final
        self.eps_decrement = ((eps_init - eps_final) / num_iterations)
        self.eps_init = eps_init 
        self.cur_val = eps_init

    def select_action(self, current_q_func, input_batch, current_observation, time):
        f = lambda t : -self.eps_decrement*t + self.eps_init
        eps = np.maximum(f(time), self.eps_final)
        if random.random() < self.cur_val:
            action = self.env.action_space.sample()
        else:
            q_vals = self.session.run(current_q_func, {current_observation: input_batch[None, :]})
            action = np.argmax(q_vals)

        # # Update epsilon
        self.cur_val = eps
        # if self.cur_val > self.eps_final:
        #     self.cur_val -=  1.5*self.eps_decrement
        return action


# Boltzmann policy, select action according to probability distribution produced by q values 
class BoltzmannPolicy():
    def __init__(self, session, num_iterations, temp_init=1, temp_final=0.1):
        self.session = session 
        self.temp_decrement = (temp_init-temp_final)/(num_iterations*2)
        self.temp_func = lambda t : 0.4*np.sin(2*np.pi*(1/2000000)*t) + 0.5 # lambda t : -self.temp_decrement*t + temp_init
        self.cur_val = temp_init
        self.temp_final = temp_final

    def select_action(self, current_q_func, input_batch, current_observation, time):
        q_vals = self.session.run(current_q_func, {current_observation: input_batch[None, :]})
        self.cur_val = np.maximum(self.temp_func(time), self.temp_final)
        q_probs = self.softmax(q_vals.flatten(), self.cur_val)
        action_value = np.random.choice(q_probs,p=q_probs)
        action = np.argmax(q_probs == action_value)
        return action 
        
    def softmax(self, x, T):
        """Compute softmax values for each sets of scores in x."""
        return np.exp(x/T) / np.sum(np.exp(x/T), axis=0)
