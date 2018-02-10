import pickle
import matplotlib.pyplot as plt


data = pickle.load(open('ram2_lr1.0_6000000_data.pkl', 'rb'))
data2 = pickle.load(open('ram_lr1.0_6600000_data.pkl', 'rb'))

reward = data['mean_reward_log']
reward2 = data2['mean_reward_log']
t = data['t_log']
t2 = data2['t_log']

plt.plot(t, reward)
plt.plot(t2, reward2)
plt.show()
