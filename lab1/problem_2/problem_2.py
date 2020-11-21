import numpy as np
import bank as bk
import matplotlib.pyplot as plt


def main():
    bank = np.array([[2, 0, 0, 0, 0, 2],
                     [0, 0, 0, 0, 0, 0], 
                     [2, 0, 0, 0, 0, 2]])
    # Environment setup 
    env = bk.Bank(bank)
    values = np.zeros((20,1))
    gamma = 0.1
    epsilon = 0.0001
    
    # Compute the value function for different gamma and plot them.
    for index, gamma in enumerate(np.linspace(0.01,0.99,20)):
        V, _ = bk.value_iteration(env, gamma, epsilon)
        values[index] = V[env.map[(0,0,1,2)]]
    plt.plot(np.linspace(0.01,0.99,20), values)
    plt.xlabel("Gamma")
    plt.ylabel("Expected reward")
    plt.title("Value function at initial state as a function of gamma")
    plt.show()
    
    # Illustrate the different policies for small and large discount factors.
    _, policy = bk.value_iteration(env, 0.01, epsilon)
    bk.illustrate_policy(env, policy, 0.01)
    _, policy = bk.value_iteration(env, 0.99, epsilon)
    bk.illustrate_policy(env, policy, 0.99)
    

if __name__ == '__main__':
    main()