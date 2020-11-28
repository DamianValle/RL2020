"""!@brief Lab 1, Problem 3 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 3 solution.
@author Martin Schuck, Damian Valle
@date 26.11.2020
"""

import numpy as np
import town as tw
import matplotlib.pyplot as plt

def main():
    town = np.array([[0, 0, 0, 0,],
                     [0, 2, 0, 0,],
                     [0, 0, 0, 0,],
                     [0, 0, 0, 0,]])
    tw.draw_town(town)
    
    env = tw.town(town)
    print('Starting Q learning, takes a few minutes for 1e7 iterations.')
    gamma = 0.8
    _, policy, v_hist = tw.q_learning(env, gamma, verbose=False)

    # Plot the required change of value function q
    plt.plot(v_hist)
    plt.xlabel("Iterations as multiples of 10 000")
    plt.ylabel("Expected reward")
    plt.title("Value function at initial state for all actions over time.")
    plt.legend(['Stay', 'Left', 'Right', 'Up', 'Down'])
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.show()
    
    print('Starting Sarsa, takes a few minutes for 5e6 iterations.')
    # Plot the different convergence rates for epsilon.
    eps_hist = list()
    for epsilon in [0.01, 0.1, 0.3, 0.5, 0.9]:
        _, policy, v_hist = tw.sarsa(env, epsilon, gamma, verbose=False)
        eps_hist.append([x[2] for x in v_hist])
    for hist in eps_hist:
        plt.plot(hist)
    plt.xlabel("Iterations as multiples of 10 000")
    plt.ylabel("Expected reward")
    plt.title("Value function at initial state for different epsilons and action right.")
    plt.legend(['0.01', '0.1', '0.3', '0.5', '0.9'])
    fig = plt.gcf()
    fig.set_size_inches(10, 7)
    plt.show()


if __name__ == '__main__':
    main()