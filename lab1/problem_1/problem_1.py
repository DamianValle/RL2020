"""!@brief Lab 1, Problem 1 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 1 solution.
@author Martin Schuck, Damian Valle
@date 21.11.2020
"""


import numpy as np
import maze as mz
import matplotlib.pyplot as plt

def main():
    # Define maze properties first
    maze = np.array([[0, 0, 1, 0, 0, 0, 0, 0],
                     [0, 0, 1, 0, 0, 1, 0, 0], 
                     [0, 0, 1, 0, 0, 1, 1, 1],
                     [0, 0, 1, 0, 0, 1, 0, 0],
                     [0, 0, 0, 0, 0, 0, 0, 0],
                     [0, 1, 1, 1, 1, 1, 1, 0], 
                     [0, 0, 0, 0, 1, 2, 0, 0]
                    ])
    mz.draw_maze(maze)
    env = mz.Maze(maze, minotaur_stay=False)
    # Player starts at (0,0), Minotaur at (6,5).
    start = (0,0,6,5)
    
    # Demonstrate optimal policy for T = 20.
    V, policy = mz.dynamic_programming(env, 20)
    demo_policy(env, policy)
    
    # Demonstrate optimal policy for T = 20 and minotaur stay.
    env = mz.Maze(maze, minotaur_stay=True)
    _, policy = mz.dynamic_programming(env, 20)
    demo_policy(env, policy)
    
    # Plot survival probability as a function of T.
    plot_survival_prob(maze, start, minotaur_stay=False)
    plot_survival_prob(maze, start, minotaur_stay=True)
    
    # Calculate geometric survival rate by solving for infinite MDP with discount factor 29/30.
    gamma = 29/30  # Geometric distribution, 30 on average.
    epsilon = 0.0001
    _, policy = mz.value_iteration(env, gamma, epsilon)
    success_cnt = 0
    for _ in range(10000):
        path = env.simulate_val_iter(start, policy)
        if path[-1][0:2] == (6,5):
            success_cnt += 1
    print(f"Probability of succeeding with E[T]=30: {success_cnt/1e4}")

    
def demo_policy(env, policy):
    """!@brief Visualizes the moves of a given policy.
    
    Minotaur always fixed at (4,4). Moves are shown as arrows. 
    """
    LIGHT_GREEN  = '#95FD99'
    BLACK        = '#000000'
    WHITE        = '#FFFFFF'
    LIGHT_PURPLE = '#E8D0FF'

    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN}

    # Size of the maze
    rows,cols = env.maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation at time step 0')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[env.maze[j,i]] for i in range(cols)] for j in range(rows)]
    
    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))
    
    # Create a table to color
    grid = plt.table(cellText=None, cellColours=colored_maze, cellLoc='center',loc=(0,0),edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    minotaur_pos = (4,4)
    grid.get_celld()[(minotaur_pos)].set_facecolor(LIGHT_PURPLE)
    grid.get_celld()[(minotaur_pos)].get_text().set_text('Minotaur')
    for x in range(7):
        for y in range(8):
            if env.maze[x,y] != 1 and (x,y) != (6,5) and (x,y) != minotaur_pos:
                a = policy[env.map[(x,y,*minotaur_pos)],0]
                # New markings
                if a == 0: 
                    arrow = 'wait'
                elif a == 1:
                    arrow = '\u2190'
                elif a == 2:
                    arrow = '\u2192'
                elif a == 3:
                    arrow = '\u2191'
                else:
                    arrow = '\u2193'
                grid.get_celld()[(x,y)].get_text().set_text(arrow)
    plt.show()
    

def plot_survival_prob(maze, start, minotaur_stay):
    """!@brief Plot function for the survival probability by simulating 10000 runs.
    
    @warning Takes around a minute to complete due to the 20*10K simulations.
    """
    env = mz.Maze(maze, minotaur_stay=minotaur_stay)
    p = np.zeros((20,1))
    for i in range(1,21):
        success_cnt = 0
        V, policy = mz.dynamic_programming(env, i)
        for _ in range(10000):
            path = env.simulate_dyn_prog(start, policy)
            if path[-1][0:2] == (6,5):
                success_cnt += 1
        p[i-1] = success_cnt/1e4
    plt.plot(range(1,21), p)
    rest = "resting" if minotaur_stay is True else "unresting"
    plt.title(f"Maze problem with {rest} minotaur.")
    plt.ylabel('Survival probability')
    plt.xlabel('Episode length')
    plt.show()
    
    
if __name__ == '__main__':
    main()