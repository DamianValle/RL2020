"""!@brief Lab 1, Problem 1 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 1 module.
@author Martin Schuck, Damian Valle
@date 21.11.2020
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Implemented methods
methods = ['DynProg', 'ValIter']

# Some colours
LIGHT_RED    = '#FFC4CC'
LIGHT_GREEN  = '#95FD99'
GRAY         = 'C0C0C0'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -100


    def __init__(self, maze, minotaur_stay=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze
        self.actions                  = self.__actions()
        self.minotaur_actions         = self.__minotaur_actions(minotaur_stay)
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions
    
    def __minotaur_actions(self, minotaur_stay):
        actions = dict()
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        if minotaur_stay:
            actions[self.STAY]   = (0, 0)
        return actions

    def __states(self):
        states = dict()
        map = dict()
        s = 0
        for i in range(self.maze.shape[0]):  # Player position
            for j in range(self.maze.shape[1]):
                for k in range(self.maze.shape[0]):  # Minotaur position
                    for l in range(self.maze.shape[1]):
                        if self.maze[i,j] != 1 and self.maze[k,l] != 1:
                            states[s] = (i, j, k, l)
                            map[(i, j, k, l)] = s
                            s += 1
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place. If the agent was caught 
            before, it stays in its position.

            Returns a list of possible next states (x,y, mino_x, mino_y) of the maze that agent transitions to, as well 
            as an indicator whether the action resulted in a wall hit.
        """
        # Is the agent caught? If so, minotaur stays at agent.
        if self.__check_caught(state):
            return [state], False
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_maze_walls =  self.__maze_wall_hit(row, col)
        # Compute possible future minotaur positions.
        minotaur_positions = self.__future_minotaur_positions(state)
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return [self.map[(self.states[state][0], self.states[state][1], min_pos[0], min_pos[1])] 
                    for min_pos in minotaur_positions], True
        else:
            return [self.map[(row, col, min_pos[0], min_pos[1])] for min_pos in minotaur_positions], False
        
    def __check_caught(self, state):
        row = self.states[state][0]
        col = self.states[state][1]
        return row == self.states[state][2] and col == self.states[state][3] and not self.maze[row,col] == 2
    
    def __future_minotaur_positions(self, state):
        minotaur_positions = list()
        for action in self.minotaur_actions:
            row = self.states[state][2] + self.minotaur_actions[action][0]
            col = self.states[state][3] + self.minotaur_actions[action][1]
            if not self.__maze_wall_hit(row, col):
                minotaur_positions.append((row, col))
            else:  # Minotaur can go through walls of thickness 1
                row += self.minotaur_actions[action][0]
                col += self.minotaur_actions[action][1]
                if not self.__maze_wall_hit(row, col):
                    minotaur_positions.append((row, col))
        return minotaur_positions
    
    def __maze_wall_hit(self, row, col):
        return (row <= -1) or (row >= self.maze.shape[0]) or (col <= -1) or \
               (col >= self.maze.shape[1]) or (self.maze[row,col] == 1)


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states, _ = self.__move(s,a)
                for next_s in next_states:
                    transition_probabilities[next_s, s, a] = 1/len(next_states)  # Minotaur walks random.
        return transition_probabilities

    def __rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states, wall_hit = self.__move(s,a)
                # Compute the average reward for action (s,a).
                reward = 0
                for next_s in next_states:
                    if len(next_states) == 1:  # Caught by the minotaur, minotaur doesn't move -> only one state.
                        reward += self.STEP_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for hitting a wall and not being caught.
                    elif wall_hit:
                        reward += self.IMPOSSIBLE_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for reaching the exit
                    elif self.states[s][0:2] == self.states[next_s][0:2] and self.maze[self.states[next_s][0:2]] == 2:
                        reward += self.GOAL_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        reward += self.STEP_REWARD * self.transition_probabilities[next_s, s, a]
                rewards[s,a] = reward
        return rewards

    def simulate_dyn_prog(self, start, policy):
        path = list()
        # Deduce the horizon from the policy shape
        horizon = policy.shape[1]
        # Initialize current state and time
        t = 0
        s = self.map[start]
        # Add the starting position in the maze to the path
        path.append(start)
        while t < horizon-1:
            # Move to next state given the policy and the current state
            next_s, _ = self.__move(s,policy[s,t])
            # __move returns all possible next states, choose next state uniformly at random.
            next_s = np.random.choice(next_s, 1)[0]
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t +=1
            s = next_s
        return path
                
    def simulate_val_iter(self, start, policy):
        path = list()
        # Initialize current state, next state, end and time
        t = 1
        s = self.map[start]
        end_tmp = np.where(self.maze == 2)
        end = (end_tmp[0][0], end_tmp[1][0])
        # Add the starting position in the maze to the path
        path.append(start)
        # Move to next state given the policy and the current state
        next_s, _ = self.__move(s,policy[s])
        # __move returns all possible next states, choose next state uniformly at random.
        next_s = np.random.choice(next_s, 1)[0]
        # Add the position in the maze corresponding to the next state
        # to the path
        path.append(self.states[next_s])
        # Loop while state is not the goal state
        T = np.random.geometric(p=1/30)
        while self.states[next_s][0:2] != end and self.states[next_s][0:2] != self.states[next_s][2:4]:
            if T == 0:
                break
            T -= 1
            # Update state
            s = next_s
            # Move to next state given the policy and the current state
            next_s, _ = self.__move(s,policy[s])
            # __move returns all possible next states, choose next state uniformly at random.
            next_s = np.random.choice(next_s, 1)[0]
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s])
            # Update time and state for next iteration
            t +=1
        return path

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions
    T         = horizon

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1))
    policy = np.zeros((n_states, T+1))
    Q      = np.zeros((n_states, n_actions))


    # Initialization
    Q            = np.copy(r)
    V[:, T]      = np.max(Q,1)
    policy[:, T] = np.argmax(Q,1)

    # The dynamic programming backwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a],V[:,t+1])
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t] = np.max(Q,1)
        # The optimal action is the one that maximizes the Q function
        policy[:,t] = np.argmax(Q,1)
    return V, policy


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float gamma        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities
    r         = env.rewards
    n_states  = env.n_states
    n_actions = env.n_actions

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states)
    Q   = np.zeros((n_states, n_actions))
    BV  = np.zeros(n_states)
    # Iteration counter
    n   = 0
    # Tolerance error
    tol = (1 - gamma)* epsilon/gamma

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
    BV = np.max(Q, 1)

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < 200:
        # Increment by one the numbers of iteration
        n += 1
        # Update the value function
        V = np.copy(BV)
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + gamma*np.dot(p[:,s,a],V)
        BV = np.max(Q, 1)
        # Show error
        #print(np.linalg.norm(V - BV))

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The Maze')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = maze.shape
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

def animate_solution(maze, path):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED}

    # Size of the maze
    rows,cols = maze.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for i in range(len(path)):
        player_pos = path[i][0:2]
        minotaur_pos = path[i][2:4]
        # New markings
        grid.get_celld()[(player_pos)].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(player_pos)].get_text().set_text('Player')
        grid.get_celld()[(minotaur_pos)].set_facecolor(LIGHT_PURPLE)
        grid.get_celld()[(minotaur_pos)].get_text().set_text('Minotaur')
        if i > 0:
            # Reset old markings if not marked by new.
            if not player_pos == path[i-1][0:2] and not minotaur_pos == path[i-1][0:2]:
                grid.get_celld()[(path[i-1][0:2])].set_facecolor(col_map[maze[path[i-1][0:2]]])
                grid.get_celld()[(path[i-1][0:2])].get_text().set_text('')
            if not player_pos == path[i-1][2:4] and not minotaur_pos == path[i-1][2:4]:
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[maze[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')
            if maze[player_pos] == 2:
                grid.get_celld()[(player_pos)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(player_pos)].get_text().set_text('Player is out')
            elif player_pos == minotaur_pos and not maze[player_pos] == 2:
                grid.get_celld()[(player_pos)].set_facecolor(GRAY)
                grid.get_celld()[(player_pos)].get_text().set_text('Player dead')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.7)
