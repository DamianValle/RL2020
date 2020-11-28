"""!@brief Lab 1, Problem 3 of the 2020/2021 Reinforcement Learning lecture at KTH.

@file Problem 3 module.
@author Martin Schuck, Damian Valle
@date 26.11.2020
"""

import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display

# Some colours
LIGHT_RED    = '#FFC4CC'
BLUE         = '#599DEB'
LIGHT_GREEN  = '#95FD99'
GRAY         = 'C0C0C0'
BLACK        = '#000000'
WHITE        = '#FFFFFF'
LIGHT_PURPLE = '#E8D0FF'
LIGHT_ORANGE = '#FAE0C3'

class town:

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
    STEP_REWARD = 0
    GOAL_REWARD = 1
    CAUGHT_REWARD = -10
    IMPOSSIBLE_REWARD = -100


    def __init__(self, town):
        """ Constructor of the environment town.
        """
        self.town                     = town
        self.actions                  = self.__actions()
        self.police_actions           = self.__police_actions()
        self.states, self.map         = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.player_start_position    = (0,0)
        self.police_start_position    = (3,3)
        self.current_state            = self.map[(0,0,3,3)]
        
    def step(self, e_greedy=False, q_fct=None, epsilon=0.1):
        if e_greedy:
            if np.random.rand() < epsilon:
                action = np.random.choice(np.arange(0,len(self.actions)))
            else:
                action = np.argmax(q_fct[self.current_state,:])
            future_positions = self.__move(self.current_state, action)
            self.current_state = np.random.choice(future_positions)  # Police also does a random walk.
        else:
            action = np.random.choice(np.arange(0,len(self.actions)))  # Choose action uniformly at random.
            future_positions = self.__move(self.current_state, action)
            self.current_state = np.random.choice(future_positions)  # Police also does a random walk.
        return action, self.current_state
    
    def reset(self):
        self.current_state = self.map[(0,0,3,3)]
        
    def reward(self, s, action, s_next):
        if self.__check_impossible(s, action, s_next):
            return self.IMPOSSIBLE_REWARD
        elif self.__check_caught(s_next):
            return self.CAUGHT_REWARD
        elif self.town[self.states[s_next][0:2]] == 2:
            return self.GOAL_REWARD
        else:
            return self.STEP_REWARD
        
    def __check_caught(self, state):
        state_tuple = self.states[state]
        return state_tuple[0] == state_tuple[2] and state_tuple[1] == state_tuple[3]
    
    def __check_impossible(self, s, action, s_next):
        pos = tuple(map(sum, zip(self.states[s][0:2],self.actions[action])))
        return self.__wall_hit(*pos)

    def __actions(self):
        actions = dict()
        actions[self.STAY]       = (0, 0)
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions
    
    def __police_actions(self):
        actions = dict()
        actions[self.MOVE_LEFT]  = (0,-1)
        actions[self.MOVE_RIGHT] = (0, 1)
        actions[self.MOVE_UP]    = (-1,0)
        actions[self.MOVE_DOWN]  = (1,0)
        return actions

    def __states(self):
        states = dict()
        map_ = dict()
        s = 0
        for i in range(self.town.shape[0]):  # Player position
            for j in range(self.town.shape[1]):
                for k in range(self.town.shape[0]):  # Police position
                    for l in range(self.town.shape[1]):
                        states[s] = (i, j, k, l)
                        map_[(i, j, k, l)] = s
                        s += 1
        return states, map_

    def __move(self, state, action):
        """ Makes a step in the town, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            Returns a list of possible next states (x,y, police_x, police_y) that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0]
        col = self.states[state][1] + self.actions[action][1]
        # Is the future position an impossible one ?
        hitting_walls =  self.__wall_hit(row, col)
        # Compute possible future police positions.
        police_positions = self.__future_police_positions(state)
        # Based on the impossiblity check return the next state.
        if hitting_walls:
            return [self.map[(self.states[state][0], self.states[state][1], police_pos[0], police_pos[1])] 
                    for police_pos in police_positions]
        else:
            return [self.map[(row, col, police_pos[0], police_pos[1])] for police_pos in police_positions]
           
    def __future_police_positions(self, state):
        police_positions = list()
        police_pos = self.states[state][2:4]
        police_positions = [(police_pos[0] + action[0], police_pos[1] + action[1]) 
                            for action in self.police_actions.values()]
        return [pos for pos in police_positions if not self.__wall_hit(pos[0], pos[1])]
    
    def __wall_hit(self, row, col):
        return (row <= -1) or (row >= self.town.shape[0]) or (col <= -1) or (col >= self.town.shape[1])
                
    def simulate(self, policy):
        path = list()
        start = (0,0,3,3)
        t = 0
        # Initialize current state, next state, end and time
        s = self.map[start]
        # Add the starting position in the town to the path
        path.append(start)
        # Move to next state given the policy and the current state
        next_s = self.__move(s,policy[s])
        # __move returns all possible next states, choose next state uniformly at random.
        next_s = np.random.choice(next_s, 1)[0]
        # Add the position in the town corresponding to the next state
        # to the path
        path.append(self.states[next_s])
        # Loop while state is not the goal state
        while t < 100:
            # Update state
            s = next_s
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s])
            # __move returns all possible next states, choose next state uniformly at random.
            next_s = np.random.choice(next_s, 1)[0]
            # Add the position in the town corresponding to the next state
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


def q_learning(env, gamma, verbose=True):
    q_fct = np.zeros((env.n_states, env.n_actions))
    sample_count = np.zeros((env.n_states, env.n_actions))
    v_history = list()
    env.reset()
    for idx in range(int(1e7)):
        if idx%10000 == 0:
            v_history.append(q_fct[15,:].copy())  # 15 is start state
            if verbose:
                print(f"Running iteration {idx}.")
        s = env.current_state
        action, s_next = env.step()
        reward = env.reward(s, action, s_next)
        sample_count[s, action] += 1
        alpha = 1/sample_count[s,action]**(2/3)
        q_fct[s,action] += alpha * (reward + gamma*np.max(q_fct[s_next,:] - q_fct[s, action]))
    policy = np.argmax(q_fct,1)
    return q_fct, policy, v_history

def sarsa(env, epsilon, gamma, verbose=True):
    q_fct = np.zeros((env.n_states, env.n_actions))
    sample_count = np.zeros((env.n_states, env.n_actions))
    v_history = list()
    env.reset()
    
    s = env.current_state
    action, s_next = env.step(e_greedy=True, q_fct=q_fct, epsilon=epsilon)
    
    for idx in range(int(1e6)):
        # Making sure the function is running.
        if idx%10000 == 0:
            v_history.append(q_fct[15,:].copy())  # 15 is start state
            if verbose:
                print(f"Running iteration {idx}.")
        # States lag behind by 1 to get a full (s, a, r, s, a) sample.
        s_next = env.current_state
        action_next, _ = env.step(e_greedy=True, q_fct=q_fct, epsilon=epsilon)
        reward = env.reward(s, action, s_next)
        sample_count[s, action] += 1
        alpha = 1/sample_count[s,action]**(2/3)
        q_fct[s,action] += alpha * (reward + gamma*q_fct[s_next,action_next] - q_fct[s, action])
        s = s_next
        action = action_next
    policy = np.argmax(q_fct,1)
    return q_fct, policy, v_history


def draw_town(town):
    town = town.copy()
    town[0,0] = 1
    town[3,3] = 3
    # Map a color to each cell in the town
    col_map = {0: WHITE, 1: LIGHT_ORANGE, 2: LIGHT_GREEN, 3: BLUE}

    # Give a color to each cell
    rows,cols    = town.shape
    colored_town = [[col_map[town[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the town
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The town')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = town.shape
    colored_town = [[col_map[town[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the town
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None, cellColours=colored_town, cellLoc='center', loc=(0,0), edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
    plt.show()


def animate_solution(town, path):

    # Map a color to each cell in the town
    col_map = {0: WHITE, 2: LIGHT_GREEN}

    # Size of the town
    rows,cols = town.shape

    # Create figure of the size of the town
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_town = [[col_map[town[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the town
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None, cellColours=colored_town, cellLoc='center', loc=(0,0), edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    # Update the color at each frame
    for i in range(len(path)):
        player_pos = path[i][0:2]
        police_pos = path[i][2:4]
        # New markings
        grid.get_celld()[(player_pos)].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(player_pos)].get_text().set_text('Player')
        grid.get_celld()[(police_pos)].set_facecolor(BLUE)
        grid.get_celld()[(police_pos)].get_text().set_text('Police')
        if i > 0:
            # Reset old markings if not marked by new.
            if not player_pos == path[i-1][0:2] and not police_pos == path[i-1][0:2]:
                grid.get_celld()[(path[i-1][0:2])].set_facecolor(col_map[town[path[i-1][0:2]]])
                grid.get_celld()[(path[i-1][0:2])].get_text().set_text('')
            if not player_pos == path[i-1][2:4] and not police_pos == path[i-1][2:4]:
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[town[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')
            if town[player_pos] == 2:
                grid.get_celld()[(player_pos)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(player_pos)].get_text().set_text('Player is robbing')
            elif player_pos == police_pos:
                grid.get_celld()[(player_pos)].set_facecolor(GRAY)
                grid.get_celld()[(player_pos)].get_text().set_text('Player caught')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.3)
