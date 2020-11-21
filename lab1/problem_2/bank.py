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

class Bank:

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
    GOAL_REWARD = 10
    CAUGHT_REWARD = -50
    IMPOSSIBLE_REWARD = -100


    def __init__(self, bank):
        """ Constructor of the environment bank.
        """
        self.bank                     = bank
        self.actions                  = self.__actions()
        self.police_actions           = self.__police_actions()
        self.states, self.map    = self.__states()
        self.n_actions                = len(self.actions)
        self.n_states                 = len(self.states)
        self.transition_probabilities = self.__transitions()
        self.rewards                  = self.__rewards()
        self.player_start_position    = (0,0)
        self.police_start_position    = (1,2)

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
        for i in range(self.bank.shape[0]):  # Player position
            for j in range(self.bank.shape[1]):
                for k in range(self.bank.shape[0]):  # Police position
                    for l in range(self.bank.shape[1]):
                        states[s] = (i, j, k, l)
                        map_[(i, j, k, l)] = s
                        s += 1
        return states, map_

    def __move(self, state, action):
        """ Makes a step in the bank, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place. If the agent was caught, 
            its position is reset.

            Returns a list of possible next states (x,y, police_x, police_y) that agent transitions to.
        """
        # Is the agent caught? If so, police stays at agent.
        if self.__check_caught(state):
            return [self.map[(0,0,1,2)]]
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
        
    def __check_caught(self, state):
        state_tuple = self.states[state]
        return state_tuple[0] == state_tuple[2] and state_tuple[1] == state_tuple[3]
    
    def __future_police_positions(self, state):
        police_positions = list()
        player_pos = self.states[state][0:2]
        police_pos = self.states[state][2:4]
        if player_pos[0] == police_pos[0]:
            police_positions.append((police_pos[0]+1,police_pos[1]))
            police_positions.append((police_pos[0]-1,police_pos[1]))
        elif player_pos[0] > police_pos[0]:
            police_positions.append((police_pos[0]+1,police_pos[1]))
        else:
            police_positions.append((police_pos[0]-1,police_pos[1]))
        if player_pos[1] == police_pos[1]:
            police_positions.append((police_pos[0],police_pos[1]+1))
            police_positions.append((police_pos[0],police_pos[1]-1))
        elif player_pos[1] > police_pos[1]:
            police_positions.append((police_pos[0],police_pos[1]+1))
        else:
            police_positions.append((police_pos[0],police_pos[1]-1))
        return [pos for pos in police_positions if not self.__wall_hit(pos[0], pos[1])]
    
    def __wall_hit(self, row, col):
        return (row <= -1) or (row >= self.bank.shape[0]) or (col <= -1) or (col >= self.bank.shape[1])


    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions)
        transition_probabilities = np.zeros(dimensions)

        # Compute the transition probabilities.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.__move(s,a)
                for next_s in next_states:
                    transition_probabilities[next_s, s, a] = 1/len(next_states)  # police walks random.
        return transition_probabilities

    def __rewards(self):
        rewards = np.zeros((self.n_states, self.n_actions))
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_states = self.__move(s,a)
                # Compute the average reward for action (s,a).
                reward = 0
                for next_s in next_states:
                    if len(next_states) == 1:  # Caught by the police.
                        reward += self.CAUGHT_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for hitting a wall and not being caught.
                    elif self.states[s][0:2] == self.states[next_s][0:2] and self.actions_names[a] != "stay":  # TODO: CHECK
                        reward += self.IMPOSSIBLE_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for reaching a bank.
                    elif self.bank[self.states[s][0:2]] == 2:
                        reward += self.GOAL_REWARD * self.transition_probabilities[next_s, s, a]
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        reward += self.STEP_REWARD * self.transition_probabilities[next_s, s, a]
                rewards[s,a] = reward
        return rewards
                
    def simulate(self, policy):
        path = list()
        start = (0,0,1,2)
        t = 0
        # Initialize current state, next state, end and time
        s = self.map[start]
        # Add the starting position in the bank to the path
        path.append(start)
        # Move to next state given the policy and the current state
        next_s = self.__move(s,policy[s])
        # __move returns all possible next states, choose next state uniformly at random.
        next_s = np.random.choice(next_s, 1)[0]
        # Add the position in the bank corresponding to the next state
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
            # Add the position in the bank corresponding to the next state
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


def value_iteration(env, gamma, epsilon):
    """ Solves the shortest path problem using value iteration
        :input bank env           : The bank environment in which we seek to
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

    # Compute policy
    policy = np.argmax(Q,1)
    # Return the obtained policy
    return V, policy

def draw_bank(bank):

    # Map a color to each cell in the bank
    col_map = {0: WHITE, 2: LIGHT_GREEN}

    # Give a color to each cell
    rows,cols    = bank.shape
    colored_bank = [[col_map[bank[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the bank
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('The bank')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    rows,cols    = bank.shape
    colored_bank = [[col_map[bank[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the bank
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_bank,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed')
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)
        
def illustrate_policy(env, policy, gamma=None):
    """!@brief Visualizes the moves of a given policy.
    
    Police always fixed at (1,0). Moves are shown as arrows. 
    """
    LIGHT_GREEN  = '#95FD99'
    WHITE        = '#FFFFFF'
    BLUE         = '#599DEB'

    col_map = {0: WHITE, 2: LIGHT_GREEN}

    # Size of the maze
    rows,cols = env.bank.shape

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    title_string = f" with gamma = {gamma}" if gamma is not None else ''
    ax.set_title('Policy simulation at time step 0' + title_string)
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_maze = [[col_map[env.bank[j,i]] for i in range(cols)] for j in range(rows)]
    
    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))
    
    # Create a table to color
    grid = plt.table(cellText=None, cellColours=colored_maze, cellLoc='center',loc=(0,0),edges='closed')

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows)
        cell.set_width(1.0/cols)

    police_pos = (1,0)
    grid.get_celld()[(police_pos)].set_facecolor(BLUE)
    grid.get_celld()[(police_pos)].get_text().set_text('Police')
    for x in range(3):
        for y in range(6):
            if (x,y) == police_pos:
                continue
            a = policy[env.map[(x,y,*police_pos)]]
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


def animate_solution(bank, path):

    # Map a color to each cell in the bank
    col_map = {0: WHITE, 2: LIGHT_GREEN}

    # Size of the bank
    rows,cols = bank.shape

    # Create figure of the size of the bank
    fig = plt.figure(1, figsize=(cols,rows))

    # Remove the axis ticks and add title title
    ax = plt.gca()
    ax.set_title('Policy simulation')
    ax.set_xticks([])
    ax.set_yticks([])

    # Give a color to each cell
    colored_bank = [[col_map[bank[j,i]] for i in range(cols)] for j in range(rows)]

    # Create figure of the size of the bank
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None, cellColours=colored_bank, cellLoc='center', loc=(0,0), edges='closed')

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
                grid.get_celld()[(path[i-1][0:2])].set_facecolor(col_map[bank[path[i-1][0:2]]])
                grid.get_celld()[(path[i-1][0:2])].get_text().set_text('')
            if not player_pos == path[i-1][2:4] and not police_pos == path[i-1][2:4]:
                grid.get_celld()[(path[i-1][2:4])].set_facecolor(col_map[bank[path[i-1][2:4]]])
                grid.get_celld()[(path[i-1][2:4])].get_text().set_text('')
            if bank[player_pos] == 2:
                grid.get_celld()[(player_pos)].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(player_pos)].get_text().set_text('Player is robbing')
            elif player_pos == police_pos:
                grid.get_celld()[(player_pos)].set_facecolor(GRAY)
                grid.get_celld()[(player_pos)].get_text().set_text('Player caught')
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.7)
