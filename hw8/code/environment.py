import numpy as np
import sys
from value_iteration import create_states_actions

class Environment():
    def __init__(self, filename):
        transitions, state_graph, num_states, num_actions, start_state, goal_state = create_states_actions(filename)

        self.transitions = transitions
        self.state_graph = state_graph
        self.start_state = start_state
        self.goal_state = goal_state
        self.num_states = num_states
        self.num_actions = num_actions
        # set curr state to initial stage
        self.curr_state = start_state

    
    def step(self, action):
        reward, next_state, _, is_terminal = self.transitions[self.curr_state][action]
        self.curr_state = next_state
        return next_state, reward, is_terminal

    def reset(self):
        self.curr_state = self.start_state
        return self.curr_state

if __name__ == "__main__":
    maze_input = sys.argv[1]
    output_file = sys.argv[2]
    action_sequence = sys.argv[3]

    actions = []
    with open(action_sequence, 'r') as infile:
        actions = infile.readlines()

    actions = [i.strip() for i in actions]
    actions = [i.split(' ') for i in actions]
    actions = [j for sub in actions for j in sub]
    actions = list(map(int, actions))

    env = Environment(maze_input)

    state_string = ""
    reward_string = ""
    terminal_string = ""
    for i in range(0, len(actions)):
        next_state, reward, is_terminal = env.step(actions[i])
        reward = int(reward)
        state_string += str(next_state) + " "
        reward_string += str(reward) + " "
        terminal_string += str(is_terminal) + " "

    state_string = state_string.strip()
    state_string = state_string.split(' ')
    
    reward_string = reward_string.strip()
    reward_string = reward_string.split(' ')

    terminal_string = terminal_string.strip()
    terminal_string = terminal_string.split(' ')

    actual_file_string = ""
    for i in range(0, len(state_string)):
        curr0, curr1 = env.state_graph[state_string[i]]
        actual_file_string += str(curr0) + " " + str(curr1) + " " + reward_string[i] + " " + terminal_string[i] + "\n"

    actual_file_string = actual_file_string.strip()
    
    with open(output_file, 'w') as outfile:
        outfile.writelines(actual_file_string)