import numpy as np
import sys

def create_states_actions(filename):
    lines = dict()
    with open(filename, 'r') as infile:
        lines = infile.readlines()

    lines = [i.strip() for i in lines]
    lines = np.array(lines)

    state = 0
    total_states = 0
    state_graph = dict()
    for i in range(0, len(lines)):
        chars = list(lines[i])
        for j in range(0, len(chars)):
            if chars[j] == '*':
                continue
            state_graph[str(state)] = (i,j)
            state += 1
            total_states += 1

    num_states = total_states
    num_actions = 4

    transitions = [[tuple() for j in range(0, num_actions)] for i in range(0, num_states)]
    
    for i in range(0, len(lines)):
        chars = list(lines[i])
        for j in range(0, len(chars)):
            if chars[j] == '*':
                continue
            state = int(list(state_graph.keys())[list(state_graph.values()).index((i,j))])
            for a in range(num_actions):
                if j == 0 and a == 0:
                    transitions[state][a] = (-1.0, state, 1)
                if j == len(lines[0])-1 and a == 2:
                    transitions[state][a] = (-1.0, state, 1)
                if i == 0 and a == 1:
                    transitions[state][a] = (-1.0, state, 1)
                if i == len(lines)-1 and a == 3:
                    transitions[state][a] = (-1.0, state, 1)
                
                if a == 0 and j != 0:
                    if lines[i][j-1] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i,j-1))])
                        transitions[state][a] = (-1.0, next_state, 1)
                    else:
                        transitions[state][a] = (-1.0, state, 1)

                if a == 2 and j != len(lines[0])-1:
                    # print (lines, lines[i][j], i,j, len(lines[0])-1)
                    if lines[i][j+1] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i,j+1))])
                        transitions[state][a] = (-1.0, next_state, 1)
                    else:
                        transitions[state][a] = (-1.0, state, 1)

                if a == 1 and i != 0:
                    if lines[i-1][j] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i-1,j))])
                        transitions[state][a] = (-1.0, next_state, 1)
                    else:
                        transitions[state][a] = (-1.0, state, 1)

                if a == 3 and i != len(lines)-1:
                    if lines[i+1][j] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i+1,j))])
                        transitions[state][a] = (-1.0, next_state, 1)
                    else:
                        transitions[state][a] = (-1.0, state, 1)

            if chars[j] == 'G':
                for a in range(num_actions):
                    transitions[state][a] = (0, state, 1)
    
    return transitions, state_graph, num_states, num_actions

def valueiter(num_states, num_actions, num_epochs, transition, discount_factor):

    def one_step_lookahead(state, V):
        A = np.zeros(num_actions)
        for a in range(0, num_actions):
            reward, next_state, prob = transition[state][a]
            A[a] += prob * (reward  + discount_factor*V[next_state])
        return A

    V = np.zeros(num_states)
    for _ in range(0, num_epochs):
        for s in range(0, num_states):
            A = one_step_lookahead(s,V)
            V[s] = np.max(A)

    policy = np.zeros((num_states, num_actions))
    for s in range(0, num_states):
        A = one_step_lookahead(s, V)
        best_action = np.argmax(A)
        policy[s][best_action] = 1.0

    return policy, V

def computeQ(num_states, num_actions, transition, discount_factor, V):
    Q = np.zeros((num_states, num_actions))

    def one_step_lookahead(state, V):
        A = np.zeros(num_actions)
        for a in range(0, num_actions):
            reward, next_state, prob = transition[state][a]
            A[a] += prob * (reward  + discount_factor*V[next_state])
        return A

    for s in range(0, num_states):
            A = one_step_lookahead(s,V)
            Q[s] = A

    return Q

if __name__ == "__main__":
    maze_input = sys.argv[1]
    value_file = sys.argv[2]
    q_value_file = sys.argv[3]
    policy_file = sys.argv[4]
    num_epochs = int(sys.argv[5])
    discount_factor = float(sys.argv[6])

    transitions, state_graph, num_states, num_actions = create_states_actions(maze_input)
    policy, V = valueiter(num_states, num_actions, num_epochs, transitions, discount_factor)
    new_policy = np.array(np.argmax(policy, axis=1), np.float32)
    Q = computeQ(num_states, num_actions, transitions, discount_factor, V)

    value_string = ""
    policy_string = ""
    Q_string = ""
    for k, v in state_graph.items():
        k = int(k)
        value_string += str(k) + " " + str(v[0]) + " " + str(v[1]) + " " + str(V[k]) + "\n"
        policy_string += str(v[0]) + " " + str(v[1]) + " " + str(new_policy[k]) + "\n"

        for a in range(0, num_actions):
            Q_string += str(v[0]) + " " + str(v[1]) + " " + str(a) + " " + str(Q[k][a]) + "\n"

    value_string = value_string.strip()
    policy_string = policy_string.strip()
    Q_string = Q_string.strip()

    with open(value_file, 'w') as outfile:
        outfile.writelines(value_string)

    with open(policy_file, 'w') as outfile:
        outfile.writelines(policy_string)

    with open(q_value_file, 'w') as outfile:
        outfile.writelines(Q_string)