import numpy as np
import sys
import time

def create_states_actions(filename):
    lines = dict()
    with open(filename, 'r') as infile:
        lines = infile.readlines()

    lines = [i.strip() for i in lines]
    lines = np.array(lines)

    state = 0
    total_states = 0

    start_state = 0
    goal_state = 0

    state_graph = dict()
    for i in range(0, len(lines)):
        chars = list(lines[i])
        for j in range(0, len(chars)):
            if chars[j] == '*':
                continue
            state_graph[str(state)] = (i,j)
            if chars[j] == 'S':
                start_state = state
            if chars[j] == 'G':
                goal_state = state
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
                    transitions[state][a] = (0.0, state, 1, 0)
                if j == len(lines[0])-1 and a == 2:
                    transitions[state][a] = (0.0, state, 1, 0)
                if i == 0 and a == 1:
                    transitions[state][a] = (0.0, state, 1, 0)
                if i == len(lines)-1 and a == 3:
                    transitions[state][a] = (0.0, state, 1, 0)
                
                if a == 0 and j != 0:
                    if lines[i][j-1] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i,j-1))])
                        is_terminal = 1 if next_state == goal_state else 0
                        reward = 10 if next_state == goal_state else 0
                        transitions[state][a] = (reward, next_state, 1, is_terminal)
                    else:
                        transitions[state][a] = (0.0, state, 1, 0)

                if a == 2 and j != len(lines[0])-1:
                    if lines[i][j+1] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i,j+1))])
                        is_terminal = 1 if next_state == goal_state else 0
                        reward = 10 if next_state == goal_state else 0
                        transitions[state][a] = (reward, next_state, 1, is_terminal)
                    else:
                        transitions[state][a] = (0.0, state, 1, 0)

                if a == 1 and i != 0:
                    if lines[i-1][j] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i-1,j))])
                        is_terminal = 1 if next_state == goal_state else 0
                        reward = 10 if next_state == goal_state else 0
                        transitions[state][a] = (reward, next_state, 1, is_terminal)
                    else:
                        transitions[state][a] = (0.0, state, 1, 0)

                if a == 3 and i != len(lines)-1:
                    if lines[i+1][j] != '*':
                        next_state = int(list(state_graph.keys())[list(state_graph.values()).index((i+1,j))])
                        is_terminal = 1 if next_state == goal_state else 0
                        reward = 10 if next_state == goal_state else 0
                        transitions[state][a] = (reward, next_state, 1, is_terminal)
                    else:
                        transitions[state][a] = (0.0, state, 1, 0)

            if chars[j] == 'G':
                for a in range(num_actions):
                    transitions[state][a] = (0, state, 1, 1)
    
    return transitions, state_graph, num_states, num_actions, start_state, goal_state

def valueiter(num_states, num_actions, num_epochs, transition, discount_factor):

    def one_step_lookahead(state, V):
        A = np.zeros(num_actions)
        for a in range(0, num_actions):
            reward, next_state, prob, _ = transition[state][a]
            A[a] += prob * (reward  + discount_factor*V[next_state])
        return A

    V = np.zeros(num_states)
    start = time.time()
    itr = 0
    threshold = 0.001
    print ("Printing iterations and values here")
    while True:
        print (itr, V)
        delta = 0
        for s in range(0, num_states):
            A = one_step_lookahead(s,V)
            best_action_value = np.max(A)
            delta = max(delta, np.abs(best_action_value - V[s]))
            V[s] = best_action_value
        itr += 1
        if delta < threshold:
            break

    end = time.time() - start
    # print ("time taken: ", end)
    # print ("iterations", itr)

    policy = np.zeros((num_states, num_actions))
    for s in range(0, num_states):
        A = one_step_lookahead(s, V)
        print ("A for state after iteration is: ", A)
        best_action = np.argmax(A)
        policy[s][best_action] = 1.0

    Q = np.zeros((num_states, num_actions))
    for s in range(0, num_states):
            A = one_step_lookahead(s,V)
            Q[s] = A
    
    return policy, V, Q

if __name__ == "__main__":
    maze_input = sys.argv[1]
    value_file = sys.argv[2]
    q_value_file = sys.argv[3]
    policy_file = sys.argv[4]
    num_epochs = int(sys.argv[5])
    discount_factor = float(sys.argv[6])

    transitions, state_graph, num_states, num_actions, start_state, goal_state = create_states_actions(maze_input)
    policy, V, Q = valueiter(num_states, num_actions, num_epochs, transitions, discount_factor)
    print (policy)

    new_policy = np.array(np.argmax(policy, axis=1), np.float32)

    value_string = ""
    policy_string = ""
    Q_string = ""
    for k, v in state_graph.items():
        k = int(k)
        value_string += str(v[0]) + " " + str(v[1]) + " " + str(np.round(V[k], decimals=2)) + "\n"
        policy_string += str(v[0]) + " " + str(v[1]) + " " + str(new_policy[k]) + "\n"

        for a in range(0, num_actions):
            Q_string += str(v[0]) + " " + str(v[1]) + " " + str(a) + " " + str(Q[k][a]) + "\n"

    print ("value string")
    print (value_string)
    print ("policy string")
    print (policy_string)
    # print ("Q string: ", Q_string)

    # value_string = value_string.strip()
    # policy_string = policy_string.strip()
    # Q_string = Q_string.strip()

    # with open(value_file, 'w') as outfile:
    #     outfile.writelines(value_string)

    # with open(policy_file, 'w') as outfile:
    #     outfile.writelines(policy_string)

    # with open(q_value_file, 'w') as outfile:
    #     outfile.writelines(Q_string)