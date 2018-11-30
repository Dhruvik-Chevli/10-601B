import numpy as np
from environment import Environment
import sys

def get_epsilon_greedy_policy(Q, epsilon, num_actions):
    
    def policyFunction(observation):
        A = np.ones(num_actions, dtype=float) * epsilon / num_actions
        best_action = np.argmax(Q[observation])
        A[best_action] += (1.0 - epsilon)
        return A
    return policyFunction

def q_learning(env, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon):
    Q = np.zeros((env.num_states, env.num_actions))

    policy = get_epsilon_greedy_policy(Q, epsilon, env.num_actions)

    for _ in range(0, num_episodes):
        state = env.reset()

        for _ in range(0, max_episode_length):
            action_probs = policy(state)
            action = np.random.choice(np.arange(len(action_probs)), p=action_probs)
            next_state, reward, is_terminal = env.step(action)

            best_next_action = np.argmax(Q[next_state])
            td_target = reward + discount_factor * Q[next_state][best_next_action]
            Q[state][action] += learning_rate * (td_target - Q[state][action])

            if is_terminal == 1:
                break

            state = env.curr_state

    V = np.zeros(env.num_states)
    for s in range(0, env.num_states):
        V[s] = np.max(Q[s])

    policy = np.zeros((env.num_states, env.num_actions))
    for s in range(0, env.num_states):
        best_action = np.argmax(Q[s])
        policy[s][best_action] = 1.0

    return policy, V, Q

if __name__ == "__main__":
    maze_input = sys.argv[1]
    value_file = sys.argv[2]
    q_value_file = sys.argv[3]
    policy_file = sys.argv[4]
    num_episodes = int(sys.argv[5])
    max_episode_length = int(sys.argv[6])
    learning_rate = float(sys.argv[7])
    discount_factor = float(sys.argv[8])
    epsilon = float(sys.argv[9])

    env = Environment(maze_input)
    policy, V, Q = q_learning(env, num_episodes, max_episode_length, learning_rate, discount_factor, epsilon)

    new_policy = np.array(np.argmax(policy, axis=1), np.float32)

    value_string = ""
    policy_string = ""
    Q_string = ""
    for k, v in env.state_graph.items():
        k = int(k)
        value_string += str(v[0]) + " " + str(v[1]) + " " + str(V[k]) + "\n"
        policy_string += str(v[0]) + " " + str(v[1]) + " " + str(new_policy[k]) + "\n"

        for a in range(0, env.num_actions):
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