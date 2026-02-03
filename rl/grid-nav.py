import random

# ---------------- CONFIGURATION ----------------
GRID_SIZE = 5

START_STATE = (0, 0)
GOAL_STATE = (4, 4)
PIT_STATES = [(1, 1), (2, 2), (3, 3)]

ACTIONS = ['up', 'down', 'left', 'right']

GOAL_REWARD = 10
PIT_REWARD = -10
STEP_REWARD = -1

alpha = 0.1
gamma = 0.9

epsilon = 1.0
epsilon_min = 0.01
epsilon_decay = 0.995

episodes = 1000

# ---------------- Q-TABLE INITIALIZATION ----------------
Q = [
    [
        [0.0 for _ in range(len(ACTIONS))]
        for _ in range(GRID_SIZE)
    ]
    for _ in range(GRID_SIZE)
]

# ---------------- HELPER FUNCTIONS ----------------
def is_terminal(state):
    return state == GOAL_STATE or state in PIT_STATES


def get_reward(state):
    if state == GOAL_STATE:
        return GOAL_REWARD
    if state in PIT_STATES:
        return PIT_REWARD
    return STEP_REWARD


def next_state(state, action):
    r, c = state

    if action == 'up':
        r = max(r - 1, 0)
    elif action == 'down':
        r = min(r + 1, GRID_SIZE - 1)
    elif action == 'left':
        c = max(c - 1, 0)
    elif action == 'right':
        c = min(c + 1, GRID_SIZE - 1)

    return (r, c)


def argmax(values):
    max_index = 0
    max_value = values[0]

    for i in range(1, len(values)):
        if values[i] > max_value:
            max_value = values[i]
            max_index = i

    return max_index


def max_value(values):
    m = values[0]
    for v in values:
        if v > m:
            m = v
    return m


def choose_action(state):
    if random.uniform(0, 1) < epsilon:
        return random.randint(0, len(ACTIONS) - 1)
    return argmax(Q[state[0]][state[1]])

# ---------------- TRAINING LOOP ----------------
for ep in range(episodes):
    state = START_STATE

    while not is_terminal(state):
        action_idx = choose_action(state)
        action = ACTIONS[action_idx]

        next_s = next_state(state, action)
        reward = get_reward(next_s)

        # Q-learning update rule
        Q[state[0]][state[1]][action_idx] += alpha * (
            reward
            + gamma * max_value(Q[next_s[0]][next_s[1]])
            - Q[state[0]][state[1]][action_idx]
        )

        state = next_s

    epsilon = max(epsilon_min, epsilon * epsilon_decay)

# ---------------- PRINT Q-TABLE ----------------
print("\nLearned Q-table:\n")

for r in range(GRID_SIZE):
    for c in range(GRID_SIZE):
        print(f"State {(r, c)} -> {Q[r][c]}")

# ---------------- PRINT POLICY ----------------
policy_symbols = {
    0: '↑',
    1: '↓',
    2: '←',
    3: '→'
}

print("\nLearned Policy Grid:\n")

for r in range(GRID_SIZE):
    row_policy = ""
    for c in range(GRID_SIZE):
        state = (r, c)

        if state == GOAL_STATE:
            row_policy += " G "
        elif state in PIT_STATES:
            row_policy += " P "
        else:
            best_action = argmax(Q[r][c])
            row_policy += f" {policy_symbols[best_action]} "

    print(row_policy)
