import random

# ---------------- CONFIGURATION ----------------
ROWS = 3
COLS = 3

alpha = 0.1
gamma = 0.9
epsilon = 0.2
episodes = 1000

ACTIONS = [0, 1, 2, 3]
ACTION_NAMES = ["UP", "DOWN", "LEFT", "RIGHT"]
ARROWS = ["↑", "↓", "←", "→"]

START = (0, 0)
GOAL = (2, 2)
OBSTACLE = (1, 1)

# ---------------- Q-TABLE INITIALIZATION ----------------
Q = {}
for i in range(ROWS):
    for j in range(COLS):
        Q[(i, j)] = [0.0, 0.0, 0.0, 0.0]

# ---------------- ENVIRONMENT STEP FUNCTION ----------------
def step(state, action):
    i, j = state
    ni, nj = i, j

    if action == 0 and i > 0:              # UP
        ni -= 1
    elif action == 1 and i < ROWS - 1:     # DOWN
        ni += 1
    elif action == 2 and j > 0:            # LEFT
        nj -= 1
    elif action == 3 and j < COLS - 1:     # RIGHT
        nj += 1

    next_state = (ni, nj)

    if next_state == OBSTACLE:
        return state, -5
    if next_state == GOAL:
        return next_state, 10

    return next_state, -1

# ---------------- ACTION SELECTION (ε-GREEDY) ----------------
def choose_action(state):
    if random.random() < epsilon:
        return random.choice(ACTIONS)
    return Q[state].index(max(Q[state]))

# ---------------- TRAINING LOOP ----------------
for ep in range(episodes):
    state = START

    while state != GOAL:
        action = choose_action(state)
        next_state, reward = step(state, action)

        Q[state][action] += alpha * (
            reward + gamma * max(Q[next_state]) - Q[state][action]
        )

        state = next_state

# ---------------- PRINT Q-TABLE ----------------
print("\nFINAL Q-TABLE VALUES\n")

for i in range(ROWS):
    for j in range(COLS):
        if (i, j) == OBSTACLE:
            print(f"State {(i, j)} : OBSTACLE\n")
            continue

        print(f"State {(i, j)}:")
        for a in range(4):
            print(f"  {ACTION_NAMES[a]} : {Q[(i, j)][a]:6.2f}")
        print()

# ---------------- PRINT OPTIMAL POLICY ----------------
print("\nOPTIMAL POLICY (ARROWS SHOW BEST ACTION)\n")

for i in range(ROWS):
    for j in range(COLS):
        if (i, j) == START:
            print(" S ", end="")
        elif (i, j) == GOAL:
            print(" G ", end="")
        elif (i, j) == OBSTACLE:
            print(" X ", end="")
        else:
            best_action = Q[(i, j)].index(max(Q[(i, j)]))
            print(f" {ARROWS[best_action]} ", end="")
    print()
