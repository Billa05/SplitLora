GRID_SIZE = 4
ACTIONS = ['U', 'D', 'L', 'R']

GAMMA = 1.0
THETA = 0.0001

START_STATE = (0, 0)
TERMINAL_STATE = (3, 3)

# Obstacles
OBSTACLES = [(1, 1), (2, 2)]

ACTION_EFFECT = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

ARROW = {
    'U': '↑',
    'D': '↓',
    'L': '←',
    'R': '→'
}

# ---------------- VALUE FUNCTION INITIALIZATION ----------------
V = [[0.0 for _ in range(GRID_SIZE)] for _ in range(GRID_SIZE)]

# ---------------- HELPER FUNCTIONS ----------------
def is_terminal(state):
    return state == TERMINAL_STATE


def is_obstacle(state):
    return state in OBSTACLES


def next_state(state, action):
    x, y = state
    dx, dy = ACTION_EFFECT[action]

    nx, ny = x + dx, y + dy

    # Boundary check
    if nx < 0 or nx >= GRID_SIZE or ny < 0 or ny >= GRID_SIZE:
        return state

    # Obstacle check
    if (nx, ny) in OBSTACLES:
        return state

    return (nx, ny)

# ---------------- VALUE ITERATION ----------------
iteration = 0

while True:
    delta = 0
    iteration += 1

    print(f"\n=========== ITERATION {iteration} ===========")

    # -------- VALUE UPDATE --------
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = (i, j)

            if is_terminal(state) or is_obstacle(state):
                continue

            old_value = V[i][j]
            action_values = []

            for action in ACTIONS:
                ns = next_state(state, action)
                reward = -1
                action_values.append(
                    reward + GAMMA * V[ns[0]][ns[1]]
                )

            V[i][j] = max(action_values)
            delta = max(delta, abs(old_value - V[i][j]))

    # -------- PRINT VALUES --------
    print("\nState Values:")
    for i in range(GRID_SIZE):
        row = []
        for j in range(GRID_SIZE):
            if (i, j) in OBSTACLES:
                row.append("  X   ")
            else:
                row.append("{:6.2f}".format(V[i][j]))
        print(row)

    # -------- PRINT POLICY --------
    print("\nPolicy (arrows):")
    for i in range(GRID_SIZE):
        row_policy = []
        for j in range(GRID_SIZE):
            state = (i, j)

            if state == START_STATE:
                row_policy.append("  S  ")
                continue

            if is_terminal(state):
                row_policy.append("  T  ")
                continue

            if is_obstacle(state):
                row_policy.append("  X  ")
                continue

            best_action = None
            best_value = float('-inf')

            for action in ACTIONS:
                ns = next_state(state, action)
                reward = -1
                value = reward + GAMMA * V[ns[0]][ns[1]]

                if value > best_value:
                    best_value = value
                    best_action = action

            row_policy.append(f"  {ARROW[best_action]}  ")

        print(row_policy)

    print("Delta:", delta)

    if delta < THETA:
        break
