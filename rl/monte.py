import random

# ---------------- STATE DEFINITIONS ----------------
state_names = {
    0: "TL",
    1: "A",
    2: "B",
    3: "C",
    4: "TR"
}

NUM_STATES = 5
TL, TR = 0, 4
NON_TERMINAL_STATES = [1, 2, 3]

# ---------------- PARAMETERS ----------------
gamma = 1.0
episodes_per_start = 10

# ---------------- VALUE INITIALIZATION ----------------
V = [0.5] * NUM_STATES
V[TL] = 0.0
V[TR] = 1.0

# For Monte Carlo averaging
returns_sum = [0.0] * NUM_STATES
returns_count = [0] * NUM_STATES

# ---------------- ENVIRONMENT STEP ----------------
def step(state):
    action = random.choice([-1, 1])   # left or right
    next_state = state + action
    reward = 1 if next_state == TR else 0
    return next_state, reward

# ---------------- PRINT UTILITIES ----------------
def print_header():
    print("Episode End Steps      TL        A        B        C       TR")
    print("------ --- ----- -------- -------- -------- -------- --------")


def print_row(ep, end_state, steps, V):
    print(
        f"{ep:>6} {state_names[end_state]:>3} {steps:>5} "
        f"{V[0]:>8.4f} {V[1]:>8.4f} {V[2]:>8.4f} {V[3]:>8.4f} {V[4]:>8.4f}"
    )

# ---------------- MONTE CARLO TRAINING ----------------
for start_state in NON_TERMINAL_STATES:
    print("\n" + "=" * 70)
    print(
        f"Monte Carlo training with fixed start state: "
        f"{state_names[start_state]} ({start_state})"
    )
    print("=" * 70)

    print_header()

    for ep in range(1, episodes_per_start + 1):
        state = start_state
        episode = []
        steps = 0

        # -------- GENERATE EPISODE --------
        while state not in [TL, TR]:
            steps += 1
            next_state, reward = step(state)
            episode.append((state, reward))
            state = next_state

        # -------- MONTE CARLO RETURN --------
        G = 0
        for state_t, reward_t in reversed(episode):
            G = gamma * G + reward_t

            returns_sum[state_t] += G
            returns_count[state_t] += 1
            V[state_t] = returns_sum[state_t] / returns_count[state_t]

        print_row(ep, state, steps, V)

# ---------------- FINAL VALUES ----------------
print("\n" + "=" * 70)
print("Final learned values (Monte Carlo)")
print("=" * 70)
print(
    f"TL={V[0]:.4f}, "
    f"A={V[1]:.4f}, "
    f"B={V[2]:.4f}, "
    f"C={V[3]:.4f}, "
    f"TR={V[4]:.4f}"
)
