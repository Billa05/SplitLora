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

# ---------------- PARAMETERS ----------------
alpha = 0.1
gamma = 1.0
episodes_per_start = 10

# ---------------- VALUE INITIALIZATION ----------------
V = [0.5] * NUM_STATES
V[TL] = 0.0
V[TR] = 1.0

# ---------------- ENVIRONMENT STEP ----------------
def step(state):
    action = random.choice([-1, 1])
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

# ---------------- TRAINING ----------------
for start_state in [1, 2, 3]:
    print("\n" + "=" * 70)
    print(
        f"Training with fixed start state: "
        f"{state_names[start_state]} ({start_state})"
    )
    print("=" * 70)

    print_header()

    for ep in range(1, episodes_per_start + 1):
        state = start_state
        steps = 0

        while state not in [TL, TR]:
            steps += 1
            next_state, reward = step(state)

            V[state] = V[state] + alpha * (
                reward + gamma * V[next_state] - V[state]
            )

            state = next_state

        print_row(ep, state, steps, V)

# ---------------- FINAL VALUES ----------------
print("\n" + "=" * 70)
print("Final learned values")
print("=" * 70)
print(
    f"TL={V[0]:.4f}, "
    f"A={V[1]:.4f}, "
    f"B={V[2]:.4f}, "
    f"C={V[3]:.4f}, "
    f"TR={V[4]:.4f}"
)
