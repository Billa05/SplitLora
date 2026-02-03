import random

# ---------- Board Utilities ----------

def empty_board():
    return [' ' for _ in range(9)]

def board_to_state(board):
    return ''.join(board)

def available_actions(board):
    return [i for i in range(9) if board[i] == ' ']

def check_winner(board):
    wins = [
        (0,1,2),(3,4,5),(6,7,8),
        (0,3,6),(1,4,7),(2,5,8),
        (0,4,8),(2,4,6)
    ]
    for a, b, c in wins:
        if board[a] == board[b] == board[c] != ' ':
            return board[a]
    if ' ' not in board:
        return 'D'
    return None

def print_board(board):
    for i in range(0, 9, 3):
        print(board[i], '|', board[i+1], '|', board[i+2])
    print()

# ---------- Q Learning ----------

Q = {}
alpha = 0.1
gamma = 0.9
epsilon = 0.2

def get_q(state, action):
    return Q.get((state, action), 0.0)

def choose_action(state, board, explore=True):
    actions = available_actions(board)

    if explore and random.random() < epsilon:
        return random.choice(actions)

    q_vals = [(get_q(state, a), a) for a in actions]
    max_q = max(q_vals)[0]
    best = [a for q, a in q_vals if q == max_q]
    return random.choice(best)

def update_q(state, action, reward, next_state, next_board):
    old_q = get_q(state, action)
    next_actions = available_actions(next_board)

    if next_actions:
        future_q = max(get_q(next_state, a) for a in next_actions)
    else:
        future_q = 0

    Q[(state, action)] = old_q + alpha * (reward + gamma * future_q - old_q)

# ---------- Training ----------

def train(episodes=150000):
    for _ in range(episodes):
        board = empty_board()
        current_player = 'X'

        while True:
            state = board_to_state(board)
            action = choose_action(state, board, explore=True)
            board[action] = current_player

            winner = check_winner(board)
            next_state = board_to_state(board)

            if winner == current_player:
                update_q(state, action, 1, next_state, board)
                break
            elif winner == 'D':
                update_q(state, action, 0.5, next_state, board)
                break
            elif winner is not None:
                update_q(state, action, -1, next_state, board)
                break
            else:
                update_q(state, action, 0, next_state, board)
                current_player = 'O' if current_player == 'X' else 'X'

# ---------- Human vs AI ----------

def play_human_vs_ai():
    board = empty_board()
    current_player = 'X'

    while True:
        print_board(board)

        if current_player == 'X':
            state = board_to_state(board)
            print("AI Q-values:")
            for a in available_actions(board):
                print(f" Action {a}: {get_q(state, a):.3f}")

            move = choose_action(state, board, explore=False)
            print("AI chooses:", move)
            board[move] = 'X'

        else:
            move = int(input("Enter position (0-8): "))
            if move not in available_actions(board):
                print("Invalid move!")
                continue
            board[move] = 'O'

        winner = check_winner(board)
        if winner:
            print_board(board)
            print("Winner:", winner)
            return

        current_player = 'O' if current_player == 'X' else 'X'

# ---------- Run ----------

train()
play_human_vs_ai()
