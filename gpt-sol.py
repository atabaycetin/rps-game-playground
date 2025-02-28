import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import random

# =========================
# Global Settings
# =========================
WINDOW_SIZE = 10  # We'll store only the last 50 opponent moves
EPSILON = 0.60  # Exploration rate for strategy selection

# Opponent history (rolling window)
opponent_history = []

# Strategy performance tracking
STRATEGY_NAMES = ["frequency", "markov2", "neural"]
NUM_STRATEGIES = len(STRATEGY_NAMES)

# Off-policy rewards: how many wins/ties/losses each strategy *would* have had
strategy_rewards = [0.0] * NUM_STRATEGIES
strategy_counts = [0] * NUM_STRATEGIES

# We'll keep track of each sub-strategy's "predicted move" on the *previous turn*
# so we can see if it would have beaten the actual last move.
last_strategy_moves = [None] * NUM_STRATEGIES

# Easiest place to store: which move we actually "played" last turn
actual_chosen_move = None
actual_chosen_strategy = None


# =========================
# 1. Frequency (Rolling Window)
# =========================
def strategy_frequency(history):
    """
    Look at the frequency of 'R', 'P', 'S' in the rolling window (history).
    Predict the opponent will play the most frequent, and return the counter.
    If there's no history, pick randomly.
    """
    if len(history) == 0:
        return random.choice(['R', 'P', 'S'])

    # Count R, P, S in the rolling window
    count_r = history.count('R')
    count_p = history.count('P')
    count_s = history.count('S')
    freq_moves = [('R', count_r), ('P', count_p), ('S', count_s)]
    freq_moves.sort(key=lambda x: x[1], reverse=True)  # sort by frequency desc
    most_freq = freq_moves[0][0]  # The most common move

    return beat_move(most_freq)


# =========================
# 2. Two-Step Markov (Rolling Window)
# =========================
# We'll track transitions from (move[i], move[i+1]) -> move[i+2].
# e.g. if we see R,P -> S multiple times, that means "after R,P often comes S".
# Then for the last two moves, we see which next move is most common.
# We'll store these counts in a dictionary that's updated each turn.
markov2_counts = {}  # key = (m1, m2), value = dict of next-move counts e.g. {'R': int, 'P': int, 'S': int}


def init_markov2_counts():
    """Initialize or clear the 2-step Markov dictionary."""
    global markov2_counts
    markov2_counts = {}


def update_markov2(history):
    """
    After we add a new move to history, update the 2-step Markov counts
    with the last 3 moves in the rolling window if we have them.
    """
    if len(history) < 3:
        return  # not enough data yet

    m1, m2, m3 = history[-3], history[-2], history[-1]
    pair = (m1, m2)
    if pair not in markov2_counts:
        markov2_counts[pair] = {'R': 0, 'P': 0, 'S': 0}
    markov2_counts[pair][m3] += 1


def strategy_markov2(history):
    """
    If we have at least 2 moves in the history, look up the most likely next move
    given the last 2 moves. Then return the move that beats that predicted move.
    If there's no data or too little history, pick random.
    """
    if len(history) < 2:
        return random.choice(['R', 'P', 'S'])

    last_two = (history[-2], history[-1])
    if last_two not in markov2_counts:
        return random.choice(['R', 'P', 'S'])

    freq_dict = markov2_counts[last_two]  # e.g. {'R': 10, 'P': 2, 'S': 4}
    predicted_move = max(freq_dict, key=freq_dict.get)  # pick the max
    return beat_move(predicted_move)


# =========================
# 3. Small Neural Net
# =========================
# We'll feed the *last 2 moves* as input, each one-hot. => 6-dimensional input.
# We'll do a small feed-forward net with 1 hidden layer.

model = None


def build_model():
    m = keras.Sequential([
        layers.Input(shape=(6,)),
        layers.Dense(12, activation='relu'),
        layers.Dense(3, activation='softmax')
    ])
    m.compile(loss='categorical_crossentropy',
              optimizer=keras.optimizers.Adam(learning_rate=0.01),
              metrics=[])
    return m


def strategy_neural(history):
    """
    If we have <2 moves, pick random. Otherwise feed the last 2 moves into the model,
    interpret the 3-output as probabilities for R, P, S, pick the highest-prob move,
    then choose the counter to that move.
    """
    if len(history) < 2 or model is None:
        return random.choice(['R', 'P', 'S'])

    x = encode_last_two(history[-2], history[-1]).reshape(1, -1)  # shape (1, 6)
    probs = model.predict(x, verbose=0)[0]  # shape (3,)
    pred_idx = np.argmax(probs)  # 0->R, 1->P, 2->S
    predicted_move = ['R', 'P', 'S'][pred_idx]
    return beat_move(predicted_move)


def encode_last_two(m1, m2):
    """
    One-hot each move (3-dim) then concatenate => 6-dim.
    R=0,P=1,S=2
    """
    return np.concatenate([encode_move(m1), encode_move(m2)])


def encode_move(m):
    if m == 'R':
        return np.array([1, 0, 0])
    elif m == 'P':
        return np.array([0, 1, 0])
    else:  # 'S'
        return np.array([0, 0, 1])


def train_neural(old_m1, old_m2, actual_next):
    """
    We'll do a quick single-sample training step:
    Input = (old_m1, old_m2), label = actual_next.
    """
    if model is None:
        return
    x = encode_last_two(old_m1, old_m2).reshape(1, -1)
    y = encode_move(actual_next).reshape(1, -1)  # one-hot target
    model.fit(x, y, epochs=1, verbose=0)


# =========================
# Utility
# =========================
def beat_move(move):
    """Return the move that beats 'move'."""
    if move == 'R':
        return 'P'
    elif move == 'P':
        return 'S'
    else:
        return 'R'


def rps_result(my_move, opp_move):
    """+1 if my_move beats opp_move, 0 if tie, -1 if loss."""
    if my_move == opp_move:
        return 0
    wins = (my_move == 'R' and opp_move == 'S') or \
           (my_move == 'P' and opp_move == 'R') or \
           (my_move == 'S' and opp_move == 'P')
    return +1 if wins else -1


# =========================
# Off-Policy Evaluation
# =========================
def get_strategy_move(strat_index):
    """Compute what the given strategy WOULD play *based on the current history*."""
    if strat_index == 0:
        return strategy_frequency(opponent_history)
    elif strat_index == 1:
        return strategy_markov2(opponent_history)
    else:
        return strategy_neural(opponent_history)


def update_off_policy_rewards(opp_move):
    """
    After seeing the new opponent move (which ended the last round),
    compute how each strategy WOULD have done last turn, given the move it WOULD have chosen.
    Then update strategy_rewards and strategy_counts.
    """
    global last_strategy_moves
    # Each strategy's last move is in last_strategy_moves[i].
    # If it's not None, we can see how it would've fared vs opp_move.
    for i in range(NUM_STRATEGIES):
        if last_strategy_moves[i] is not None:
            outcome = rps_result(last_strategy_moves[i], opp_move)
            strategy_rewards[i] += outcome
            strategy_counts[i] += 1


def pick_strategy():
    """
    Epsilon-greedy among all strategies, using average reward = total_reward / usage_count.
    If usage_count=0, treat average as 0.0.
    """
    # With probability EPSILON, pick random
    if random.random() < EPSILON:
        return random.randint(0, NUM_STRATEGIES - 1)

    best_idx = 0
    best_avg = -999999
    for i in range(NUM_STRATEGIES):
        if strategy_counts[i] == 0:
            avg = 0.0
        else:
            avg = strategy_rewards[i] / strategy_counts[i]
        if avg > best_avg:
            best_avg = avg
            best_idx = i
    return best_idx


# =========================
# Main Player Function
# =========================
def player(prev_play, _=[]):
    """
    Called each turn with the opponent's move (prev_play).
    We'll:
      1. If we see a new move from the opponent, that means last round ended.
         => Off-policy update for all strategies (comparing their last predicted move vs prev_play).
         => Also update the rolling window, Markov2, and train the neural net if we have enough data.
      2. Pick which strategy to use for THIS turn (epsilon-greedy best so far).
      3. Compute the move from that strategy, store it as the 'last_strategy_moves' for next turn.
      4. Return the move.
    """
    global opponent_history
    global last_strategy_moves, actual_chosen_move, actual_chosen_strategy
    global model

    # Initialize model if needed
    if model is None:
        model = build_model()
        init_markov2_counts()

    # ========== STEP 1: We got a new opponent move => last round is complete
    if prev_play in ['R', 'P', 'S']:
        # Off-policy evaluation: compare each strategy's last move to this new opp move
        update_off_policy_rewards(prev_play)

        # Update rolling window
        opponent_history.append(prev_play)
        if len(opponent_history) > WINDOW_SIZE:
            opponent_history.pop(0)  # keep only last 50

        # Update Markov2
        update_markov2(opponent_history)

        # Train neural net if we have at least 3 moves (to form old_m1, old_m2 -> new)
        if len(opponent_history) >= 3:
            old_m1 = opponent_history[-3]
            old_m2 = opponent_history[-2]
            new_m = opponent_history[-1]
            train_neural(old_m1, old_m2, new_m)

    # ========== STEP 2: Pick a strategy for THIS turn
    chosen_strat = pick_strategy()
    actual_chosen_strategy = chosen_strat

    # ========== STEP 3: Compute the move from that strategy
    my_move = get_strategy_move(chosen_strat)
    actual_chosen_move = my_move

    # Also compute what EACH strategy WOULD do, store in last_strategy_moves
    for i in range(NUM_STRATEGIES):
        last_strategy_moves[i] = get_strategy_move(i)

    # ========== Return my move
    return my_move
