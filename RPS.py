from collections import defaultdict, Counter
import random

# keep track of rewards of each strategy
# rewards = {'patter_detection': 0, '': 0, '': 0, '': 0}

counter_moves = {'R': 'P', 'P': 'S', 'S': 'R'}
my_history = list()


def freqStat(history):
    return counter_moves[counter_moves[max(set(history), key=history.count)]]


def detect_pattern(opponent_history, n=2):
    """
    Detects patterns in the opponent's moves using an n-gram model.
    n: The size of the n-gram (default is 2, which means pairs of moves)
    """
    ngram_model = defaultdict(int)

    # Generate n-grams and count their occurrences
    for i in range(len(opponent_history) - n + 1):
        ngram = tuple(opponent_history[i:i + n])
        ngram_model[ngram] += 1

    return ngram_model


def predict_next_move(opponent_history, ngram_model, n=2):
    """
    Predicts the next move based on the opponent's history and n-gram model.
    """
    if len(opponent_history) < n - 1:
        return random.choice(['R', 'P', 'S'])  # Not enough data, random choice

    last_ngram = tuple(opponent_history[-(n - 1):])  # Get last sequence
    possible_next_moves = {move: 0 for move in ['R', 'P', 'S']}

    # Check the most common move that follows the last n-gram
    for move in ['R', 'P', 'S']:
        next_ngram = last_ngram + (move,)
        if next_ngram in ngram_model:
            possible_next_moves[move] = ngram_model[next_ngram]

    # Predict the move with the highest occurrence
    predicted_move = max(possible_next_moves, key=possible_next_moves.get)

    return predicted_move


def pattern_det_wrapper(opponent_history, n=2):
    n = 2  # Number of moves to track in the pattern detection
    ngram_model = detect_pattern(opponent_history, n)
    predicted_move = predict_next_move(opponent_history, ngram_model, n)
    return predicted_move


def player(prev_play, opponent_history=[]):
    opponent_history.append(prev_play)
    if prev_play == '':
        move = random.choice(['R', 'P', 'S'])
        my_history.append(move)
        return move
    move = freqStat(my_history)

    my_history.append(move)

    return move
