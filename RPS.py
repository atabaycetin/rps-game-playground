
"""

"""

import random

memory = dict()
memory['opponent_history'] = []
memory['cycle_found'] = False
memory['period'] = None
memory['cycle_substring'] = ""



game_count = 0
my_last = ''
strat_pick = ''
rewards = dict()
rewards['counter_last'] = 0
# rewards['frequency'] = 0
# rewards['markov2'] = 0
rewards['pattern'] = 0
guesses = dict()
guesses['counter_last'] = list()
# guesses['frequency'] = list()
# guesses['markov2'] = list()
guesses['pattern'] = list()

last_strategy = ''

# Returns 1 for win, -1 for loss, 0 for tie
def win(my_move, opp_move):
    if my_move == 'R' and opp_move == 'P':
        return -1
    elif my_move == 'P' and opp_move == 'R':
        return +1
    elif my_move == 'P' and opp_move == 'S':
        return -1
    elif my_move == 'S' and opp_move == 'P':
        return +1
    elif my_move == 'S' and opp_move == 'R':
        return -1
    elif my_move == 'R' and opp_move == 'S':
        return +1
    else:
        return 0

# Strategy for Quincy
def compute_prefix_function(S):
    """
    Computes the prefix function (pi array) for the string S.
    pi[i] = length of the longest proper prefix of S[:i+1]
            which is also a suffix of S[:i+1].
    """
    n = len(S)
    pi = [0] * n
    k = 0
    for i in range(1, n):
        while k > 0 and S[k] != S[i]:
            k = pi[k - 1]
        if S[k] == S[i]:
            k += 1
        pi[i] = k
    return pi
def smallest_period_kmp(S):
    """
    Returns the length of the smallest period of S using the prefix function.
    If S is entirely composed of repeats of some substring of length p,
    we return p. Otherwise, we return len(S).
    """
    n = len(S)
    if n == 0:
        return 0
    pi = compute_prefix_function(S)
    p = pi[n - 1]       # length of the longest border
    period = n - p
    if n % period == 0:
        return period
    else:
        return n
def beat_move(move):
    """
    Returns the move that beats 'move':
      R -> P,  P -> S,  S -> R
    """
    if move == 'R':
        return 'P'
    elif move == 'P':
        return 'S'
    else:
        return 'R'
def counter_pattern(prev_play, opponent_history=[]):
    global memory, game_count
    memory['opponent_history'] = opponent_history
    cycle_found = memory['cycle_found']
    period = memory['period']
    cycle_substring = memory['cycle_substring']

    if game_count == 1000:
        cycle_found = False
    #print(cycle_found, period)
    if cycle_found and period is not None and period > 0:
        # Predict next move from the known cycle
        next_index = len(opponent_history) % period
        predicted_move = cycle_substring[next_index]
        #print("Cycle Exploited")
        #print(beat_move(predicted_move))
        return beat_move(predicted_move)

    else:
        # We have not found a cycle yet, so see if we can detect one now
        opp_hist = "".join(opponent_history)
        period = smallest_period_kmp(opp_hist)
        # If p < len(S), we've found a repeating block
        if period < len(opp_hist):
            memory['cycle_found'] = True

            memory['period'] = period
            memory['cycle_substring'] = opp_hist[:period]
            # Predict immediately using the newly found cycle
            next_index = len(opponent_history) % period
            predicted_move = memory['cycle_substring'][next_index]
            print("Cycle Found")
            print(opp_hist[:period])
            print(beat_move(predicted_move))
            return beat_move(predicted_move)
        else:
            # No cycle yet => fallback: random
            return random.choice(['R', 'P', 'S'])


# Strategy for Kris
def counter_last(opp_last, opponent_history=[]):
    global my_last
    if len(opponent_history) > 1:
        if my_last == 'R':
            my_last = 'S'
            return 'S'
        elif my_last == 'P':
            my_last = 'R'
            return 'R'
        else:
            my_last = 'P'
            return 'P'

def player(prev_play, opponent_history=[]):
    global my_last, strat_pick, rewards, game_count
    game_count += 1
    # reset history and rewards dictionary when we change players (every 1000 games)
    if (len(opponent_history)) == 1000:
        opponent_history.clear()
        rewards = {key: 0 for key in rewards}

    opponent_history.append(prev_play)

    if my_last == '' or prev_play == '':
        my_last = random.choice(['R', 'P', 'S'])
        return my_last

    if len(opponent_history) > 2:
        rewards['counter_last'] += win(guesses['counter_last'][-1], prev_play)
        rewards['pattern'] += win(guesses['pattern'][-1], prev_play)


    strat_pick = max(rewards, key=rewards.get)
    #print("#Game", game_count)
    #print(strat_pick)
    #print("Reward Counter_last:", rewards['counter_last'])
    #print("Reward Counter_pattern:", rewards['pattern'])
    if strat_pick == 'counter_last':
        my_last = counter_last(prev_play, opponent_history)
        guesses['counter_last'].append(my_last)
        guesses['pattern'].append(counter_pattern(prev_play, opponent_history))
        return my_last

    elif strat_pick == 'pattern':
        my_last = counter_pattern(prev_play, opponent_history)
        guesses['pattern'].append(my_last)
        guesses['counter_last'].append(counter_last(prev_play, opponent_history))
        return my_last
    #print(strat_pick)
    #print(rewards['counter_last'])
    #print(rewards['pattern'])
    return counter_last(prev_play, opponent_history)