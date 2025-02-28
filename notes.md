# RPS_game.py Notes:

## play function:
    arguments: player1, player2, #games, verbose
    process: compares 
    returns: win rate of player1

## quincy player function:
    arguments:  prev_play (does nothing), counter default = 0
    process:    increments counter
                `choices = ["R", "R", "P", "P", "S"]`
    returns:    choices[counter[0] % len(choices)]

    Desc.:      it returns R, R, P, P, S periodically every 5 rounds

    Possible Tactic:    You can create a model to recognize this pattern and move accordingly

## mrugesh player function:
    arguments:  `prev_opponent_play, opponent_history=[]`
    process:    takes last 10 moves of the player and finds the most frequent
                if cannot find the most frequent, takes **S**
                finds the winning response according to the most freq
    returns:    winning response

    Possible Tactic:    You can change your response when there is a frequent move played

## kris player function:
    arguments:  prev_opponent_play
    process:    finds the ideal response according to the last play
    returns:    **R** for the first round
                winning response to the last play
    
    Possible Tactic:    Don't play the same move again

## abbey player function: ----------------------------------------------------------------------------------------------
    arguments:  prev_opponent_play, opponent_history=[], play_order
    process:    
    returns:    winning response according to the prediction

## human player function:
    takes the input and returns it

## random player function:
    returns a random move using random.choice function