from games.tictactoe import TicTacToe
from montecarlo import MonteCarloAI

import copy
import random

settings = {
    "rounds": 0,                # number of total training rounds
    "games": 0,                 # number of games per training round
    "simulations": 0,           # number of mcts simulations per move
    "epochs": 0,                # number of times to run training data through network each round
    "model": "best"            # name of NN model, or the name you want to store new model       
}

game = TicTacToe()

ai = MonteCarloAI(game, settings)

state = copy.deepcopy(game.initialState)

if random.random() > 0.5:
    print("Human to play first.")
else:
    print("AI to play first.")
    game.print_game(state)
    action = ai.make_best_move(state)
    state = game.result(state, action)
    print("Ai made move: " + action)
    state = game.switch_players(state)


while True:
    # Plays move for Human
    game.print_game(state)
    playerAction = game.get_user_input()
    state = game.result(state, playerAction)
    outcome = game.is_terminal(state)
    if outcome is not None:
        game.print_game(state)
        if outcome == 1:
            print("Human Wins!")
        elif outcome == 0:
            print("Draw!")
        else:
            print("AI Wins!")
        break

    game.print_game(state)
    state = game.switch_players(state)

    # Plays move for Ai
    AiAction = ai.make_best_move(state)
    state = game.result(state, AiAction)
    print("Ai made move: " + AiAction)
    outcome = game.is_terminal(state)
    if outcome is not None:
        game.print_game(state)
        if outcome == 1:
            print("Human Wins!")
        elif outcome == 0:
            print("Draw!")
        else:
            print("AI Wins!")
        break
    state = game.switch_players(state)


