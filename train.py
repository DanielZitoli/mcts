from games.tictactoe import TicTacToe
from montecarlo import MonteCarloAI


settings = {
    "rounds": 3,               # number of total training rounds
    "games": 5,               # number of games per training round
    "simulations": 100,        # number of mcts simulations per move
    "epochs": 3,                # number of times to run training data through network each round
    "model": "second"          # name of NN model, or the name you want to store new model       
}

game = TicTacToe()

ai = MonteCarloAI(game, settings)

ai.train()