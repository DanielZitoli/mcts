from games.tictactoe import TicTacToe
from montecarlo import MonteCarloAI

game = TicTacToe()

settings = {
    "rounds": 0,               # number of total training rounds
    "games": 0,                # number of games per training round
    "simulations": 100,        # number of mcts simulations per move
    "epochs": 0,               # number of times to run training data through network each round
    "model": "second"          # name of NN model, or the name you want to store new model       
}

ai = MonteCarloAI(game, settings)
model = ai.model

state = [1, 1, 0, -1, -1, 0, -1, 1, 0]

print(ai.make_best_move(state))

input = game.create_input(state)
output = model.predict([input])
print(output)