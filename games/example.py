class twoPlayerGame():
    def __init__(self):
        self.initialState = []

    def actions(self, state):
        """Returns a list of all possible moves"""
        pass

    def is_valid_move(self, state, move):
        """Returns True if the move given is a valid move on the current board"""
        pass

    def result(self, state, move):
        """Returns that state that results in the move being played on the current board"""
        pass

    def is_terminal(self, state):
        """Return if there is a winner of the game"""
        pass

    def get_network_model(self):
        """Return a compiled Neural Network"""
        pass