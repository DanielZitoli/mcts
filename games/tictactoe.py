import tensorflow as tf

class TicTacToe():
    def __init__(self):
        self.initialState = [0] * 9

    def actions(self, state):
        """Returns a list of all possible actions"""
        return [i for i in range(9) if state[i] == 0]
    
    def parse_legal_actions(self, state, priors):
        """Takes as input a list of prior probabilities predicted by policy network and returns a list 
        of tuples (action, prior) of legal actions with the prior probabilities normalized"""
        
        actions = []
        total_probability = 0
        for action in self.actions(state):
            actions.append([action, priors[action]])
            total_probability += priors[action]
        for action in actions:
            action[1] = action[1]/total_probability
        return actions


    def is_valid_action(self, state, action):
        """Returns True if the action given is a valid action in the current state"""
        if action is type("int") and action >= 0 and action < 9:
            return state[action] == 0
        return False

    def result(self, state, action, to_play=1):
        """Returns that state that results in the action being played on the current state"""
        newState = state.copy()
        newState[action] = to_play
        return newState

    def is_terminal(self, state):
        """Return if there is a winner of the game"""
        for i in range(3):
            if state[i] and state[i] == state[i+3] and state[i] == state[i+6]:
                return state[i]
            if state[3*i] and state[3*i] == state[3*i + 1] and state[3*i] == state[3*i + 2]:
                return state[3*i]

        if state[0] and state[0] == state[4] and state[0] == state[8]:
            return state[0]
        if state[2] and state[2] == state[4] and state[2] == state[6]:
            return state[2]
        
        if all(state):
            return 0

        return None

    def switch_players(self, state):
        """Returns the inverted state with player 1 having the next move"""
        for num in state:
            num *= -1
        return state
    
    def print_game(self, state):
        """Prints board representation to the console"""
        pieces = [' ', 'X', 'O']
        print(f"{pieces[state[0]]}|{pieces[state[1]]}|{pieces[state[2]]}")
        print("-----")
        print(f"{pieces[state[3]]}|{pieces[state[4]]}|{pieces[state[5]]}")
        print("-----")
        print(f"{pieces[state[6]]}|{pieces[state[7]]}|{pieces[state[8]]}")
    
    def get_user_input(self):
        while True:
            action = int(input())
            if self.is_valid_action(action):
                break
        return action
        

    def create_input(self, state):
        """Takes a given state and returns the representation that will be fed to the Neural Network"""
        return state.copy()
    
    def create_labels(self, distribution):
        """Takes as input an action visitation distribution found by mcts and returns a label used to train the network"""
        output = self.initialState.copy()
        for action, probability in distribution:
            output[action] = probability
        return output

    def get_network_model(self):
        """Return a compiled Neural Network"""
        
        model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=(9,)),

            tf.keras.layers.Dense(128, activation='sigmoid'), 
            
            tf.keras.layers.Dense(9, activation="softmax")
        ])

        # categorical_crossentropy loss function seems to be best for networks that predict many different types of labels
        model.compile(optimizer='adam',
                loss="categorical_crossentropy", 
                metrics=['accuracy'])

        return model