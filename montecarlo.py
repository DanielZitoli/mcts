import numpy as np
import tensorflow as tf
import random
import os
import copy

class MonteCarloAI():
    def __init__(self, game, settings):
        self.game = game
        self.rounds = settings["rounds"]
        self.games = settings["games"]
        self.simulations = settings["simulations"]
        self.epochs = settings["epochs"]
        self.model_name = settings["model"]
        self.model_directory = os.path.join("models", self.game.__class__.__name__, self.model_name)
        self.training_data = []

        model = self.fetch_model()
        if model:
            self.model = model
        else:
            self.model = self.game.get_network_model()
    
    def train(self):
        """Trains the model by collecting data self-play and fitting network after each round"""
        for round in range(self.rounds):
            print("Training Round:" + str(round+1) + "/" + str(self.rounds) + ".")
            for _ in range(self.games):
                self.play_game()
            self.train_model()

        print("Saving model to " + self.model_name)
        self.save_model()
    
    def play_game(self):
        """Plays a single game"""
        state = copy.deepcopy(self.game.initialState)

        while not self.game.is_terminal(state):
            move = self.make_best_move(state, training=True)
            state = self.game.result(state, move)
            state = self.game.switch_players(state)

    def train_model(self):
        """Fits the model using self-play data"""
        random.shuffle(self.training_data)
        data, labels = [], []
        for state, label in self.training_data:
            data.append(state)
            labels.append(label)
        self.model.fit(data, labels, epochs=self.epochs)
        self.training_data = []

    def make_best_move(self, state, training=False):
        """Selects best move given a current state by performing random rollouts"""
        root = Node(state, None, 0, None, 1)

        root.expand(self.game, self.model)

        for _ in range(self.simulations):
            #SELECT
            node = root
            while node.children:
                node = node.select_child()
            
            #EXPAND
            # if the selected node has already been visited, generate it's next states then pick one randomly
            if node.visit_count:
                node.expand(self.game, self.model)
                if node.children:
                    node = random.choice(node.children)
            
            #SIMULATE
            value = node.simulate(self.game)

            #BACKPROPOGRATE
            node.backpropogate(value)
        
        self.game.print_game(state)
        print(root)
        print()
        for child in root.children:
            print(child)

        if training:
            #creates training data from root state and state visit distributions
            data, label = self.game.create_input(state), self.game.create_labels(root.policy_distribution())
            self.training_data.append((data, label))
            return root.select_child().action
        else:
            #selects child with highest visit count
            best_child = None
            highest_count = 0
            for child in root.children:
                if child.visit_count > highest_count:
                    best_child = child
                    highest_count = child.visit_count
            return best_child.action

    
    def fetch_model(self):
        """Searches for the model saved in the file at model/{game_name}/{model_name} and returns the saved model.
            If the model doesn't exist, return None"""
        if os.path.isdir(self.model_directory):
            return tf.keras.models.load_model(self.model_directory)
        else:
            return None

    def save_model(self):
        """Saves model to directory ending in self.model_name for later use"""
        self.model.save(self.model_directory)

        

class Node():
    def __init__(self, state, action, prior, parent, to_play):
        self.state = copy.deepcopy(state)
        self.action = action
        self.prior = prior
        self.parent = parent
        self.to_play = to_play
        self.visit_count = 0
        self.value_count = 0
        self.children = []

    def __str__(self):
        return f"Move: {self.action}, Value: {self.value_count}, Games: {self.visit_count}."

    def get_value(self):
        """Returns value of the node in range [0,1]"""
        return self.value_count/self.visit_count

    def get_ucb(self):
        """Returns Upper Confidence Bound of node"""
        if self.visit_count == 0:
            return 100
        node_value = self.value_count/self.visit_count 
        exploration_value = np.sqrt(2*np.log(self.parent.visit_count)/(self.visit_count))
        
        return exploration_value + node_value
    
    def policy_distribution(self):
        """Returns a normalized distribution of all children visited by the root"""
        distribution = []
        visit_sum = 0
        for child in self.children:
            distribution.append([child.action, child.visit_count])
            visit_sum += child.visit_count
        #normalizes probabilty by dividing the total visits
        for move in distribution:
            move[1] = move[1]/visit_sum
        
        return distribution

    def select_child(self):
        """Selects child of given node with the greatest UCB score"""
        maxUCB = -2
        best_child = None
        for child in self.children:
            curUCB = child.get_ucb()
            if curUCB > maxUCB:
                maxUCB = curUCB
                best_child = child
        return best_child

    def expand(self, game, model):
        """Adds a list to the given node of all of it's legal children states with prior probabilities from policy network"""
        priors = model.predict([game.create_input(self.state)])[0]
        actions = game.parse_legal_actions(self.state, priors)

        for action, prior in actions:
            self.children.append(Node(
                state=game.result(self.state, action),
                action=action,
                prior=prior,
                parent=self,
                to_play=-1*self.to_play
                ))

    def simulate(self, game):
        """Preforms a random rollout from the given state and returns a value of the terminal state,
           or the value given from the value network"""
        state = copy.deepcopy(self.state)
        to_play = self.to_play
        while game.is_terminal(state) is None:
            #first, check if state's value from network is above threshold, if so return that value
            random_action = random.choice(game.actions(state))
            state = game.result(state, random_action)
            state = game.switch_players(state)
            to_play *= -1

        return game.is_terminal(state) * to_play * self.to_play
    
    def backpropogate(self, value):
        """Backpropogates the value from the simulation up the search path to the root node"""
        self.visit_count += 1
        self.value_count += value   

        if self.parent:
            self.parent.backpropogate(-1*value)
    
