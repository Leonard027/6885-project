from __future__ import print_function
import theano
import theano.tensor as T
import lasagne
import pickle

class PolicyValueNet:
    """
    Policy-value network implemented with Theano and Lasagne.
    This network predicts action probabilities and evaluates the state value.
    """
    def __init__(self, board_width, board_height, model_file=None):
        self.board_width = board_width
        self.board_height = board_height
        self.learning_rate = T.scalar('learning_rate')  # Learning rate placeholder
        self.l2_const = 1e-4  # L2 regularization coefficient
        self.create_policy_value_net()
        self._loss_train_op()  # Define the loss and training operation
        if model_file:
            try:
                net_params = pickle.load(open(model_file, 'rb'))
            except:
                # For compatibility with Python 3
                net_params = pickle.load(open(model_file, 'rb'), encoding='bytes')
            lasagne.layers.set_all_param_values(
                [self.policy_net, self.value_net], net_params
            )

    def create_policy_value_net(self):
        """
        Build the policy-value network with convolutional layers for feature extraction
        and separate heads for action probabilities and state value prediction.
        """
        self.state_input = T.tensor4('state')  # Input tensor for the board state
        self.winner = T.vector('winner')  # Target value for state evaluation
        self.mcts_probs = T.matrix('mcts_probs')  # Target probabilities for actions

        # Input layer for the board state
        network = lasagne.layers.InputLayer(
            shape=(None, 4, self.board_width, self.board_height),
            input_var=self.state_input
        )
        
        # Feature extraction with convolutional layers
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=32, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=64, filter_size=(3, 3), pad='same')
        network = lasagne.layers.Conv2DLayer(
            network, num_filters=128, filter_size=(3, 3), pad='same')
        
        # Policy network head
        policy_net = lasagne.layers.Conv2DLayer(
            network, num_filters=4, filter_size=(1, 1))
        self.policy_net = lasagne.layers.DenseLayer(
            policy_net, num_units=self.board_width * self.board_height,
            nonlinearity=lasagne.nonlinearities.softmax)

        # Value network head
        value_net = lasagne.layers.Conv2DLayer(
            network, num_filters=2, filter_size=(1, 1))
        value_net = lasagne.layers.DenseLayer(value_net, num_units=64)
        self.value_net = lasagne.layers.DenseLayer(
            value_net, num_units=1,
            nonlinearity=lasagne.nonlinearities.tanh)

        # Obtain outputs for action probabilities and state value
        self.action_probs, self.value = lasagne.layers.get_output(
            [self.policy_net, self.value_net])

        # Compile Theano function for inference
        self.policy_value = theano.function(
            [self.state_input],
            [self.action_probs, self.value],
            allow_input_downcast=True
        )

    def policy_value_fn(self, board):
        """
        Predict action probabilities and state value for a given board.

        Args:
            board: Board object representing the game state.

        Returns:
            act_probs: List of (action, probability) pairs for legal moves.
            value: Predicted value of the board state.
        """
        legal_positions = board.availables
        current_state = board.current_state()
        act_probs, value = self.policy_value(
            current_state.reshape(-1, 4, self.board_width, self.board_height))
        act_probs = zip(legal_positions, act_probs.flatten()[legal_positions])
        return act_probs, value[0][0]

    def _loss_train_op(self):
        """
        Define the loss function and training operation.

        Loss consists of three components:
        1. Mean squared error for value prediction: (z - v)^2
        2. Cross-entropy for policy prediction: -pi^T * log(p)
        3. L2 regularization: c * ||theta||^2
        """
        params = lasagne.layers.get_all_params(
            [self.policy_net, self.value_net], trainable=True)

        # Value prediction loss
        value_loss = lasagne.objectives.squared_error(
            self.winner, self.value.flatten())
        
        # Policy prediction loss
        policy_loss = lasagne.objectives.categorical_crossentropy(
            self.action_probs, self.mcts_probs)

        # L2 regularization
        l2_penalty = lasagne.regularization.apply_penalty(
            params, lasagne.regularization.l2)

        # Combine losses
        self.loss = self.l2_const * l2_penalty + lasagne.objectives.aggregate(
            value_loss + policy_loss, mode='mean')

        # Policy entropy for monitoring (optional)
        self.entropy = -T.mean(T.sum(
            self.action_probs * T.log(self.action_probs + 1e-10), axis=1))

        # Training operation using Adam optimizer
        updates = lasagne.updates.adam(self.loss, params,
                                       learning_rate=self.learning_rate)
        self.train_step = theano.function(
            [self.state_input, self.mcts_probs, self.winner, self.learning_rate],
            [self.loss, self.entropy],
            updates=updates,
            allow_input_downcast=True
        )

    def get_policy_param(self):
        """
        Retrieve the current parameters of the policy-value network.

        Returns:
            net_params: List of parameter values for the network.
        """
        net_params = lasagne.layers.get_all_param_values(
            [self.policy_net, self.value_net])
        return net_params

    def save_model(self, model_file):
        """
        Save the network parameters to a file.

        Args:
            model_file: Path to the file where parameters will be saved.
        """
        net_params = self.get_policy_param()  # Get model parameters
        pickle.dump(net_params, open(model_file, 'wb'), protocol=2)
