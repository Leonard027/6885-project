import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, losses

class PolicyValueNet(tf.keras.Model):
    """
    Policy-value network implemented with TensorFlow 2.x (tf.keras).
    This model predicts action probabilities and evaluates the state value.
    """
    def __init__(self, board_width, board_height, model_file=None):
        super(PolicyValueNet, self).__init__()
        self.board_width = board_width
        self.board_height = board_height

        # Convolutional layers for feature extraction
        self.conv1 = layers.Conv2D(32, (3, 3), padding="same", activation='relu')
        self.conv2 = layers.Conv2D(64, (3, 3), padding="same", activation='relu')
        self.conv3 = layers.Conv2D(128, (3, 3), padding="same", activation='relu')

        # Policy head
        self.policy_conv = layers.Conv2D(4, (1, 1), padding="same", activation='relu')
        self.policy_flat = layers.Flatten()
        self.policy_fc = layers.Dense(board_width * board_height, activation='softmax')

        # Value head
        self.value_conv = layers.Conv2D(2, (1, 1), padding="same", activation='relu')
        self.value_flat = layers.Flatten()
        self.value_fc1 = layers.Dense(64, activation='relu')
        self.value_fc2 = layers.Dense(1, activation='tanh')

        if model_file:
            self.restore_model(model_file)

    def call(self, inputs, training=False):
        """
        Forward pass to compute action probabilities and state value.

        Args:
            inputs: Input tensor with shape [N, board_height, board_width, 4].
            training: Boolean indicating whether the model is in training mode.

        Returns:
            act_probs: Action probabilities with shape [N, board_height * board_width].
            value: State value with shape [N, 1].
        """
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.conv3(x)

        # Policy head
        policy = self.policy_conv(x)
        policy = self.policy_flat(policy)
        act_probs = self.policy_fc(policy)

        # Value head
        value = self.value_conv(x)
        value = self.value_flat(value)
        value = self.value_fc1(value)
        value = self.value_fc2(value)

        return act_probs, value

    def policy_value(self, state_batch):
        """
        Predict action probabilities and state values for a batch of states.

        Args:
            state_batch: Tensor with shape [N, board_height, board_width, 4].

        Returns:
            act_probs: Numpy array of action probabilities with shape [N, board_width * board_height].
            value: Numpy array of state values with shape [N, 1].
        """
        act_probs, value = self(state_batch, training=False)
        return act_probs.numpy(), value.numpy()

    def policy_value_fn(self, board):
        """
        Interface for Monte Carlo Tree Search (MCTS) to get action probabilities and value.

        Args:
            board: Board object representing the current game state.

        Returns:
            act_probs: List of (action, probability) tuples for legal moves.
            value: Float representing the predicted value of the board state.
        """
        legal_positions = board.availables
        current_state = board.current_state().reshape(-1, self.board_height, self.board_width, 4)
        act_probs, value = self.policy_value(current_state)
        act_probs = act_probs[0]  # Take the first batch
        value = value[0][0]  # Extract the scalar value
        act_probs = list(zip(legal_positions, act_probs[legal_positions]))
        return act_probs, value

    def train_step_custom(self, state_batch, mcts_probs, winner_batch, lr):
        """
        Perform a custom training step.

        Args:
            state_batch: Tensor of shape [N, board_height, board_width, 4].
            mcts_probs: Tensor of shape [N, board_width * board_height].
            winner_batch: Tensor of shape [N, 1].
            lr: Learning rate.

        Returns:
            loss: Float, total loss value.
            entropy: Float, policy entropy for monitoring.
        """
        optimizer = optimizers.Adam(learning_rate=lr)

        with tf.GradientTape() as tape:
            act_probs, value = self(state_batch, training=True)

            # Compute value loss
            value_loss = tf.reduce_mean(tf.square(winner_batch - value))

            # Compute policy loss
            policy_loss = -tf.reduce_mean(tf.reduce_sum(mcts_probs * tf.math.log(act_probs + 1e-10), axis=1))

            # Total loss
            loss = value_loss + policy_loss

        # Compute gradients and apply updates
        grads = tape.gradient(loss, self.trainable_variables)
        optimizer.apply_gradients(zip(grads, self.trainable_variables))

        # Compute entropy
        entropy = -tf.reduce_mean(tf.reduce_sum(act_probs * tf.math.log(act_probs + 1e-10), axis=1))
        return loss.numpy(), entropy.numpy()

    def save_model(self, model_path):
        """
        Save the model weights to a file.

        Args:
            model_path: String, path to save the model (e.g., 'model.h5').
        """
        if not model_path.endswith('.h5'):
            model_path += '.h5'
        self.save_weights(model_path)

    def restore_model(self, model_path):
        """
        Load the model weights from a file.

        Args:
            model_path: String, path to the saved model.
        """
        self.load_weights(model_path).expect_partial()
