import os
import random
import numpy as np
from collections import defaultdict, deque
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet  # TensorFlow 2.x version
import tensorflow as tf
import matplotlib.pyplot as plt
import datetime

# Configure TensorFlow to use GPU and manage GPU memory
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Only show errors and warnings

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    try:
        # Enable memory growth to optimize GPU usage
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        print("GPU is enabled.")
    except RuntimeError as e:
        print(e)
else:
    print("No GPU detected. Using CPU.")

class TrainPipeline:
    """
    Training pipeline for AlphaZero Gomoku.
    This class handles data collection, policy updates, and performance evaluation.
    """
    def __init__(self, init_model=None):
        # Game and board settings
        self.board_width = 6
        self.board_height = 6
        self.n_in_row = 4
        self.board = Board(width=self.board_width, height=self.board_height, n_in_row=self.n_in_row)
        self.game = Game(self.board)

        # Training hyperparameters
        self.learn_rate = 2e-3
        self.lr_multiplier = 1.0  # Adaptive learning rate multiplier
        self.temp = 1.0  # Temperature for exploration
        self.n_playout = 400  # Number of MCTS playouts
        self.c_puct = 5  # Exploration-exploitation balance
        self.buffer_size = 10000  # Replay buffer size
        self.batch_size = 512  # Mini-batch size for training
        self.data_buffer = deque(maxlen=self.buffer_size)
        self.play_batch_size = 1  # Number of self-play games per batch
        self.epochs = 5  # Epochs per training update
        self.kl_targ = 0.02  # Target KL divergence
        self.check_freq = 50  # Frequency of policy evaluation
        self.game_batch_num = 50  # Total training batches
        self.best_win_ratio = 0.0  # Best win ratio against baseline
        self.pure_mcts_playout_num = 1000  # Playouts for baseline MCTS

        # Initialize the policy-value network
        if init_model:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height, model_file=init_model)
        else:
            self.policy_value_net = PolicyValueNet(self.board_width, self.board_height)

        # Compile the network with an optimizer
        self.policy_value_net.compile_model(learning_rate=self.learn_rate)

        # MCTS player with the initialized policy-value network
        self.mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=self.c_puct, n_playout=self.n_playout, is_selfplay=1)

        # Metrics storage for visualization
        self.batch_index_list = []  # Batch indices
        self.loss_list = []  # Training loss values
        self.entropy_list = []  # Policy entropy values
        self.kl_list = []  # KL divergence values
        self.explained_var_old_list = []  # Explained variance (old predictions)
        self.explained_var_new_list = []  # Explained variance (new predictions)
        self.win_ratio_list = []  # Win ratios against baseline

    def get_equi_data(self, play_data):
        """
        Augment data by generating rotations and flips of the board state.
        This helps the network learn rotational and reflection invariance.
        """
        extend_data = []
        for state, mcts_prob, winner in play_data:
            for i in [1, 2, 3, 4]:
                # Rotate the state and probabilities
                rotated_state = np.rot90(state, i, axes=(1, 2))
                rotated_mcts_prob = np.rot90(np.flipud(mcts_prob.reshape(self.board_height, self.board_width)), i)
                extend_data.append((rotated_state, np.flipud(rotated_mcts_prob).flatten(), winner))
                
                # Flip the state horizontally
                flipped_state = np.fliplr(rotated_state)
                flipped_mcts_prob = np.fliplr(rotated_mcts_prob)
                extend_data.append((flipped_state, np.flipud(flipped_mcts_prob).flatten(), winner))
        return extend_data

    def collect_selfplay_data(self, n_games=1):
        """
        Collect training data from self-play games using the current policy.
        The resulting data is augmented for better generalization.
        """
        for _ in range(n_games):
            winner, play_data = self.game.start_self_play(self.mcts_player, temp=self.temp)
            play_data = list(play_data)
            self.episode_len = len(play_data)
            play_data = self.get_equi_data(play_data)
            self.data_buffer.extend(play_data)

    def policy_update(self):
        """
        Perform one training step on a batch of data from the replay buffer.
        Includes adaptive learning rate adjustment and KL divergence tracking.
        """
        mini_batch = random.sample(self.data_buffer, self.batch_size)
        state_batch = np.array([data[0] for data in mini_batch])
        mcts_probs_batch = np.array([data[1] for data in mini_batch])
        winner_batch = np.array([data[2] for data in mini_batch]).reshape(-1, 1)

        # Adjust state format for TensorFlow [N, 4, H, W] -> [N, H, W, 4]
        state_batch = np.transpose(state_batch, (0, 2, 3, 1))

        # Predict old probabilities for KL divergence computation
        old_probs, old_v = self.policy_value_net.policy_value(state_batch)
        for i in range(self.epochs):
            loss, entropy = self.policy_value_net.train_step_custom(
                state_batch, mcts_probs_batch, winner_batch, self.learn_rate * self.lr_multiplier
            )
            new_probs, new_v = self.policy_value_net.policy_value(state_batch)
            kl = np.mean(np.sum(old_probs * (np.log(old_probs + 1e-10) - np.log(new_probs + 1e-10)), axis=1))
            if kl > self.kl_targ * 4:
                break

        # Adjust learning rate based on KL divergence
        if kl > self.kl_targ * 2 and self.lr_multiplier > 0.1:
            self.lr_multiplier /= 1.5
        elif kl < self.kl_targ / 2 and self.lr_multiplier < 10:
            self.lr_multiplier *= 1.5

        explained_var_old = 1 - np.var(winner_batch - old_v.flatten()) / np.var(winner_batch)
        explained_var_new = 1 - np.var(winner_batch - new_v.flatten()) / np.var(winner_batch)

        print(("kl:{:.5f}, lr_multiplier:{:.3f}, loss:{}, entropy:{}, explained_var_old:{:.3f}, explained_var_new:{:.3f}")
              .format(kl, self.lr_multiplier, loss, entropy, explained_var_old, explained_var_new))

        # Store metrics for analysis
        self.batch_index_list.append(len(self.batch_index_list) + 1)
        self.loss_list.append(loss)
        self.entropy_list.append(entropy)
        self.kl_list.append(kl)
        self.explained_var_old_list.append(explained_var_old)
        self.explained_var_new_list.append(explained_var_new)

        return loss, entropy

    def policy_evaluate(self, n_games=10):
        """
        Evaluate the current policy by playing against a pure MCTS baseline.
        """
        current_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                         c_puct=self.c_puct, n_playout=self.n_playout)
        pure_mcts_player = MCTSPlayer(self.policy_value_net.policy_value_fn,
                                      c_puct=5, n_playout=self.pure_mcts_playout_num)

        win_cnt = defaultdict(int)
        for i in range(n_games):
            winner = self.game.start_play(current_mcts_player, pure_mcts_player, start_player=i % 2, is_shown=0)
            win_cnt[winner] += 1
        win_ratio = 1.0 * (win_cnt[1] + 0.5 * win_cnt[-1]) / n_games
        print("num_playouts:{}, win: {}, lose: {}, tie:{}".format(
            self.pure_mcts_playout_num, win_cnt[1], win_cnt[2], win_cnt[-1]))

        # Store win ratio for analysis
        self.win_ratio_list.append(win_ratio)
        return win_ratio

    def run(self):
        """
        Execute the training loop, including data collection, policy updates, and evaluation.
        """
        try:
            for i in range(self.game_batch_num):
                self.collect_selfplay_data(self.play_batch_size)
                print("batch i:{}, episode_len:{}".format(i + 1, self.episode_len))
                if len(self.data_buffer) > self.batch_size:
                    loss, entropy = self.policy_update()

                # Evaluate periodically and save the best model
                if (i + 1) % self.check_freq == 0:
                    print("Current self-play batch: {}".format(i + 1))
                    win_ratio = self.policy_evaluate()
                    self.policy_value_net.save_model('./current_policy.weights.h5')
                    if win_ratio > self.best_win_ratio:
                        print("New best policy!")
                        self.best_win_ratio = win_ratio
                        self.policy_value_net.save_model('./best_policy.weights.h5')
                        if self.best_win_ratio == 1.0 and self.pure_mcts_playout_num < 5000:
                            self.pure_mcts_playout_num += 1000
                            self.best_win_ratio = 0.0

            # Plot metrics after training
            self.plot_metrics()
        except KeyboardInterrupt:
            print('\nTraining interrupted.')
            self.plot_metrics()

    def plot_metrics(self):
        """
        Plot and save training metrics such as loss, entropy, KL divergence, and win ratio.
        """
        plt.figure(figsize=(18, 10))

        # Plot loss
        plt.subplot(3, 2, 1)
        plt.plot(self.batch_index_list, self.loss_list, label='Loss', color='blue')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Loss')
        plt.title('Loss over Training')
        plt.grid(True)
        plt.legend()

        # Plot entropy
        plt.subplot(3, 2, 2)
        plt.plot(self.batch_index_list, self.entropy_list, label='Entropy', color='orange')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Entropy')
        plt.title('Entropy over Training')
        plt.grid(True)
        plt.legend()

        # Plot KL divergence
        plt.subplot(3, 2, 3)
        plt.plot(self.batch_index_list, self.kl_list, label='KL Divergence', color='green')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('KL Divergence')
        plt.title('KL Divergence over Training')
        plt.grid(True)
        plt.legend()

        # Plot explained variance (old)
        plt.subplot(3, 2, 4)
        plt.plot(self.batch_index_list, self.explained_var_old_list, label='Explained Var Old', color='red')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Explained Var Old')
        plt.title('Explained Variance (Old) over Training')
        plt.grid(True)
        plt.legend()

        # Plot explained variance (new)
        plt.subplot(3, 2, 5)
        plt.plot(self.batch_index_list, self.explained_var_new_list, label='Explained Var New', color='purple')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Explained Var New')
        plt.title('Explained Variance (New) over Training')
        plt.grid(True)
        plt.legend()

        # Plot win ratio
        plt.subplot(3, 2, 6)
        x_vals = [self.check_freq * (i + 1) for i in range(len(self.win_ratio_list))]
        plt.plot(x_vals, self.win_ratio_list, label='Win Ratio', color='brown')
        plt.xlabel('Self-Play Batch')
        plt.ylabel('Win Ratio')
        plt.title('Win Ratio over Training')
        plt.grid(True)
        plt.legend()

        plt.tight_layout()
        plt.savefig('training_metrics.png')
        plt.show()
        print("Training metrics saved to 'training_metrics.png' and displayed.")

if __name__ == '__main__':
    training_pipeline = TrainPipeline()
    training_pipeline.run()
