import os
from game import Board, Game
from mcts_alphaZero import MCTSPlayer
from policy_value_net_tensorflow import PolicyValueNet  # TensorFlow 2.x compatible

class HumanPlayer:
    """
    Represents a human player in the Gomoku game.
    """

    def __init__(self):
        self.player = None

    def set_player_ind(self, player_index):
        """
        Assign the player index (1 or 2).

        Args:
            player_index (int): Identifier for the human player.
        """
        self.player = player_index

    def get_action(self, board):
        """
        Prompt the human player for their move.

        Args:
            board (Board): The current game board.

        Returns:
            int: The chosen move index.
        """
        try:
            user_input = input("Enter your move (row,column): ")
            if isinstance(user_input, str):
                location = [int(coord) for coord in user_input.split(",")]
            move = board.location_to_move(location)
        except Exception as error:
            print(f"Error in input: {error}")
            move = -1

        if move == -1 or move not in board.availables:
            print("Invalid move. Please try again.")
            return self.get_action(board)
        return move

    def __str__(self):
        return f"Human Player {self.player}"

def run():
    """
    Execute a game of Human vs AI using AlphaZero's policy-value model.
    """
    # Game configuration
    win_condition = 5  # Number of consecutive pieces required to win
    board_width, board_height = 10, 10  # Dimensions of the board
    model_path = 'best_policy.weights.h5'  # Path to the trained model

    # Verify the existence of the trained model file
    if not os.path.exists(model_path):
        print(f"Trained model '{model_path}' is missing. Please ensure it is available.")
        return

    try:
        # Initialize board and game logic
        board = Board(width=board_width, height=board_height, n_in_row=win_condition)
        game = Game(board)

        # Load the policy-value network and create the AI player
        policy_net = PolicyValueNet(board_width, board_height, model_file=model_path)
        policy_net.build(input_shape=(None, board_height, board_width, 4))  # Prepare the model
        ai_player = MCTSPlayer(policy_net.policy_value_fn,
                                c_puct=5,
                                n_playout=400)  # Adjust playouts for performance

        # Create the human player
        human_player = HumanPlayer()

        # Start the game (AI goes first by default, adjust start_player for human first)
        game.start_play(human_player, ai_player, start_player=1, is_shown=1)
    except KeyboardInterrupt:
        print('\nGame terminated by user.')

if __name__ == '__main__':
    run()
