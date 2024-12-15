import numpy as np

class Board:
    """
    Represents the Gomoku game board, defining its attributes and operations.
    
    Attributes:
        width (int): The number of columns on the board.
        height (int): The number of rows on the board.
        n_in_row (int): The number of consecutive stones required to win.
        states (dict): Tracks the board state as a mapping of {position: player}.
        players (list): A list of player identifiers, default is [1, 2].
        current_player (int): Identifier of the player whose turn it is.
        availables (list): A list of valid moves (available positions).
        last_move (int): The last move executed on the board.
    """

    def __init__(self, **kwargs):
        self.width = int(kwargs.get('width', 6))
        self.height = int(kwargs.get('height', 6))
        self.n_in_row = int(kwargs.get('n_in_row', 4))
        self.players = [1, 2]
        self.states = {}
        self.availables = []
        self.current_player = self.players[0]
        self.last_move = -1

    def init_board(self, start_player=0):
        """
        Initialize the board for a new game.

        Args:
            start_player (int): Index of the player to start the game (0 or 1).
        """
        if self.width < self.n_in_row or self.height < self.n_in_row:
            raise ValueError(f"Board dimensions must be at least {self.n_in_row}.")

        self.current_player = self.players[start_player]
        self.availables = list(range(self.width * self.height))
        self.states = {}
        self.last_move = -1

    def move_to_location(self, move):
        """
        Convert a move index to a board location (row, column).

        Args:
            move (int): The position index of the move.

        Returns:
            list: [row, column] representing the location.
        """
        return [move // self.width, move % self.width]

    def location_to_move(self, location):
        """
        Convert a board location to a move index.

        Args:
            location (list): A list [row, column] representing the position.

        Returns:
            int: The position index, or -1 if invalid.
        """
        if len(location) != 2:
            return -1
        row, col = location
        move = row * self.width + col
        return move if move in range(self.width * self.height) else -1

    def current_state(self):
        """
        Get the current state of the board from the current player's perspective.

        Returns:
            np.ndarray: A 4-channel representation of the board state.
        """
        state = np.zeros((4, self.width, self.height))
        if self.states:
            moves, players = np.array(list(self.states.keys())), np.array(list(self.states.values()))
            current_moves = moves[players == self.current_player]
            opponent_moves = moves[players != self.current_player]
            state[0][current_moves // self.width, current_moves % self.height] = 1.0
            state[1][opponent_moves // self.width, opponent_moves % self.height] = 1.0

            if self.last_move != -1:
                state[2][self.last_move // self.width, self.last_move % self.height] = 1.0

        if len(self.states) % 2 == 0:
            state[3][:, :] = 1.0
        return state[:, ::-1, :]

    def do_move(self, move):
        """
        Execute a move on the board and update the current player.

        Args:
            move (int): The position index of the move.
        """
        self.states[move] = self.current_player
        self.availables.remove(move)
        self.current_player = self.players[1] if self.current_player == self.players[0] else self.players[0]
        self.last_move = move

    def has_a_winner(self):
        """
        Determine if there is a winner on the board.

        Returns:
            tuple: (bool, int) indicating whether a winner exists and the winner's ID.
        """
        n = self.n_in_row
        for move, player in self.states.items():
            row, col = divmod(move, self.width)

            directions = [(0, 1), (1, 0), (1, 1), (1, -1)]
            for dr, dc in directions:
                count = 1
                for i in range(1, n):
                    r, c = row + dr * i, col + dc * i
                    if self.states.get(r * self.width + c) == player:
                        count += 1
                    else:
                        break
                for i in range(1, n):
                    r, c = row - dr * i, col - dc * i
                    if self.states.get(r * self.width + c) == player:
                        count += 1
                    else:
                        break
                if count >= n:
                    return True, player

        return False, -1

    def game_end(self):
        """
        Check if the game has ended.

        Returns:
            tuple: (bool, int) indicating if the game ended and the winner's ID or -1 if tie.
        """
        win, winner = self.has_a_winner()
        if win:
            return True, winner
        if not self.availables:
            return True, -1
        return False, -1

    def get_current_player(self):
        """
        Get the identifier of the current player.

        Returns:
            int: Current player's ID (1 or 2).
        """
        return self.current_player


class Game:
    """
    Manages the Gomoku gameplay between two players.
    """

    def __init__(self, board):
        self.board = board

    def graphic(self, board, player1, player2):
        """
        Display the current board state.

        Args:
            board (Board): The game board instance.
            player1 (int): Player 1's ID.
            player2 (int): Player 2's ID.
        """
        print(f"Player {player1}: X", f"Player {player2}: O")
        print("\n   ", ' '.join([f"{x:2}" for x in range(board.width)]))
        for i in range(board.height - 1, -1, -1):
            row = [board.states.get(i * board.width + j, -1) for j in range(board.width)]
            symbols = ['X' if x == player1 else 'O' if x == player2 else '_' for x in row]
            print(f"{i:2} ", ' '.join(symbols))

    def start_play(self, player1, player2, start_player=0, is_shown=1):
        """
        Start a game between two players.

        Args:
            player1: Player 1 (implements get_action and set_player_ind).
            player2: Player 2 (implements get_action and set_player_ind).
            start_player (int): Starting player index (0 or 1).
            is_shown (bool): Whether to display the board during play.

        Returns:
            int: The winner's ID or -1 if the game ends in a tie.
        """
        self.board.init_board(start_player)
        players = {self.board.players[0]: player1, self.board.players[1]: player2}
        player1.set_player_ind(self.board.players[0])
        player2.set_player_ind(self.board.players[1])

        if is_shown:
            self.graphic(self.board, player1.player, player2.player)

        while True:
            current_player = self.board.get_current_player()
            player = players[current_player]
            move = player.get_action(self.board)
            self.board.do_move(move)
            if is_shown:
                self.graphic(self.board, player1.player, player2.player)
            end, winner = self.board.game_end()
            if end:
                if is_shown:
                    print(f"Game over. Winner: {'Tie' if winner == -1 else winner}")
                return winner
