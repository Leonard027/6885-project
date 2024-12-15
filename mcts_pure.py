import numpy as np
import copy
from operator import itemgetter

def rollout_policy_fn(board):
    """
    Fast and simple policy function for rollouts.
    Selects moves randomly during the simulation phase.
    """
    probabilities = np.random.rand(len(board.availables))
    return zip(board.availables, probabilities)

def policy_value_fn(board):
    """
    Basic policy-value function for pure MCTS:
    - Assigns uniform probabilities to all available moves.
    - Returns a zero value (no heuristic evaluation).
    """
    probabilities = np.ones(len(board.availables)) / len(board.availables)
    return zip(board.availables, probabilities), 0.0

class TreeNode:
    """
    Node in the MCTS search tree.

    Attributes:
        parent (TreeNode): Reference to the parent node.
        children (dict): Maps actions to child nodes.
        n_visits (int): Number of times this node has been visited.
        Q (float): Average value of this node.
        u (float): Exploration bonus.
        P (float): Prior probability of choosing this action.
    """
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}
        self.n_visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.P = prior_p

    def expand(self, action_priors):
        """
        Add new child nodes for available actions.
        action_priors: List of (action, probability) pairs.
        """
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """
        Select the action with the highest Q + u score.
        Returns:
            (action, next_node): The chosen action and corresponding node.
        """
        return max(
            self.children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def update(self, leaf_value):
        """
        Update this node's Q-value and visit count using the evaluation.
        leaf_value: Scalar in [-1,1] representing the leaf evaluation.
        """
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """
        Propagate updates back to all ancestor nodes.
        The parent's perspective alternates with the child's.
        """
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """
        Compute the node's value using Q and u.
        Q + u = Q + c_puct * P * sqrt(parent visits) / (1 + visits)
        """
        self.u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u

    def is_leaf(self):
        """Determine if this node is a leaf (no expanded children)."""
        return len(self.children) == 0

    def is_root(self):
        """Check if this node is the root."""
        return self.parent is None

class MCTS:
    """
    Pure Monte Carlo Tree Search without neural network guidance.
    Evaluates states using random rollouts.
    """
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        """
        Args:
            policy_value_fn: Function providing action probabilities and state evaluation.
            c_puct (float): Exploration-exploitation tradeoff parameter.
            n_playout (int): Number of simulations per move.
        """
        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def _playout(self, state):
        """
        Perform a single MCTS simulation from root to leaf.
        Uses rollout for leaf evaluation and backpropagates the results.
        """
        node = self.root
        # Selection
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            state.do_move(action)

        # Expansion
        action_probs, _ = self.policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)

        # Rollout and evaluation
        leaf_value = self._evaluate_rollout(state)

        # Backpropagation
        node.update_recursive(-leaf_value)

    def _evaluate_rollout(self, state, limit=1000):
        """
        Perform a rollout until the game ends or a move limit is reached.
        Returns:
            +1 if the current player wins,
            -1 if the opponent wins,
            0 for a tie.
        """
        player = state.get_current_player()
        for _ in range(limit):
            end, winner = state.game_end()
            if end:
                break
            action_probs = rollout_policy_fn(state)
            best_action = max(action_probs, key=itemgetter(1))[0]
            state.do_move(best_action)
        else:
            print("WARNING: Rollout reached move limit")

        if winner == -1:
            return 0.0  # Tie
        return 1.0 if winner == player else -1.0

    def get_move(self, state):
        """
        Execute all playouts and select the most visited action.
        """
        for _ in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        # Choose the action with the highest visit count
        return max(self.root.children.items(), key=lambda act_node: act_node[1].n_visits)[0]

    def update_with_move(self, last_move):
        """
        Move the tree's root to the node corresponding to last_move.
        """
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer:
    """
    Player controlled by the pure MCTS algorithm.
    Used for benchmarking or as a baseline AI opponent.
    """
    def __init__(self, c_puct=5, n_playout=2000):
        """
        Args:
            c_puct (float): Exploration parameter.
            n_playout (int): Number of simulations per turn.
        """
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.player = None

    def set_player_ind(self, p):
        """
        Set the player's identifier (1 or 2).
        """
        self.player = p

    def reset_player(self):
        """
        Reset the MCTS tree for a new game.
        """
        self.mcts.update_with_move(-1)

    def get_action(self, board):
        """
        Get the next move from MCTS based on the current board state.
        """
        available_moves = board.availables
        if available_moves:
            move = self.mcts.get_move(board)
            self.mcts.update_with_move(-1)  # Reset MCTS for the next turn
            return move
        else:
            print("WARNING: No available moves. The board is full.")
            return -1

    def __str__(self):
        return f"MCTS Player {self.player}"
