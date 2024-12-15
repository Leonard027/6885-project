import numpy as np
import copy

def softmax(x):
    """Compute softmax probabilities for a given array."""
    exp_x = np.exp(x - np.max(x))
    return exp_x / np.sum(exp_x)

class TreeNode:
    """
    Represents a single node in the MCTS tree, tracking visit counts, values, and prior probabilities.
    """
    def __init__(self, parent, prior_p):
        self.parent = parent
        self.children = {}  # Maps actions to child nodes
        self.n_visits = 0
        self.Q = 0.0
        self.u = 0.0
        self.P = prior_p

    def expand(self, action_priors):
        """Expand the tree by creating child nodes for all given action-priors."""
        for action, prob in action_priors:
            if action not in self.children:
                self.children[action] = TreeNode(self, prob)

    def select(self, c_puct):
        """Select the action that maximizes Q + u."""
        return max(
            self.children.items(),
            key=lambda act_node: act_node[1].get_value(c_puct)
        )

    def update(self, leaf_value):
        """Update Q-value and visit count based on the leaf node's value."""
        self.n_visits += 1
        self.Q += (leaf_value - self.Q) / self.n_visits

    def update_recursive(self, leaf_value):
        """Recursively update all ancestor nodes with the leaf value."""
        if self.parent:
            self.parent.update_recursive(-leaf_value)
        self.update(leaf_value)

    def get_value(self, c_puct):
        """Compute the node's value using Q and u."""
        self.u = c_puct * self.P * np.sqrt(self.parent.n_visits) / (1 + self.n_visits)
        return self.Q + self.u

    def is_leaf(self):
        """Check if the node is a leaf (no children expanded)."""
        return len(self.children) == 0

    def is_root(self):
        """Check if the node is the root of the tree."""
        return self.parent is None

class MCTS:
    """
    Monte Carlo Tree Search implementation.
    """
    def __init__(self, policy_value_fn, c_puct=5, n_playout=10000):
        self.root = TreeNode(None, 1.0)
        self.policy = policy_value_fn
        self.c_puct = c_puct
        self.n_playout = n_playout

    def _playout(self, state):
        """Execute a single playout from root to leaf, updating the tree."""
        node = self.root
        while not node.is_leaf():
            action, node = node.select(self.c_puct)
            state.do_move(action)

        action_probs, leaf_value = self.policy(state)
        end, winner = state.game_end()
        if not end:
            node.expand(action_probs)
        else:
            leaf_value = 0 if winner == -1 else 1 if winner == state.get_current_player() else -1

        node.update_recursive(-leaf_value)

    def get_move_probs(self, state, temp=1e-3):
        """Run playouts and return actions and their probabilities."""
        for _ in range(self.n_playout):
            state_copy = copy.deepcopy(state)
            self._playout(state_copy)

        actions_visits = [(act, node.n_visits) for act, node in self.root.children.items()]
        actions, visits = zip(*actions_visits)
        action_probs = softmax((1.0 / temp) * np.log(np.array(visits) + 1e-10))

        return actions, action_probs

    def update_with_move(self, last_move):
        """Update the root node based on the last move."""
        if last_move in self.root.children:
            self.root = self.root.children[last_move]
            self.root.parent = None
        else:
            self.root = TreeNode(None, 1.0)

    def __str__(self):
        return "MCTS"

class MCTSPlayer:
    """
    An AI player using MCTS.
    """
    def __init__(self, policy_value_fn, c_puct=5, n_playout=2000, is_selfplay=False):
        self.mcts = MCTS(policy_value_fn, c_puct, n_playout)
        self.is_selfplay = is_selfplay
        self.player = None

    def set_player_ind(self, p):
        """Set the player index."""
        self.player = p

    def reset_player(self):
        """Reset the MCTS state."""
        self.mcts.update_with_move(-1)

    def get_action(self, board, temp=1e-3, return_prob=False):
        """Get the next action from the MCTS tree."""
        sensible_moves = board.availables
        move_probs = np.zeros(board.width * board.height)
        if sensible_moves:
            actions, probs = self.mcts.get_move_probs(board, temp)
            move_probs[list(actions)] = probs

            if self.is_selfplay:
                move = np.random.choice(
                    actions,
                    p=0.75 * probs + 0.25 * np.random.dirichlet(0.3 * np.ones(len(probs)))
                )
                self.mcts.update_with_move(move)
            else:
                move = np.random.choice(actions, p=probs)
                self.mcts.update_with_move(-1)

            if return_prob:
                return move, move_probs
            return move
        else:
            print("WARNING: No valid moves available.")
            return -1, move_probs

    def __str__(self):
        return f"MCTS Player {self.player}"
