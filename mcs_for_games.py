import numpy as np
from collections import defaultdict

class TicTacToeMove:
    def __init__(self, x_coordinate, y_coordinate, value):
        self.x_coordinate = x_coordinate
        self.y_coordinate = y_coordinate
        self.value = value

    def __repr__(self):
        return "x:" + str(self.x_coordinate) + " y:" + str(self.y_coordinate) + " v:" + str(self.value)


class TicTacToeGameState:
    x = 1
    o = -1

    def __init__(self, state, next_to_move=1):
        if len(state.shape) != 2 or state.shape[0] != state.shape[1]:
            raise ValueError("Please play on 2D square board")
        self.board = state
        self.board_size = state.shape[0]
        self.next_to_move = next_to_move

    def game_result(self):
        rowsum = np.sum(self.board, 0)
        colsum = np.sum(self.board, 1)
        diag_sum_tl = self.board.trace()
        diag_sum_tr = self.board[::-1].trace()

        # Check if any row, column, or diagonal has a sum equal to the board size,
        # indicating a win for 'X'.
        if any(rowsum == self.board_size) or any(colsum == self.board_size) or diag_sum_tl == self.board_size or diag_sum_tr == self.board_size:
            return 1.
        # Check if any row, column, or diagonal has a sum equal to the negative board size,
        # indicating a win for 'O'.
        elif any(rowsum == -self.board_size) or any(colsum == -self.board_size) or diag_sum_tl == -self.board_size or diag_sum_tr == -self.board_size:
            return -1.
        # Check if the board is full (no empty cells), indicating a tie.
        elif np.all(self.board != 0):
            return 0.
        else:
            return None

    def is_game_over(self):
        # Check if the game is over based on the game result.
        return self.game_result() is not None

    def is_move_legal(self, move):
        if move.value != self.next_to_move:
            return False
        x_in_range = move.x_coordinate < self.board_size and move.x_coordinate >= 0
        if not x_in_range:
            return False
        y_in_range = move.y_coordinate < self.board_size and move.y_coordinate >= 0
        if not y_in_range:
            return False
        return self.board[move.x_coordinate, move.y_coordinate] == 0

    def move(self, move):
        if not self.is_move_legal(move):
            raise ValueError("move " + move + " on board " + self.board + " is not legal")
        new_board = np.copy(self.board)
        new_board[move.x_coordinate, move.y_coordinate] = move.value
        next_to_move = TicTacToeGameState.o if self.next_to_move == TicTacToeGameState.x else TicTacToeGameState.x
        return TicTacToeGameState(new_board, next_to_move)

    def get_legal_actions(self):
        # Get the indices of empty cells in the board and create TicTacToeMove objects
        # for each empty cell with the current player's value.
        indices = np.where(self.board == 0)
        return [TicTacToeMove(coords[0], coords[1], self.next_to_move) for coords in list(zip(indices[0], indices[1]))]


class MonteCarloTreeSearchNode:

    def __init__(self, state: TicTacToeGameState, parent=None):
        self._number_of_visits = 0.
        self._results = defaultdict(int)
        self.state = state
        self.parent = parent
        self.children = []

    @property
    def untried_actions(self):
        if not hasattr(self, '_untried_actions'):
            self._untried_actions = self.state.get_legal_actions()
        return self._untried_actions

    @property
    def q(self):
        # Calculate the "Q" value for the node, representing the quality of the node's move.
        wins = self._results[self.parent.state.next_to_move]
        loses = self._results[-1 * self.parent.state.next_to_move]
        return wins - loses

    @property
    def n(self):
        # Return the number of visits to the node.
        return self._number_of_visits

    def expand(self):
        # Select an untried action, create a new node with the resulting state,
        # and add it as a child to the current node.
        action = self.untried_actions.pop()
        next_state = self.state.move(action)
        child_node = MonteCarloTreeSearchNode(next_state, parent=self)
        self.children.append(child_node)
        return child_node

    def is_terminal_node(self):
        # Check if the node represents a terminal state (game over).
        return self.state.is_game_over()

    def rollout(self):
        # Perform a random rollout from the current state until a terminal state is reached
        # and return the result (1 for 'X' win, -1 for 'O' win, 0 for a tie).
        current_rollout_state = self.state
        while not current_rollout_state.is_game_over():
            possible_moves = current_rollout_state.get_legal_actions()
            action = self.rollout_policy(possible_moves)
            current_rollout_state = current_rollout_state.move(action)
        return current_rollout_state.game_result()

    def backpropagate(self, result):
        # Update the visit count and result count of the node and its ancestors.
        self._number_of_visits += 1.
        self._results[result] += 1.
        if self.parent:
            self.parent.backpropagate(result)

    def is_fully_expanded(self):
        # Check if all the node's actions have been tried.
        return len(self.untried_actions) == 0

    def best_child(self, c_param=1.4):
        # Select the best child node based on the UCB1 formula, taking into account
        # the exploration and exploitation trade-off.
        choices_weights = [
            (c.q / (c.n)) + c_param * np.sqrt((2 * np.log(self.n) / (c.n)))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def rollout_policy(self, possible_moves):
        # Select a random move from a list of possible moves.
        return possible_moves[np.random.randint(len(possible_moves))]


class MonteCarloTreeSearch:
    def __init__(self, node: MonteCarloTreeSearchNode):
        self.root = node

    def best_action(self, simulations_number):
        # Perform a specified number of Monte Carlo Tree Search iterations
        # and return the best action (move) to take.
        for _ in range(0, simulations_number):
            v = self.tree_policy()
            reward = v.rollout()
            v.backpropagate(reward)
        return self.root.best_child(c_param=0.)

    def tree_policy(self):
        # Select the next node to visit during the tree traversal phase.
        current_node = self.root
        while not current_node.is_terminal_node():
            if not current_node.is_fully_expanded():
                return current_node.expand()
            else:
                current_node = current_node.best_child()
        return current_node


def init():
    # Initialize the game state with an empty board.
    state = np.zeros((3, 3))
    initial_board_state = TicTacToeGameState(state=state, next_to_move=1)
    return initial_board_state


def graphics(board):
    # Prints the current Tic Tac Toe board state in a user-friendly format.
    # :param board: numpy array representing the board
    for i in range(3):
        print("")
        print("{0:3}".format(i).center(8) + "|", end='')
        for j in range(3):
            if board[i][j] == 0:
                print('_'.center(8), end='')
            if board[i][j] == 1:
                print('X'.center(8), end='')
            if board[i][j] == -1:
                print('O'.center(8), end='')
    print("")
    print("______________________________")


def get_action(state):
    try:
        location = input("Your move: ")
        if isinstance(location, str):
            location = [int(n, 10) for n in location.split(",")]
        if len(location) != 2:
            return -1
        x = location[0]
        y = location[1]
        move = TicTacToeMove(x, y, 1)
    except Exception as e:
        move = -1
    if move == -1 or not state.is_move_legal(move):
        print("Invalid move")
        move = get_action(state)
    return move


def judge(state):
    game_result = state.game_result()
    if game_result is not None:
        if game_result == 1.0:
            print("You win!")
        elif game_result == 0.0:
            print("Tie!")
        elif game_result == -1.0:
            print("You lose!")
        return True
    else:
        return False
#Code which gives Our turn as First turn

current_state = init()
current_board = current_state.board
graphics(current_board)

while True:
    move = get_action(current_state)
    current_state = current_state.move(move)
    current_board = current_state.board
    graphics(current_board)
    if judge(current_state):
        break

    root = MonteCarloTreeSearchNode(state=current_state, parent=None)
    mcts = MonteCarloTreeSearch(root)
    best_node = mcts.best_action(1000)
    current_state = best_node.state
    current_board = current_state.board
    graphics(current_board)
    if judge(current_state):
        break

  #Code which gives Agents turn as First turn

# current_state, current_board = init()
# graphics(current_board)
# while True:
#     move1 = get_action(current_state)
#     current_state = current_state.move(move1)
#     current_board = current_state.board
#     graphics(current_board)
#
#     board_state = TicTacToeGameState(state=current_board, next_to_move=1)
#     root = MonteCarloTreeSearchNode(state=board_state, parent=None)
#     mcts = MonteCarloTreeSearch(root)
#     best_node = mcts.best_action(1000)
#     current_state = best_node.state
#     current_board = current_state.board
#     graphics(current_board)
#     if judge(current_state) == 1:
#         break
#     elif judge(current_state) == -1:
#         continue
