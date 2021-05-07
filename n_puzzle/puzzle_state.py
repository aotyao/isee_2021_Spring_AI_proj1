import numpy as np
from enum import Enum
import copy


# Enum of operation in EightPuzzle problem
class Move(Enum):
    """
    The class of move operation
    NOTICE: The direction denotes the 'blank' space move
    """
    Up = 0
    Down = 1
    Left = 2
    Right = 3


# EightPuzzle state
class PuzzleState(object):
    """
    Class for state in EightPuzzle-Problem
    Attr:
        square_size: Chessboard size, e.g: In 8-puzzle problem, square_size = 3
        state: 'square_size' x 'square_size square', '-1' indicates the 'blank' block  (For 8-puzzle, state is a 3 x 3 array)
        g: The cost from initial state to current state
        h: The value of heuristic function
        pre_move:  The previous operation to get to current state
        pre_state: Parent state of this state
    """
    def __init__(self, square_size = 3):
        self.square_size = square_size
        self.state = None
        self.g = 0
        self.h = 0
        self.pre_move = None
        self.pre_state = None

        self.generate_state()

    def __eq__(self, other):
        return (self.state == other.state).all()       # all()方法，只要有1个零，null，‘’，返回0，没有返回1

    def blank_pos(self):
        """
        Find the 'blank' position of current state
        :return:
            row: 'blank' row index, '-1' indicates the current state may be invalid
            col: 'blank' col index, '-1' indicates the current state may be invalid
        """
        index = np.argwhere(self.state == -1)   # 返回state里面-1所在的坐标
        row = -1
        col = -1
        if index.shape[0] == 1:  # find blank   shape是读取矩阵的长度，shape[0]是矩阵第一维度的长度
            row = index[0][0]
            col = index[0][1]
        return row, col

    def num_pos(self, num):
        """
        Find the 'num' position of current state
        :return:
            row: 'num' row index, '-1' indicates the current state may be invalid
            col: 'num' col index, '-1' indicates the current state may be invalid
        """
        index = np.argwhere(self.state == num)
        row = -1
        col = -1
        if index.shape[0] == 1:  # find number
            row = index[0][0]
            col = index[0][1]
        return row, col

    def is_valid(self):
        """
        Check current state is valid or not (A valid state should have only one 'blank')
        :return:
            flag: boolean, True - valid state, False - invalid state
        """
        row, col = self.blank_pos()
        if row == -1 or col == -1:
            return False
        else:
            return True

    def clone(self):
        """
        Return the state's deepcopy
        :return:
        """
        return copy.deepcopy(self)

    def generate_state(self, random=False, seed=None):
        """
        Generate a new state
        :param random: True - generate state randomly, False - generate a normal state
        :param seed: Choose the seed of random, only used when random = True
        :return:
        """
        # arange左闭右开，reshape中-1表示不清楚几列，让计算机自己算
        self.state = np.arange(0, self.square_size ** 2).reshape(self.square_size, -1)
        self.state[self.state == 0] = -1  # Set blank

        if random:
            np.random.seed(seed)
            np.random.shuffle(self.state)   # 随机排列

    def display(self):
        """
        Print state
        :return:
        """
        print("----------------------")
        for i in range(self.state.shape[0]):
            # print("{}\t{}\t{}\t".format(self.state[i][0], self.state[i][1], self.state[i][2]))
            # print(self.state[i, :])
            for j in range(self.state.shape[1]):
                if j == self.state.shape[1] - 1:
                    print("{}\t".format(self.state[i][j]))
                else:
                    print("{}\t".format(self.state[i][j]), end='')
        print("----------------------\n")


def check_move(curr_state, move):
    """
    Check the operation 'move' can be performed on current state 'curr_state'
    :param curr_state: Current puzzle state
    :param move: Operation to be performed
    :return:
        valid_op: boolean, True - move is valid; False - move is invalid
        src_row: int, current blank row index
        src_col: int, current blank col index
        dst_row: int, future blank row index after move
        dst_col: int, future blank col index after move
    """
    # assert isinstance(move, Move)  # Check operation type
    assert curr_state.is_valid()

    if not isinstance(move, Move):
        move = Move(move)

    src_row, src_col = curr_state.blank_pos()
    dst_row, dst_col = src_row, src_col
    valid_op = False

    if move == Move.Up:  # Number moves up, blank moves down
        dst_row -= 1
    elif move == Move.Down:
        dst_row += 1
    elif move == Move.Left:
        dst_col -= 1
    elif move == Move.Right:
        dst_col += 1
    else:  # Invalid operation
        dst_row = -1
        dst_col = -1

    if dst_row < 0 or dst_row > curr_state.state.shape[0] - 1 or dst_col < 0 or dst_col > curr_state.state.shape[1] - 1:
        valid_op = False
    else:
        valid_op = True

    return valid_op, src_row, src_col, dst_row, dst_col


def once_move(curr_state, move):
    """
    Perform once move to current state
    :param curr_state:
    :param move:
    :return:
        valid_op: boolean, flag of this move is valid or not. True - valid move, False - invalid move
        next_state: EightPuzzleState, state after this move
    """
    valid_op, src_row, src_col, dst_row, dst_col = check_move(curr_state, move)

    next_state = curr_state.clone()

    if valid_op:
        it = next_state.state[dst_row][dst_col]
        next_state.state[dst_row][dst_col] = -1
        next_state.state[src_row][src_col] = it
        next_state.pre_state = curr_state
        next_state.pre_move = move
        return True, next_state
    else:
        return False, next_state


def check_state(src_state, dst_state):
    """
    Check current state is same as destination state
    :param src_state:
    :param dst_state:
    :return:
    """
    return (src_state.state == dst_state.state).all()


def run_moves(curr_state, dst_state, moves):
    """
    Perform list of move to current state, and check the final state is same as destination state or not
    Ideally, after we perform moves to current state, we will get a state same as the 'dst_state'
    :param curr_state: EightPuzzleState, current state
    :param dst_state: EightPuzzleState, destination state
    :param moves: List of Move
    :return:
        flag of moves: True - We can get 'dst_state' from 'curr_state' by 'moves'
    """
    pre_state = curr_state.clone()
    next_state = None

    for move in moves:
        valid_move, next_state = once_move(pre_state, move)

        if not valid_move:
            return False

        pre_state = next_state.clone()

    if check_state(next_state, dst_state):
        return True
    else:
        return False


def runs(curr_state, moves):
    """
    Perform list of move to current state, get the result state
    NOTICE: The invalid move operation would be ignored
    :param curr_state:
    :param moves:
    :return:
    """
    pre_state = curr_state.clone()
    next_state = None

    for move in moves:
        valid_move, next_state = once_move(pre_state, move)
        pre_state = next_state.clone()
    return next_state


def print_moves(init_state, moves):
    """
    While performing the list of move to current state, this function will also print how each move is performed
    :param init_state: The initial state
    :param moves: List of move
    :return:
    """
    print("Initial state")
    init_state.display()

    pre_state = init_state.clone()
    next_state = None

    for idx, move in enumerate(moves):
        if move == Move.Up:  # Number moves up, blank moves down
            print("{} th move. Goes up.".format(idx))
        elif move == Move.Down:
            print("{} th move. Goes down.".format(idx))
        elif move == Move.Left:
            print("{} th move. Goes left.".format(idx))
        elif move == Move.Right:
            print("{} th move. Goes right.".format(idx))
        else:  # Invalid operation
            print("{} th move. Invalid move: {}".format(idx, move))

        valid_move, next_state = once_move(pre_state, move)

        if not valid_move:
            print("Invalid move: {}, ignore".format(move))

        next_state.display()

        pre_state = next_state.clone()

    print("We get final state: ")
    next_state.display()


def generate_moves(move_num = 30):
    """
    Generate a list of move in a determined length randomly
    :param move_num:
    :return:
        move_list: list of move
    """
    move_dict = {}
    move_dict[0] = Move.Up
    move_dict[1] = Move.Down
    move_dict[2] = Move.Left
    move_dict[3] = Move.Right

    index_arr = np.random.randint(0, 4, move_num)
    index_list = list(index_arr)

    move_list = [move_dict[idx] for idx in index_list]

    return move_list


def convert_moves(moves):
    """
    Convert moves from int into Move type
    :param moves:
    :return:
    """
    if len(moves):
        if isinstance(moves[0], Move):
            return moves
        else:
            return [Move(move) for move in moves]
    else:
        return moves


"""
NOTICE:
1. init_state is a 3x3 numpy array, the "space" is indicated as -1, for example
    1 2 -1              1 2
    3 4 5   stands for  3 4 5
    6 7 8               6 7 8
2. moves contains directions that transform initial state to final state. Here
    0 stands for up
    1 stands for down
    2 stands for left
    3 stands for right
    We 
   There might be several ways to understand "moving up/down/left/right". Here we define
   that "moving up" means to move 'space' up, not move other numbers up. For example
    1 2 5                1 2 -1
    3 4 -1   move up =>  3 4 5
    6 7 8                6 7 8
   This definition is actually consistent with where your finger moves to
   when you are playing 8 puzzle game.
   
3. It's just a simple example of A-Star search. You can implement this function in your own design.  
"""


def astar_search_for_puzzle_problem(init_state, dst_state, method = 'manhattan'):
    """
    Use AStar-search to find the path from init_state to dst_state
    :param init_state:  Initial puzzle state
    :param dst_state:   Destination puzzle state
    :return:  All operations needed to be performed from init_state to dst_state
        moves: list of Move. e.g: move_list = [Move.Up, Move.Left, Move.Right, Move.Up]
    """

    # ------------------------自己写的函数，写外面的话还要import，所以写里面了------------------------------ #
    # --------------------------------------------------------------------------------------------- #

    def update_cost(curr_state, dst_state, method):
        """
        function    :update curr_state's g and h value
        para1 & 2   :current state and end state
        para3       :the method determining heuristic function
                        it can be
                        'euclidean':欧氏距离,每个数离目标点的欧式距离
                        'manhattan':曼哈顿距离，就是横纵坐标的距离和，block distance
                        'chebyshev':切比雪夫距离，横纵坐标距离更大的那个
                        'hamming'  :所有格子中，不在自己该在的位置上的数字个数
        """

        # 下面循环计算h
        h = 0
        for i in range(curr_state.state.shape[0]):
            for j in range(curr_state.state.shape[1]):
                # 这两层循环遍历每一个格点

                # 获取坐标
                curr_row, curr_col = curr_state.num_pos(curr_state.state[i][j])
                dst_row, dst_col = dst_state.num_pos(curr_state.state[i][j])
                x = abs(curr_row - dst_row)
                y = abs(curr_col - dst_col)

                if method == 'euclidean':  #
                    h = h + (x ** 2 + y ** 2) ** 0.5
                elif method == 'manhattan':
                    h = h + x + y
                elif method == 'chebyshev':
                    h = h + max(x, y)
                elif method == 'hamming':
                    if x != 0 or y != 0:
                        h = h + 1

        curr_state.h = h
        curr_state.g = curr_state.pre_state.g + 1
        return curr_state

    def find_front_node(open_list):
        # 返回列表中代价最小的节点
        index = 0
        cost = open_list[0].g + open_list[0].h
        for i in range(1, len(open_list)):
            cost_i = open_list[i].g + open_list[i].h
            if cost_i < cost:
                index = i
        return index, open_list[index]

    def get_path(curr_state):
        move_list = []
        state = curr_state
        # 依次向上寻找
        while state.pre_move != None:
            move_list.append(state.pre_move)
            state = state.pre_state
        # 最后倒置
        move_list.reverse()
        return move_list

    def expand_state(curr_state):
        childs = []
        for i in range(4):
            valid, next_state = once_move(curr_state, i)
            if valid:
                childs.append(next_state)
        return childs

    def state_in_list(state, list):
        in_list = False
        match_state = None
        for item in list:
            if item == state:
                in_list = True
                match_state = item
                break
        return in_list, match_state

    # ----------------------------------------------------------------------------------------#

    start_state = init_state.clone()
    end_state = dst_state.clone()

    open_list = []   # You can also use priority queue instead of list
    close_list = []

    move_list = []  # The operations from init_state to dst_state

    # Initial A-star
    open_list.append(start_state)

    while len(open_list) > 0:
        # Get best node from open_list
        curr_idx, curr_state = find_front_node(open_list)

        # Delete best node from open_list
        open_list.pop(curr_idx)

        # Add best node in close_list
        close_list.append(curr_state)

        # Check whether found solution
        if curr_state == dst_state:
            moves = get_path(curr_state)
            return moves

        # Expand node
        childs = expand_state(curr_state)

        for child_state in childs:

            # Explored node
            in_list, match_state = state_in_list(child_state, close_list)
            if in_list:
                continue

            # Assign cost to child state. You can also do this in Expand operation
            child_state = update_cost(child_state, dst_state, method)

            # Find a better state in open_list
            in_list, match_state = state_in_list(child_state, open_list)
            if in_list:
                if match_state.g + match_state.h > child_state.g + child_state.h:
                    match_state.g = child_state.g
                    match_state.h = child_state.h
                    match_state.pre_state = child_state.pre_state
                    match_state.pre_move = child_state.pre_move
                continue

            open_list.append(child_state)


