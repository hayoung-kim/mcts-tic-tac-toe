from env import env as game
import numpy as np
from VanilaMCTS import VanilaMCTS
import time


env = game.GameState()
state_size, win_mark = game.Return_BoardParams()
board_shape = [state_size, state_size]
game_board = np.zeros(board_shape, dtype=int)

game_end = False
whos_turn = {0: 'o', 1: 'x'}
mcts_player = 'x'
current_player = 'o'

while not game_end:
    action_onehot = 0
    if current_player == mcts_player:
        mcts = VanilaMCTS(n_iterations=1500, depth=15, exploration_constant=100, game_board=game_board, player=current_player)
        best_action, best_q, depth = mcts.solve()
        action_onehot = np.zeros([state_size**2])
        action_onehot[best_action] = 1
        calculate_mcts = False

    # take action and get game info
    game_board, check_valid_position, win_index, turn = env.step(action_onehot)
    current_player = whos_turn[turn]

    if win_index != 0:
        game_board = np.zeros(board_shape, dtype=int)
        time.sleep(0.1)
