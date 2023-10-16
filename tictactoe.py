import gymnasium as gym
from gymnasium import spaces
import numpy as np
from enum import Enum
from functools import reduce
import random

class TicTacToeEnv(gym.Env):
    DRAW = 0
    PLAYER_CIRCLE = 1
    PLAYER_CROSS = 2
    PLAYING = 3

    SIZE = 3
    SQUARES = SIZE * SIZE
    
    def __init__(self, board = None):
        self.reset(board)
        self.observation_space = spaces.Box(low=0, high=2, shape=(self.SQUARES,), dtype=np.int64)
        self.action_space = spaces.Discrete(self.SQUARES)
        self.reward_range = (0, 1)

    def reset(self, board = None):
        if (board is None):
            board = [0] * 9
        self.board : list[int] = board
        self.turn : int = 0
        return (self.board_str(), {})

    def step(self, value: int) :
        state = self.play_move(value)
        done = state != self.PLAYING

        reward = -1
        if state == self.PLAYER_CIRCLE or state == self.PLAYER_CROSS:
            reward = 10
        if state == self.DRAW:
            reward = -3
        
        if not done:
            state = self.play_move(random.choice(self.legal_moves()))
            done = state != self.PLAYING

            if state == self.PLAYER_CIRCLE or state == self.PLAYER_CROSS:
                reward = -10
            if state == self.DRAW:
                reward = -3

        # Not returning truncated, since we assume that the agent will use legal_moves()
        return self.board_str(), reward, done, False, {}
    
    def render(self):
        print('-----'.join(
                [
                    f"\n{'|'.join([str(self.board[i*3 + j]) for j in range(self.SIZE)])}\n"
                    for i in range(self.SIZE)
                ]
            )
        )
                
    def board_str(self):
        return "".join(list(map(str, self.board)))
    
    def play_move(self, move: int) -> int:
        assert move in self.legal_moves()
        color = 0
        if self.turn % 2 == 0:
            color = self.PLAYER_CIRCLE
        else:
            color = self.PLAYER_CROSS
        
        self.board[move] = color
        self.turn += 1
        return self.current_game_state()

    def current_game_state(self) -> int:
        for player_color in [self.PLAYER_CIRCLE, self.PLAYER_CROSS]:
            # Check horizontally
            for j in range(0, self.SQUARES, self.SIZE):
                if reduce(lambda acc, current_color : acc and current_color == player_color,  [self.board[i] for i in range(j, j+3)], True) == True:
                    return player_color
                
            # Check vertically
            for j in range(0, 3):
                if reduce(lambda acc, current_color : acc and current_color == player_color,  [self.board[i] for i in range(j, j+3*self.SIZE, self.SIZE)], True) == True:
                    return player_color
            
            if self.board[0] == player_color and self.board[4] == player_color and self.board[8] == player_color:
                return player_color
            if self.board[2] == player_color and self.board[4] == player_color and self.board[6] == player_color:
                return player_color

        for i in range(self.SQUARES):
            if self.board[i] == 0:
                # Playing
                return self.PLAYING
        # Board is filled => Draw
        return self.DRAW
    
    def legal_moves(self):
        return [move for move, state in enumerate(self.board) if state == 0]