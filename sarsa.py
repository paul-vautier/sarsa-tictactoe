import json

import numpy as np
from tqdm import tqdm

from tictactoe import TicTacToeEnv
from utils import rendering, plot_stats

from typing import Callable, Optional

env = TicTacToeEnv()

env.reset()
frame = env.render()

_action_values: dict[bytes, np.ndarray] = {}

def action_values(state: np.ndarray) -> np.ndarray:
    _hash: bytes = state.data.tobytes()
    if _hash not in _action_values:
        _action_values[_hash] = np.zeros((env.SQUARES,))
    return _action_values[_hash]


def policy(state: np.ndarray, epsilon=1/4) -> int:
    '''
    Epsilon-greedy policy. Responsible for the exploration of the space. 
    Random action is chosen and returned `epsilon`% of the time (`epsilon` \in [0,1]).
    :param state: ndarray of the state
    :param epsilon: float in range [0,1]
    :return: picked action according to the defined policy
    '''
    av: np.ndarray = action_values(state)
    
    # `square` = (i,) ==> `square[0]` = i
    if np.random.random() < epsilon:
        legals = np.array([-100. if state[square[0]] != 0 else 1 for square, _ in np.ndenumerate(av)])
    else:
        legals = np.array([-100. if state[square[0]] != 0 else value for square, value in np.ndenumerate(av)])

    choice: int = np.random.choice(np.flatnonzero(legals == legals.max()))
    
    return choice


def n_step_sarsa(
    policy: Callable[[np.ndarray, float], int],
    action_values: Callable[[np.ndarray], np.ndarray],
    episodes: int,
    alpha=0.1,
    gamma=0.99,
    epsilon=0.2,
    n=8,
):
    stats = {'Returns': []}

    for ep in tqdm(range(1, episodes+1)):

        state, _ = env.reset()
        done = False 
        action = policy(state)
        ep_return = 0
        transitions = []
        t = 0

        while t - n < len(transitions):
            if not done:
                next_state, reward, done, _, _ = env.step(action)
                ep_return += reward
                next_action = policy(next_state, epsilon) if not done else None
                transitions.append([next_state, reward, next_action])
            
            if t >= n:
                # G = R1 + gamma * R2 + .. + gamma**t * Rn + Q(sn, an)

                # Q(sn, an)
                G = action_values(next_state)[next_action] if not done else 0.

                for state_t, reward_t, action_t in reversed(transitions[t-n:]):
                    G = reward_t + gamma * G
                qsa = action_values(state_t)[action_t]
                action_values(state_t)[action_t] = qsa + alpha * (G - qsa)
            
            t += 1
            state = next_state
            action = next_action

        stats['Returns'].append(ep_return)
    return stats

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
    
class NumpyDecoder(json.JSONDecoder):
    def decode(self, obj):
        if isinstance(obj, list):
            return np.array(obj)
        return json.JSONDecoder.decode(self, obj)
    
def save():
    with open("out.json", "w") as outfile:
        json_dump = json.dumps(_action_values,cls=NumpyEncoder)
        outfile.write(json_dump)

def load():
    with open("out.json", "r") as r:
        _action_values = {k: np.array(v) for k, v in json.load(r,cls=NumpyDecoder).items()}
        print(len(_action_values))


plot_stats(n_step_sarsa(policy, action_values, episodes=200000, epsilon=.3))
rendering(env, policy, episodes=1)
 


