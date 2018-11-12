from pommerman import make, agents
from . import BaseAgent
from .. import MCTS
from .. import constants

class SmartAgent(BaseAgent):
    
    # values kept in SmartAgent's history board
    history_board_values = [
        constants.Item.Passage.value,
        constants.Item.Rigid.value,
        constants.Item.Wood.value,
        constants.Item.ExtraBomb.value,
        constants.Item.IncrRange.value,
        constants.Item.Kick.value
    ]

    def __init__(self, training_env=None):
        super().__init__()
        self.modelled_env = training_env or \
                            make('PommeTeamCompetition-v0', agent_list=[agents.SimpleAgent() for _ in range(4)])   # 4 players

        self.history = None

    def build_history(self, agent_obs):
        '''Update history by incrementing timestep by 1, and
        by new observation received from current timestep'''

        def _get_history_board(board):
            # Board values stored in history
            history_board = np.copy(board)
            np.putmask(history_board, history_board not in self.history_board_values, 0)
            return history_board

        def _get_explosion_range(row, col):
            indices = {
                'up': ([row - i, col] for i in range(1, self.history['bomb_blast_strength'][row, co])),
                'down': ([row + i, col] for i in range(self.history['bomb_blast_strength'][row, co])),
                'left': ([row, col - i] for i in range(1, self.history['bomb_blast_strength'][row, co])),
                'right': ([row, col + i] for i in range(1, self.history['bomb_blast_strength'][row, co]))
            }
            return indices

        # Note: all three 11x11 boards are numpy arrays
        if self.history is None:
            self.history = {
                'bomb_life' = np.array(agent_obs['bomb_life']),
                'bomb_blast_strength' = np.array(agent_obs['bomb_blast_strength']),
                'board' = _get_history_board(agent_obs['board'])
            }
            return

        # Update history by incrementing timestep by 1
        bomb_life = self.history['bomb_life']
        bomb_blast_strength = self.history['bomb_blast_strength']
        board = self.history['board']

        exploded_map = np.zeros_like(board)
        original_bomb_life = np.copy(bomb_life)

        # Decrease bomb_life by 1
        np.putmask(bomb_life, bomb_life > 0, bomb_life - 1)

        rest_bombs = np.logical_and(bomb_life, np.ones_like(bomb_life))
        exploding_bombs = np.logical_xor(bomb_life, original_bomb_life)
        has_new_explosions = any(exploding_bombs)

        while has_new_explosions:
            has_new_explosions = False
            pos_row, pos_col = exploding_bombs.non_zero()
            for i in range(np.count_nonzero(exploding_bombs)):
                for _, indices in _get_explosion_range(pos_row[i], pos_col[i]):
                    for r, c in indices:
                        if not all(
                            [r >= 0, c >= 0, r < constants.BOARD_SIZE, c < constants.BOARD_SIZE])
                            break
                        if board[r][c] == constants.Item.Rigid.value:
                            break
                        exploded_map[r][c] = 1
                        if board[r][c] == constants.Item.Wood.value:
                            break
            pos_row, pos_col = rest_bombs.non_zero()
            exploding_bombs = np.zeros_like(exploding_bombs)
            for i in range(np.count_nonzero(rest_bombs)):
                if exploded_map[pos_row[i]][pos_col[i]]:
                    has_new_explosions = True
                    exploding_bombs[pos_row[i]][pos_col[i]] = True
                    rest_bombs[pos_row[i]][pos_col[i]] = False

        # Elementwise multiplication
        erase_bomb = np.logical_not(exploded_map)
        self.history['board'] = np.multiply(board, erase_bomb)
        self.history['bomb_blast_strength'] = np.multiply(bomb_blast_strength, erase_bomb)

        # Update history by agent observation.
        # If board from observation has fog value, do nothing &
        # keep original updated history.
        # Otherwise, overwrite history by observation.
        obs_board = agent_obs['board']
        visible_row, visible_col = np.where(obs_board != 5)
        for r, c in zip(visible_row, visible_col):
            board[r, c] = obs_board[r, c] if obs_board[r, c] in self.history_board_values else 0
            bomb_life[r, c] = agent_obs['bomb_life'][r, c]
            bomb_blast_strength[r, c] = agent_obs['bomb_blast_strength'][r, c]

    def act(self, obs, action_space):

        # TODO: left some pseudo-code here, but general
        # TODO: idea here is simple
        pi = MCTS.perform_MCTS(self.modelled_env, self.agent_id)
        self.history = self.build_history(obs)
        training_pool.append((self.history, pi))
        return random.sample(list(range(6)), p=pi)
