import datetime
import json
import math
import pathlib

import numpy
import torch

from .abstract_game import AbstractGame


class MuZeroConfig:
    def __init__(self):
        # fmt: off
        # 更多資訊請參考: https://github.com/werner-duvaud/muzero-general/wiki/Hyperparameter-Optimization
        
        # 優先從 config.json 載入所有配置
        # 以下為備用默認值，當 config.json 不存在時使用
        self._set_defaults()
        self._load_from_json()
        self._update_dependent_params()
        # fmt: on
    
    def _set_defaults(self):
        """設置所有參數的默認值"""
        self.seed = 0
        self.max_num_gpus = 1
        
        # 遊戲設定
        self.board_size = 9
        self.stacked_observations = 0
        self.muzero_player = "random"
        self.opponent = "random"
        
        # 自我對弈
        self.num_workers = 2
        self.selfplay_on_gpu = True
        self.max_moves = 81
        self.num_simulations = 50
        self.discount = 1
        self.temperature_threshold = None
        self.root_dirichlet_alpha = 0.3
        self.root_exploration_fraction = 0.25
        self.pb_c_base = 19652
        self.pb_c_init = 1.25
        
        # 神經網路
        self.network = "resnet"
        self.support_size = 10
        self.downsample = False
        self.blocks = 4
        self.channels = 64
        self.reduced_channels_reward = 2
        self.reduced_channels_value = 2
        self.reduced_channels_policy = 4
        self.resnet_fc_reward_layers = [64]
        self.resnet_fc_value_layers = [64]
        self.resnet_fc_policy_layers = [64]
        self.encoding_size = 32
        self.fc_representation_layers = []
        self.fc_dynamics_layers = [64]
        self.fc_reward_layers = [64]
        self.fc_value_layers = []
        self.fc_policy_layers = []
        
        # 訓練
        self.save_model = True
        self.training_steps = 10000
        self.batch_size = 32
        self.checkpoint_interval = 50
        self.value_loss_weight = 1
        self.optimizer = "Adam"
        self.weight_decay = 1e-4
        self.momentum = 0.9
        self.lr_init = 0.002
        self.lr_decay_rate = 0.9
        self.lr_decay_steps = 10000
        
        # 重放緩衝區
        self.replay_buffer_size = 10000
        self.num_unroll_steps = 81
        self.td_steps = 81
        self.PER = True
        self.PER_alpha = 0.5
        self.use_last_model_value = False
        self.reanalyse_on_gpu = True
        
        # 早停
        self.early_stop_patience = None
        self.early_stop_threshold = 0.0001
        
        # 比例調整
        self.self_play_delay = 0
        self.training_delay = 0
        self.ratio = 1

    def _load_from_json(self):
        """從 config.json 載入配置參數（如果文件存在）"""
        config_path = pathlib.Path(__file__).resolve().parents[1] / "config.json"
        if config_path.exists():
            try:
                with open(config_path, 'r', encoding='utf-8') as f:
                    config_data = json.load(f)
                
                # 更新配置參數（忽略以 _ 開頭的註解欄位）
                for key, value in config_data.items():
                    if not key.startswith('_'):
                        setattr(self, key, value)
                
                print(f"✓ 已從 config.json 載入配置")
            except Exception as e:
                print(f"✗ 載入 config.json 時出錯: {e}")
                print("  使用默認配置")
        else:
            print(f"✗ 找不到 config.json，使用默認配置")
    
    def _update_dependent_params(self):
        """更新依賴於其他參數的配置"""
        # 根據棋盤大小計算
        self.observation_shape = (3, self.board_size, self.board_size)
        self.action_space = list(range(self.board_size * self.board_size))
        self.players = list(range(2))  # 固定為2人遊戲
        
        # 自動設置路徑
        board_size_folder = f"{self.board_size}x{self.board_size}"
        self.results_path = pathlib.Path(__file__).resolve().parents[1] / "results" / pathlib.Path(__file__).stem / board_size_folder / datetime.datetime.now().strftime("%Y-%m-%d--%H-%M-%S")
        
        # GPU 設置
        self.train_on_gpu = torch.cuda.is_available()
    
    def visit_softmax_temperature_fn(self, trained_steps):
        """
        用於改變訪問計數分佈的參數，以確保隨著訓練進展，動作選擇變得更加貪婪。
        值越小，最佳動作 (即訪問計數最高的動作) 被選中的可能性越大。

        Returns:
            正浮點數。
        """
        if trained_steps < 0.5 * self.training_steps:
            return 1.0
        elif trained_steps < 0.75 * self.training_steps:
            return 0.5
        else:
            return 0.25


class Game(AbstractGame):
    """
    遊戲包裝器。
    """

    def __init__(self, seed=None):
        # 從配置中獲取棋盤大小
        config = MuZeroConfig()
        self.env = Gomoku(board_size=config.board_size)

    def step(self, action):
        """
        對遊戲應用動作。

        Args:
            action : 要執行的 action_space 中的動作。

        Returns:
            新的觀察、獎勵和表示遊戲是否結束的布林值。
        """
        observation, reward, done = self.env.step(action)
        return observation, reward, done

    def to_play(self):
        """
        返回當前玩家。

        Returns:
            當前玩家，應該是配置中 players 列表的元素。
        """
        return self.env.to_play()

    def legal_actions(self):
        """
        應該返回每回合的合法動作，如果不可用，可以返回整個動作空間。
        在每回合，遊戲必須能夠處理返回動作中的一個。

        對於計算合法移動耗時過長的複雜遊戲，可以將合法動作定義為等於動作空間，
        但如果動作不合法則返回負獎勵。

        Returns:
            整數陣列，動作空間的子集。
        """
        return self.env.legal_actions()

    def reset(self):
        """
        重置遊戲以開始新遊戲。

        Returns:
            遊戲的初始觀察。
        """
        return self.env.reset()

    def close(self):
        """
        正確關閉遊戲。
        """
        pass

    def render(self):
        """
        顯示遊戲觀察。
        """
        self.env.render()
        input("按 Enter 鍵繼續下一步 ")

    def human_to_action(self):
        """
        對於多人遊戲，詢問用戶合法動作並返回對應的動作編號。

        Returns:
            來自動作空間的整數。
        """
        valid = False
        while not valid:
            valid, action = self.env.human_input_to_action()
        return action

    def action_to_string(self, action):
        """
        將動作編號轉換為表示該動作的字串。
        Args:
            action_number: 來自動作空間的整數。
        Returns:
            表示該動作的字串。
        """
        return self.env.action_to_human_input(action)


class Gomoku:
    def __init__(self, board_size=9):
        self.board_size = board_size
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        self.board_markers = [
            chr(x) for x in range(ord("A"), ord("A") + self.board_size)
        ]

    def to_play(self):
        return 0 if self.player == 1 else 1

    def reset(self):
        self.board = numpy.zeros((self.board_size, self.board_size), dtype="int32")
        self.player = 1
        return self.get_observation()

    def step(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        self.board[x][y] = self.player

        done = self.is_finished()

        reward = 1 if done else 0

        self.player *= -1

        return self.get_observation(), reward, done

    def get_observation(self):
        board_player1 = numpy.where(self.board == 1, 1.0, 0.0)
        board_player2 = numpy.where(self.board == -1, 1.0, 0.0)
        board_to_play = numpy.full((self.board_size, self.board_size), self.player, dtype="int32")
        return numpy.array([board_player1, board_player2, board_to_play])

    def legal_actions(self):
        legal = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    legal.append(i * self.board_size + j)
        return legal

    def is_finished(self):
        has_legal_actions = False
        directions = ((1, -1), (1, 0), (1, 1), (0, 1))
        for i in range(self.board_size):
            for j in range(self.board_size):
                # if no stone is on the position, don't need to consider this position
                if self.board[i][j] == 0:
                    has_legal_actions = True
                    continue
                # value-value at a coord, i-row, j-col
                player = self.board[i][j]
                # check if there exist 5 in a line
                for d in directions:
                    x, y = i, j
                    count = 0
                    for _ in range(5):
                        if (x not in range(self.board_size)) or (
                            y not in range(self.board_size)
                        ):
                            break
                        if self.board[x][y] != player:
                            break
                        x += d[0]
                        y += d[1]
                        count += 1
                        # if 5 in a line, store positions of all stones, return value
                        if count == 5:
                            return True
        return not has_legal_actions

    def render(self):
        marker = "  "
        for i in range(self.board_size):
            marker = marker + self.board_markers[i] + " "
        print(marker)
        for row in range(self.board_size):
            print(chr(ord("A") + row), end=" ")
            for col in range(self.board_size):
                ch = self.board[row][col]
                if ch == 0:
                    print(".", end=" ")
                elif ch == 1:
                    print("X", end=" ")
                elif ch == -1:
                    print("O", end=" ")
            print()

    def human_input_to_action(self):
        human_input = input("Enter an action: ")
        if (
            len(human_input) == 2
            and human_input[0] in self.board_markers
            and human_input[1] in self.board_markers
        ):
            x = ord(human_input[0]) - 65
            y = ord(human_input[1]) - 65
            if self.board[x][y] == 0:
                return True, x * self.board_size + y
        return False, -1

    def action_to_human_input(self, action):
        x = math.floor(action / self.board_size)
        y = action % self.board_size
        x = chr(x + 65)
        y = chr(y + 65)
        return x + y
