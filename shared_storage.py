import copy

import ray
import torch


@ray.remote
class SharedStorage:
    """
    Class which run in a dedicated thread to store the network weights and some information.
    """

    def __init__(self, checkpoint, config):
        self.config = config
        self.current_checkpoint = copy.deepcopy(checkpoint)
        self.best_total_reward = float('-inf')  # 追蹤最優模型的獎勵
        self.last_save_num_games = 0  # 上次保存模型時的遊戲數量

    def save_checkpoint(self, path=None):
        if not path:
            path = self.config.results_path / "model.checkpoint"

        torch.save(self.current_checkpoint, path)
    
    def save_checkpoint_by_games(self):
        """每100盤保存一次模型"""
        num_games = self.current_checkpoint.get("num_played_games", 0)
        
        # 檢查是否需要保存（每100盤）
        if num_games > 0 and num_games // 100 > self.last_save_num_games // 100:
            checkpoint_path = self.config.results_path / f"model_games_{num_games}.checkpoint"
            torch.save(self.current_checkpoint, checkpoint_path)
            self.last_save_num_games = num_games
            print(f"\n已保存模型: {checkpoint_path.name}")
    
    def save_best_model(self):
        """保存最優模型"""
        total_reward = self.current_checkpoint.get("total_reward", float('-inf'))
        
        # 如果當前模型表現更好，保存為最優模型
        if total_reward > self.best_total_reward:
            self.best_total_reward = total_reward
            best_model_path = self.config.results_path / "best_model.checkpoint"
            torch.save(self.current_checkpoint, best_model_path)
            print(f"\n🎯 新的最優模型! 獎勵: {total_reward:.2f} - 已保存至 best_model.checkpoint")
            return True
        return False

    def get_checkpoint(self):
        return copy.deepcopy(self.current_checkpoint)

    def get_info(self, keys):
        if isinstance(keys, str):
            return self.current_checkpoint[keys]
        elif isinstance(keys, list):
            return {key: self.current_checkpoint[key] for key in keys}
        else:
            raise TypeError

    def set_info(self, keys, values=None):
        if isinstance(keys, str) and values is not None:
            self.current_checkpoint[keys] = values
        elif isinstance(keys, dict):
            self.current_checkpoint.update(keys)
        else:
            raise TypeError
