import torch
import pathlib

# 檢查最新的模型
checkpoint_path = pathlib.Path("results/gomoku/2025-12-14--12-17-53/model.checkpoint")

if checkpoint_path.exists():
    cp = torch.load(checkpoint_path, map_location='cpu')
    
    print("=" * 50)
    print("模型訓練統計")
    print("=" * 50)
    print(f"訓練步數: {cp['training_step']}")
    print(f"已玩遊戲: {cp['num_played_games']}")
    print(f"總 Loss: {cp['total_loss']:.2f}")
    print(f"Value Loss: {cp['value_loss']:.2f}")
    print(f"Policy Loss: {cp['policy_loss']:.2f}")
    print(f"Reward Loss: {cp['reward_loss']:.2f}")
    print(f"學習率: {cp['lr']}")
    print(f"平均獎勵: {cp['total_reward']:.4f}")
    print(f"平均 Episode 長度: {cp['episode_length']:.2f}")
else:
    print("找不到模型文件")
