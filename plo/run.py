import os
import sys
import pickle
assert os.path.isdir("./simple_distributed_rl/srl/")  # srlがここにある想定です
sys.path.insert(0, "./simple_distributed_rl/")
from typing import Any, List, Optional
import numpy as np

import srl
from srl import runner

from srl.base.rl.algorithms.discrete_action import DiscreteActionConfig, DiscreteActionWorker
from srl.base.rl.base import RLParameter

from srl.rl.models import alphazero as alphazero_model
from srl.rl.models import mlp
from srl.utils import common

# --- env & algorithm load
from srl.algorithms import alphazero  # isort: skip

import buttle_line_env

def main():
   
    env_config = srl.EnvConfig("ButtleLineEnv")
    rl_config = alphazero.Config(
        num_simulations=10, 
        memory_warmup_size=100, 
        batch_size=64
    )

    config = runner.Config(env_config, rl_config)

    # --- ベストプレイヤーを保存するファイル
    best_parameter_path = "_best_player6.dat"
    rl_config.parameter_path = best_parameter_path  # ベストプレイヤーから毎回始める
    best_rl_config = rl_config.copy()  # ベストプレイヤー用の設定
    remote_memory = None  # 学習を通して経験を保存

    # --- 学習ループ
    for i in range(3):

        # 学習用プレイヤー(None)とベストプレイヤー(best_rl_config)で対戦し学習
        config.players = [None, best_rl_config]
        parameter, remote_memory, _ = runner.train(
            config,
            max_episodes=30000,
            remote_memory=remote_memory,
            disable_trainer=True,
        )

        # 学習後のプレイヤーを評価する(ベストプレイヤーと対戦)
        config.players = [None, best_rl_config]
        rewards = runner.evaluate(
            config,
            parameter,
            max_episodes=3000,
            shuffle_player=True,
        )
        rewards = np.mean(rewards, axis=0)
        print(f"------------ {i} evaluate: {rewards}")

        # 学習用プレイヤー(None)の勝率が55%以上なら置きかえる
        # 閾値は、勝ち:1、負け-1なので0.05
        if rewards[0] - rewards[1] > 0.002:
            print("UPDATE!")
            parameter.save(best_parameter_path)

if __name__ == "__main__":
    #sys.stdout = open("./log/run_10.log", "w")
    
    main()
  
    #sys.stdout.close()
    #sys.stdout = sys.__stdout__
