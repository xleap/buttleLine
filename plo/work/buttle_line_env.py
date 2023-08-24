import os
import sys
import pickle
import copy
import random
import itertools
from dataclasses import dataclass
from typing import Any, List, Tuple, cast

abs_path = os.path.dirname(os.path.abspath(__file__))
assert os.path.isdir(abs_path + "/simple_distributed_rl/srl/")  # srlがここにある想定です
sys.path.insert(0, abs_path + "/simple_distributed_rl/")
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import srl
from srl import runner
from srl.base.env import registration
from srl.base.env.genre import TurnBase2Player
from srl.base.spaces.space import SpaceBase
from srl.base.define import EnvObservationTypes
from srl.base.env.env_run import EnvRun, SpaceBase
from srl.base.rl.config import RLConfig
from srl.base.rl.processor import Processor
from srl.base.spaces import ArrayDiscreteSpace, DiscreteSpace
from srl.algorithms import alphazero

import action as action_ai

registration.register(
    id="ButtleLineEnv",
    entry_point=__name__ + ":ButtleLineEnv",
    kwargs={},
)

class ButtleLineEnv(TurnBase2Player):
    W: int = 9
    H: int = 6
    C: int = 7
    action_ai = action_ai.AI()
    state = None
    step_rewards = np.array([0.0, 0.0])
    done = False

    def __init__(self):
        self._next_player_index = 0
        
    @property
    def action_space(self) -> DiscreteSpace:
        return DiscreteSpace(self.W * self.C)

    @property
    def observation_space(self) -> ArrayDiscreteSpace:
        return ArrayDiscreteSpace(self.W * self.H, 0, 69)

    @property
    def observation_type(self) -> EnvObservationTypes:
        return EnvObservationTypes.DISCRETE

    @property
    def max_episode_steps(self) -> int:
        return 60

    @property
    def next_player_index(self) -> int:
        return self._next_player_index

    def call_reset(self) -> Tuple[List[int], dict]:
        self.action = 0
        self.layout = [
            [
                [],[],[],[],[],[],[],[],[]
            ],
            [
                [],[],[],[],[],[],[],[],[]
            ]
        ]
        self.flags = [{'owner': None}, {'owner': None}, {'owner': None}, 
                      {'owner': None}, {'owner': None}, {'owner': None}, 
                      {'owner': None}, {'owner': None}, {'owner': None}]
        self.hands = [[], []]
        self._next_player_index = 0
        self.turn = 1
        self.stock = []
        self.field = [0] * (self.W * self.H)

        for i in range(10, 70, 1):
            self.stock.append({'color': i//10, 'number': i%10+1})
            random.shuffle(self.stock)
        
        self.hands[0] = self.stock[0:7]
        self.hands[1] = self.stock[7:14]

        self.invalid_actions = [
            self._calc_invalid_actions(0),
            self._calc_invalid_actions(1),
        ]

        return self.field, {}

    def call_step(self, action) -> Tuple[List[int], float, float, bool, dict]:
        
        self.action = action

        w = action%9
        # 置かれている枚数
        l = len(self.layout[self._next_player_index][w])

        # 反則手判定
        if l >= 3:
            if self._next_player_index == 0:
                self.step_rewards[0] -= 1.0
                self.step_rewards[1] += 1.0
            else:
                self.step_rewards[0] += 1.0
                self.step_rewards[1] -= 1.0
            
            return self.field, self.step_rewards[0], self.step_rewards[1], True, {}
        
        enemy_player = 1 if self._next_player_index == 0 else 0
        my_player = 0 if self._next_player_index == 0 else 1

        # 更新
        self._step(action)

        self.checkFlag(action%9, my_player, enemy_player, self.turn)

        # 加点計算
        add_score = add_num_sum = add = 0.0

        if l == 2:
            # 役ができたら加点
            add_score, add_num_sum = self.add(w, my_player)
            add = 0.1 * add_score + 0.01 * add_num_sum
        elif l == 1:
            # 自分が作れる最大の役を予測
            add_score, add_num_sum = self.estimate(w, my_player, enemy_player)
            add = 0.1 * (add_score - 1)

        # 勝敗判定
        winner = self.winner()

        # 報酬設定
        if winner == 0:
            self.step_rewards[0] += 1.0
            self.step_rewards[1] -= 1.0
            self.done = True
        elif winner == 1:
            self.step_rewards[0] -= 1.0
            self.step_rewards[1] += 1.0
            self.done = True

        # 手番交代
        self._next_player_index = enemy_player
        self.turn += 1

        if self.turn % 2 == 0:
            self.step_rewards[0] += add
        else:
            self.step_rewards[1] += add
        
        return self.field, self.step_rewards[0], self.step_rewards[1], self.done, {}

    def backup(self) -> Any:
        return pickle.dumps(
            [
                self._next_player_index,
                self.W,
                self.H,
                self.field,
                self.invalid_actions,
                self.stock,
                self.hands,
                self.layout,
                self.flags,
                self.turn,
                self.step_rewards,
            ]
        )

    def restore(self, data: Any) -> None:
        d = pickle.loads(data)
        self._next_player_index = d[0]
        self.W = d[1]
        self.H = d[2]
        self.field = d[3]
        self.invalid_actions = d[4]
        self.stock = d[5]
        self.hands = d[6]
        self.layout = d[7]
        self.flags = d[8]
        self.turn = d[9]
        self.step_rewards = d[10]
    
    def _calc_invalid_actions(self, player_index) -> List[int]:
        
        dirs_list = []
        index = 0

        # カード
        for y in range(self.C):

            # 場
            for x in range(self.W):
                
                check = True

                if y < len(self.hands[player_index]):
                    check = self.action_ai.check(self.layout[player_index], self.layout[(player_index+1)%2], self.flags, self.hands[player_index], x, y)

                    if not check:
                        dirs_list.append(index)
                else:
                    dirs_list.append(index)
    
                index += 1
        
        return dirs_list

    def _step(self, action):

        # 置く場所
        w = action%9
        # 選択したカードの番号
        c = action//9
        
        # 置かれている枚数
        l = len(self.layout[self._next_player_index][w])
        # カード情報
        select_card = self.hands[self._next_player_index][c]
        # 配置した場所
        point = (self._next_player_index * self.W * 3) + w + (l * self.W)
        # --- update
        self.field[point] = (select_card['color'] * 10) + (select_card['number'] - 1)

        # 選択したカードを場に出す
        card = self.hands[self._next_player_index].pop(c)
        self.layout[self._next_player_index][w].append(card)

        # ドロー
        if self.turn < 47:
            self.hands[self._next_player_index].append(self.stock[self.turn+13])
      
        # 置ける場所を更新
        self.invalid_actions = [
            self._calc_invalid_actions(0),
            self._calc_invalid_actions(1),
        ]
 
    def get_invalid_actions(self) -> List[int]:
       return self.invalid_actions[self._next_player_index]
    
    def checkFlag(self, layout_num, my_player, enemy_player, turn=1):

        # 場のカード枚数
        my_player_cards = self.layout[my_player][layout_num]
        enemy_player_cards = self.layout[enemy_player][layout_num]

        # お互いの役が確定した場合
        if len(my_player_cards) == 3 and len(enemy_player_cards) == 3:
            
            # それぞれの役を取得 
            sScore, sSum = self.action_ai.score(my_player_cards)
            eScore, eSum = self.action_ai.score(enemy_player_cards)

            if sScore > eScore:
                self.flags[layout_num]['owner'] = my_player
            elif sScore < eScore:
                self.flags[layout_num]['owner'] = enemy_player
            else:
                if sSum > eSum:
                    self.flags[layout_num]['owner'] = my_player
                elif sSum < eSum:
                    self.flags[layout_num]['owner'] = enemy_player
                else:
                    if turn%2 == 1:
                        self.flags[layout_num]['owner'] = enemy_player
                    else:
                        self.flags[layout_num]['owner'] = my_player
        
        for judge in range(self.W):

            # 場のカード枚数
            my_player_judge = self.layout[my_player][judge]
            enemy_player_judge = self.layout[enemy_player][judge]

            # 片方のみ役が確定している場合の判定
            if (len(my_player_judge) == 3 and len(enemy_player_judge) == 2) or (len(my_player_judge) == 2 and len(enemy_player_judge) == 3):
                
                # 予測するプレイヤー
                ePlayer = None
                # 確定しているプレイヤー
                lPlayer = None

                if len(my_player_judge) == 3:
                    ePlayer = enemy_player
                    lPlayer = my_player
                else:
                    ePlayer = my_player
                    lPlayer = enemy_player
    
                # 最大（予測）の役
                eScore, eSum = self.action_ai.judge(self.layout[ePlayer], self.layout[lPlayer], self.hands[ePlayer], judge)
                
                # 確定の役
                lScore, lSum = self.action_ai.score(self.layout[lPlayer][judge])

                if lScore > eScore:
                    self.flags[judge]['owner'] = lPlayer
                elif lScore == eScore and lSum >= eSum:
                    self.flags[judge]['owner'] = lPlayer

    def winner(self):
        
        winner = None
        s,e = 0,0
        for i in range(len(self.flags)):

            # 3連続チェック
            if self.flags[i]['owner']:
                if i < 7:
                    if self.flags[i]['owner'] == self.flags[i+1]['owner'] and self.flags[i+1]['owner'] == self.flags[i+2]['owner']:
                        return self.flags[i]['owner']

            if self.flags[i]['owner'] == 0:
                s += 1
            elif self.flags[i]['owner'] == 1:
                e += 1
        
        if s >= 5:
            winner = 0
        elif e >= 5:
            winner = 1

        return winner
    
    def add(self, w, my_player):
    
        # 出そうとしている役のスコア
        cards = copy.deepcopy(self.layout[my_player][w])

        return self.action_ai.score(cards)
         
    # 自分の最大の役を予測
    def estimate(self, w, my_player, enemy_player):

        lStock = []
        eStock = copy.deepcopy(self.stock)

        for l in range(9):
            lStock.extend(copy.deepcopy(self.layout[my_player][l]))
            lStock.extend(copy.deepcopy(self.layout[enemy_player][l]))
        
        for ls in lStock:
            eStock = list(itertools.filterfalse(lambda x: x['color'] == ls['color'] and x['number'] == ls['number'], eStock))

        # 自分の最大の役を予測
        eScore = eSum = 0

        for x in range(len(eStock) - 2):
            # 場の状態
            eLayout = copy.deepcopy(self.layout[my_player][w])
            eLayout.append(eStock[x])
           
            score, sum = self.action_ai.score(eLayout)

            if score > eScore:
                eScore = score
                eSum = sum
        
        return eScore, eSum

class LayerProcessor(Processor):

    def preprocess_observation_space(
        self,
        env_observation_space: SpaceBase,
        env_observation_type: EnvObservationTypes,
        env: EnvRun,
        rl_config: RLConfig,
    ) -> Tuple[SpaceBase, EnvObservationTypes]:
        _env = cast(ButtleLineEnv, env.get_original_env())
        return _env.observation_space, _env.observation_type

    def preprocess_observation(self, observation: np.ndarray, env: ButtleLineEnv) -> np.ndarray:
        _env = cast(ButtleLineEnv, env.get_original_env())
        _field = np.zeros(_env.H * _env.W)
        return _field

class alpha():
    
    env_config = srl.EnvConfig("ButtleLineEnv")
    rl_config = alphazero.Config(
        num_simulations=10, 
        memory_warmup_size=100, 
        batch_size=64
    )

    best_parameter_path = abs_path + "/_best_player4.dat"
    rl_config.parameter_path = best_parameter_path  # ベストプレイヤーから毎回始める
    rl_config.processors = [LayerProcessor()]
    config = runner.Config(env_config, rl_config)
    parameter = config.make_parameter()
    parameter.load(best_parameter_path)

    # 学習結果(頭悪い)から手を選択
    def operation_alpha(self, layout, other_layout, flags, hand, other_hand_length, stock_length, play_first):

        foul_count = 0

        while True:

            env = ButtleLineEnv()
            worker = self.config.make_worker_player(self.rl_config, self.parameter)
            worker.on_reset(env)

            env.call_reset()
            env.flags = copy.deepcopy(flags)
            env.turn = 46 - stock_length
            env._next_player_index = (stock_length + 1) % 2
            
            player = 0 if play_first else 1
            other = 1 if play_first else 0
        
            env.layout[player] = copy.deepcopy(layout)
            env.layout[other] = copy.deepcopy(other_layout)   
            env.hands[player] = copy.deepcopy(hand)
        
            env.invalid_actions = [
                env._calc_invalid_actions(0),
                env._calc_invalid_actions(1),
            ]
            
            try:
                action = worker.policy(env)
            except:
                print('リトライ', file=sys.stderr) 
                continue
            
            # 置く場所
            choice = action%9
            # 選択したカードの番号
            num = action//9

            # 反則手(3枚以上置かれている、またはフラッグ獲得済)
            if(len(env.layout[player][choice]) > 2 or env.flags[choice]['owner']):
                
                # 20回連続はゲーム側の問題のため続行
                if(foul_count < 20):
                    foul_count += 1
                    print('再抽選{}回目'.format(foul_count), file=sys.stderr) 
                    continue
                else:
                    print('最終抽選', file=sys.stderr) 

                    layout = env.layout[player]
                    hand = env.hands[player]
                    score = sum = 0
                    
                    for c in range(len(hand)):
                        for w in range(len(layout)):
                            if env.flags[w]['owner'] is None and len(layout[w]) < 3:

                                eScore = eSum = 0

                                if  len(layout[w]) == 2:
                                    cards = copy.deepcopy(layout[w])
                                    cards.append(hand[c])
                                    eScore, eSum = env.action_ai.score(cards)

                                # 期待値が一番高いものを選択（勝てるとは限らない）
                                if (score == 0 and sum == 0) or eScore > score or (eScore == score and eSum > sum):
                                    score = eScore
                                    sum = eSum
                                    num = c
                                    choice = w

            return num, choice    