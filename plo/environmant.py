# -*- coding: utf-8 -*-
#! /usr/bin/env python3

import json
import random
import copy
import itertools
from itertools import combinations
import action
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

import buttle_line_env

class Flag():

     def __init__(self):
        self.owner = None 

class Card():

    def __init__(self, i , j):
        self.color = i
        self.number = j

class Environment():
    metadata = {'render.modes': ['human']}

    def __init__(self, player=0, turn=1):
        
        self.layout = [
            [
                [],[],[],[],[],[],[],[],[]
            ],
            [
                [],[],[],[],[],[],[],[],[]
            ]
        ]
        self.player = player
        self.turn = turn
        self.flags = [Flag(), Flag(), Flag(), Flag(), Flag(), Flag(), Flag(), Flag(), Flag()]
        self.hands = [
                        [],
                        []
                    ]
        self.stock = []

        for i in range(10, 70, 1):
            self.stock.append(Card(i//10, i%10+1))
            random.shuffle(self.stock)
        
        self.hands[0] = self.stock[0:7]
        self.hands[1] = self.stock[7:14]
        #print('{}| '.format("stock"), end='')

    def reset(self):
        self.layout = [
            [
                [],[],[],[],[],[],[],[],[]
            ],
            [
                [],[],[],[],[],[],[],[],[]
            ]
        ]
        self.player = 0
        self.turn = 1
        self.flags = [Flag(), Flag(), Flag(), Flag(), Flag(), Flag(), Flag(), Flag(), Flag()]
        self.hands = [
                        [],
                        []
                    ]
        self.stock = []

        for i in range(10, 70, 1):
            self.stock.append(Card(i//10, i%10+1))
            random.shuffle(self.stock)
        
        self.hands[0] = self.stock[0:7]
        self.hands[1] = self.stock[7:14]

    def print(self):
        
        length = len(self.layout)
        print('\n')
        print('{}| '.format("flags"), end='')

        for j in range(len(self.flags)):
            print('\t{} '.format(self.flags[j].owner), end='')
        
        print('\n')
        print('{}| \n'.format("e.hand"), end='')

        for i in range(len(self.hands[1])):
            print(json.dumps({'color':self.hands[1][i].color, 'num':self.hands[1][i].number}))
                
        for i in range(length):
            player = length-i-1
            print('\n {}| \n'.format(player), end='')
            
            for l in range(len(self.layout[player])):
                for card in self.layout[player][l]:
                    print(' {} '.format(json.dumps({'from':l, 'color':card.color, 'num':card.number}), end=''))
                    
        print('\n')
        print('{}| \n'.format("s.hand"), end='')

        for i in range(len(self.hands[0])):
            print(json.dumps({'color':self.hands[0][i].color, 'num':self.hands[0][i].number}))
        print('\n')
        
    def choice(self, player=0, turn=1, a=0, b=0):
        
        self.check(a, b, player)
        
        layout = self.layout[player]

        # カード選択
        card = self.hands[player].pop(b)
        layout[a].append(card)
    
    def check(self, a, b, player):
        # 勝敗がついていない、かつ3枚おかれていない
        if not self.flags[a].owner and len(self.layout[player][a]) < 3:

            # 負け確は置かない
            if not self.winnerCheck(a, b, player):
                return False

        return True
    
    def winnerCheck(self, a, b, player):
        eScore = 0
        eMax = 0
        if len(self.layout[player][a]) == 2 and len(self.layout[(player+1)%2][a]) == 3:
            #eScore, eMax = self.judge(player, a)
            # 確定している手
            eScore, eMax = self.judge((player+1)%2, a)

            # 出そうとしている役
            cards = copy.deepcopy(self.layout[player][a])
            cards.append(self.hands[player][b])
            lScore, lMax = self.score(cards)

            if lScore > eScore:
                return True
            elif lScore == eScore and lMax > eMax:
                return True

        return False

    def judge(self, ePlayer, judge):
        # 予測する場のカード
        cards = copy.deepcopy(self.layout[ePlayer][judge])

        # 場と手札から相手の最大の役を予測
        #lStock = copy.deepcopy(self.hands[ePlayer])
        lStock = []
        lStock.extend(copy.deepcopy(self.hands[(ePlayer+1)%2]))
        eStock = copy.deepcopy(self.stock)

        for r in range(9):
            lStock.extend(copy.deepcopy(self.layout[0][r]))
            lStock.extend(copy.deepcopy(self.layout[1][r]))
        
        for ls in lStock:
            eStock = list(itertools.filterfalse(lambda x: x.color == ls.color and x.number == ls.number, eStock))

        # 場と手札から相手の最大の役を予測
        eScore = 0
        eMax = 0

        #for x in range(len(eStock)+len(cards)-2):
        for x in combinations(eStock, 3-len(cards)):
            # 場の状態
            layout = copy.deepcopy(cards)
            #layout.append(eStock[x])

            for i in range(len(x)):
                layout.append(x[i])
                        
            score, max = self.score(layout)

            if score > eScore:
                eScore = score
                eMax = max
            elif score == eScore and max > eMax:
                eMax = max
        
        return eScore, eMax
    
    def checkFlag(self, a, player, turn=1):
        
        if not self.flags[a].owner:
            # 場のカード枚数
            s = self.layout[0][a]
            e = self.layout[1][a]

            # 山札を取得
            #rStock = copy.deepcopy(self.stock[turn + 14:])
            #for i in range(len(rStock)):
            #    print('山札：{}'.format(json.dumps({'color':rStock[i].color, 'num':rStock[i].number})))
            
            # お互いの役が確定した場合
            if len(s) == 3 and len(e) == 3:
                
                # それぞれの役を取得 
                sScore, sMax = self.score(s)
                eScore, eMax = self.score(e)

                if sScore > eScore:
                    self.flags[a].owner = 0
                elif sScore < eScore:
                    self.flags[a].owner = 1
                else:
                    if sMax > eMax:
                        self.flags[a].owner = 0
                    elif sMax < eMax:
                        self.flags[a].owner = 1
                    else:
                        self.flags[a].owner = (self.player+1)%2
            else:
                # 片方のみ役が確定している場合の判定
                for judge in range(9):
                    # 場のカード枚数
                    sJudge = self.layout[0][judge]
                    eJudge = self.layout[1][judge]

                    if (len(sJudge) == 3 and len(eJudge) < 3) or (len(sJudge) < 3 and len(eJudge) == 3):
                        
                        # 予測するプレイヤー
                        ePlayer = None
                        # 確定しているプレイヤー
                        lPlayer = None

                        if len(sJudge) == 3:
                            ePlayer = 1
                            lPlayer = 0
                        else:
                            ePlayer = 0
                            lPlayer = 1
            
                        # 最大（予測）の役
                        eScore, eMax = self.judge(ePlayer, judge)
                        
                        # 確定の役
                        lScore, lMax = self.score(self.layout[lPlayer][judge])

                        if lScore > eScore:
                            self.flags[judge].owner = lPlayer
                        elif lScore == eScore and lMax >= eMax:
                            self.flags[judge].owner = lPlayer

    def isThree(self, cards):
        return cards[0].number == cards[1].number and cards[1].number == cards[2].number
    
    def isFlush(self, cards):
        return cards[0].color == cards[1].color and cards[1].color == cards[2].color
    
    def isStraight(self, numList):
        return numList[2] - numList[1] == 1 and numList[1] - numList[0] == 1

    def score(self, cards):
        # 数字の配列を生成(昇順)
        numList = [i.number for i in cards]
        numList.sort()
        max = numList[2]

        if self.isFlush(cards) and self.isStraight(numList):
            score = 5
        elif self.isThree(cards):
            score = 4
        elif self.isFlush(cards):
            score = 3
        elif self.isStraight(numList):
            score = 2
        else:
            score = 1
        return score, max
    
    def winner(self):
        s,e = 0,0
        for i in range(0, len(self.flags), 1):
            if self.flags[i].owner == 0:
                s = s+1
            elif self.flags[i].owner == 1:
                e = e+1
        
        return s,e

def get_action(layout, other_layout, flags, hand, other_hand_length, stock_length, play_first):

    def get_legal_actions():
        for i, _ in enumerate(hand):
            for j, (flag, layout_line) in enumerate(zip(flags, layout)):
                 if flag.owner is None and len(layout_line) < 3:
                    yield{'from':i, 'to':j}

    return tuple(get_legal_actions())[0]

if __name__ == "__main__":
    local = Environment()
    local.print()
    ai = action.AI()
    command = buttle_line_env.xxx()

    while True:
        input_play_first = input("ENTER COMMAND (f to play_first) >>> ")
        input_manual = input("ENTER COMMAND (m to manual) >>> ")
            
        turn = 0
        player = 0

        first_player = 0 if input_play_first == 'f' else 1
                
        while True:

            turn = turn+1
            print('ターン{}'.format(turn))
            print('プレイヤー{}'.format(player))
        
            if input_manual == 'm' and player == first_player:
                choice = int(input("choice >>> "))
                num = input("num >>> ")
                local.choice(player, turn, choice, int(num))
            else:               
                
                play_first = turn % 2 == 1

                if not (input_play_first == 'f') ^ play_first:               
                    #other_layout = copy.deepcopy(local.layout[(player+1)%2])
                    choice, num = command.operation_alpha(local.layout[player], local.layout[(player+1)%2], local.flags, local.hands[player], len(local.hands[(player+1)%2]), 46-turn, play_first)
                    #local.layout[(player+1)%2] = other_layout 
                else:
                    # カード選択（ランダム選択）
                    choice, num = ai.action(local.layout[player], local.layout[(player+1)%2], local.flags, local.hands[player], len(local.hands[(player+1)%2]), 46-turn, play_first)
                
                # 選択したカードを場に出す
                card = local.hands[player].pop(num)
                local.layout[player][choice].append(card)

            #action = get_action(local.layout[0], local.layout[1], local.flags, local.hands[0], len(local.hands[1]), 46-turn, True)
            print(json.dumps({'from':choice, 'to':num}))
            
            #ドロー
            if turn < 47:
                local.hands[player].append(local.stock[turn+13])
            
            # 勝敗判定
            local.checkFlag(choice, player, turn)
            
            # 盤面表示
            local.print()
            player = (player+1)%2

            s,e = local.winner()

            if s == 3 and e == 0:
                print('player0 win!!')
                break
            elif s == 0 and e == 3:
                print('player1 win!!')
                break
            elif  s >= 5 or e>= 5:
                if s > e:
                    print('player0 win!!')
                else:
                    print('player1 win!!')
                break

        break