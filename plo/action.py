import random
import copy
import itertools

class AI():

    def __init__(self):
        self.stock = []

        for i in range(10, 70, 1):
            self.stock.append({'color':i//10, 'number':i%10+1})

    def action(self, layout, other_layout, flags, hand, other_hand_length, stock_length, play_first):
  
        # ランダム手
        return self.random_choice(layout, other_layout, flags, hand, other_hand_length, stock_length, play_first)
     
    def random_choice(self, layout, other_layout, flags, hand, other_hand_length, stock_length, play_first):

        layout_num = [0,1,2,3,4,5,6,7,8]
        card_num = [0,1,2,3,4,5,6]
        check = False
        count = 0

        while not check and count < 100:
            select_layout = random.choice(layout_num)
            select_card = random.choice(card_num)

            if select_card < len(hand):
                check = self.check(layout, other_layout, flags, hand, select_layout, select_card)
            
            count += 1
        
        # 上限越えの無限ループ対策
        if count == 100:  
            for i in range(9):
                if len(layout[i]) < 3:
                    return i, 0

        return select_layout, select_card

    def check(self, layout, other_layout, flags, hand, layout_num, card_num):
        # 勝敗がついていない、かつ3枚おかれていない
        if not flags[layout_num]['owner'] and len(layout[layout_num]) < 3:

            return True
            # 負け確は置けない
            #if not self.winnerCheck(layout, other_layout, hand, layout_num, card_num):
            #    return False
            #else:
            #    # 置ける
            #    return True
        else:
            return False

    def winnerCheck(self, layout, other_layout, hand, layout_num, card_num):
        
        eScore = 0
        eSum = 0

        if len(layout[layout_num]) == 2 and len(other_layout[layout_num]) == 3:
            eScore, eSum = self.judge(layout, other_layout, hand, layout_num)
            cards = copy.deepcopy(layout[layout_num])
            cards.append(hand[card_num])

            # 出そうとしている役
            lScore, lSum = self.score(cards)

            if lScore > eScore:
                return True
            elif lScore == eScore and lSum > eSum:
                return True
            else:
                # 負け確
                return False
            
        return True

    def judge(self, layout, other_layout, hand, layout_num):
        
        if len(other_layout) == 3:
            score, sum = self.score(other_layout)
            return score, sum
        
        # 予測する場のカード
        cards = layout[layout_num]

        # 場と手札から相手の最大の役を予測
        lStock = copy.deepcopy(hand)
        eStock = copy.deepcopy(self.stock)

        for l in range(9):
            lStock.extend(copy.deepcopy(layout[l]))
            lStock.extend(copy.deepcopy(other_layout[l]))
        
        for ls in lStock:
            eStock = list(itertools.filterfalse(lambda x: x['color'] == ls['color'] and x['number'] == ls['number'], eStock))

        # 場と手札から相手の最大の役を予測
        eScore = 0
        eSum = 0

        for x in range(len(eStock)-len(cards)):
            # 場の状態
            eLayout = copy.deepcopy(cards)
            eLayout.append(eStock[x])
            
            for i in range(len(eLayout), 3, 1):
                eLayout.append(eStock[x+i])
            
            score, sum = self.score(eLayout)

            if score > eScore:
                eScore = score
                eSum = sum
        
        return eScore, eSum

    def isThree(self, cards):
        return cards[0]['number'] == cards[1]['number'] and cards[1]['number'] == cards[2]['number']

    def isFlush(self, cards):
        return cards[0]['color'] == cards[1]['color'] and cards[1]['color'] == cards[2]['color']

    def isStraight(self, numList):
        return numList[2] - numList[1] == 1 and numList[1] - numList[0] == 1

    def score(self, cards):
        # 数字の配列を生成(昇順)
        numList = [i['number'] for i in cards]
        numList.sort()
        sum = numList[0] + numList[1] + numList[2]

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
        return score, sum