from gymnasium.envs.toy_text.frozen_lake import FrozenLakeEnv
class LessSlipperyFrozenLakeEnv(FrozenLakeEnv):
    """
    修改 FrozenLake 轉移機率的自定義環境。
    將預期動作的成功機率從 1/3 提高到 0.5。
    """
    def __init__(self, **kwargs):
        # 初始化父類 (FrozenLakeEnv)
        super().__init__(**kwargs)
        # print(self.P)
        # 覆寫轉移機率 P
        self.P = self.build_less_slippery_P()
        # print(self.P)

    def build_less_slippery_P(self, intended_prob=0.5):
        """
        建立新的轉移機率矩陣。
        - intended_prob: 預期動作成功的機率 (例如 0.5)
        - side_prob: 轉向兩側的機率 ( (1 - intended_prob) / 2 )
        """
        # P 是一個字典，鍵是狀態，值是動作的轉移列表
        new_P = {}
        
        # side_prob 為 (1-0.5)/2 = 0.25
        side_prob = (1.0 - intended_prob) / 2.0
        
        # 迭代所有狀態 (s) 和所有動作 (a)
        for s in self.P.keys():
            new_P[s] = {}
            for a in self.P[s].keys():
                
                # 原始轉移機率列表 (包含三個可能的結果)
                original_transitions = self.P[s][a]
                new_transitions = []
                
                # 重新分配機率
                for i, (prob, next_state, reward, terminated) in enumerate(original_transitions):
                    
                    if i == 1: # 這是預期的動作方向
                        new_prob = intended_prob
                    else: # 這是兩側的滑行方向
                        new_prob = side_prob
                        
                    new_transitions.append((new_prob, next_state, reward, terminated))
                    
                new_P[s][a] = new_transitions
                
        return new_P