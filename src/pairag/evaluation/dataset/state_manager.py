import json
import os
from enum import Enum


class DatasetState(Enum):
    QCA = "qca"
    QCAP = "qcap"
    RETRIEVAL = "retrieval_evaluation"
    RESPONSE = "response_evaluation"
    E2E = "end2end_evaluation"


class StateManager:
    def __init__(self, state_file="state.json"):
        self.state_file = state_file
        # 使用字典来保存每个状态的完成情况
        self.states = {state: False for state in DatasetState}
        self.load_state()

    def load_state(self):
        """从JSON文件加载所有状态的完成情况"""
        if os.path.exists(self.state_file):
            try:
                with open(self.state_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    for state in DatasetState:
                        # 更新状态的完成情况，如果JSON中有对应的键
                        if state.value in data:
                            self.states[state] = data[state.value]
                    print(f"加载的状态: {self.states}")
            except Exception as e:
                print(f"加载状态时出错: {e}")
                # 如果出错，保持所有状态为未完成
        else:
            print("状态文件不存在，初始化所有状态为未完成")

    def save_state(self):
        """将所有状态的完成情况保存到JSON文件"""
        try:
            with open(self.state_file, "w", encoding="utf-8") as f:
                # 将状态字典转换为以状态值为键的字典
                data = {
                    state.value: completed for state, completed in self.states.items()
                }
                json.dump(data, f, ensure_ascii=False, indent=4)
            print(f"状态已保存为: {data}")
        except Exception as e:
            print(f"保存状态时出错: {e}")

    def mark_completed(self, new_state):
        """标记某个状态为完成并保存"""
        if isinstance(new_state, DatasetState):
            self.states[new_state] = True
            self.save_state()
            print(f"状态已标记为完成: {new_state}")
        else:
            raise ValueError("new_state 必须是 DatasetState 的实例")

    def is_completed(self, state):
        """判断某个状态是否已经完成"""
        if isinstance(state, DatasetState):
            return self.states.get(state, False)
        else:
            raise ValueError("state 必须是 DatasetState 的实例")

    def get_completed_states(self):
        """获取所有已完成的状态"""
        return [state for state, completed in self.states.items() if completed]

    def reset_state(self):
        """重置所有状态为未完成"""
        for state in self.states:
            self.states[state] = False
        self.save_state()
        print("所有状态已重置为未完成")
