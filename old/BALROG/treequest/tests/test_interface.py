from typing import Optional, Tuple

import treequest as tq

# (1) initial promptを返す関数

# (2) 評価結果を返す関数

# (3) refinement promptを返す関数


# class MyTask(abmcts.Task):

# 	def initial_prompt():
# 		…


# algo = abmcts.AB_MCTS_A_G(tau=0.5)
# abmcts.run(task=task, algorithm=run)


def test_generate():
    def generate_fn_0(parent: Optional[str], action_id: int = 0) -> Tuple[str, float]:
        # parentを受け取って、次のノードを作り（LLMを呼びパースし評価する）、返す
        # Noneのときは親ノードなしの最初のノード
        # floatが評価結果
        return "next", 0.0

    generate_fn = {"Action 0": generate_fn_0}

    algorithm = tq.StandardMCTS()

    state = algorithm.init_tree()

    for _ in range(10):
        state = algorithm.step(state, generate_fn)

    assert len(state.tree.get_nodes()) == 11

    # with open("hoge.pkl", 'wb') as f:
    #     pickle.save(state, wb)

    for _ in range(10):
        state = algorithm.step(state, generate_fn)

    assert len(state.tree.get_nodes()) == 21

    # with open("hoge.pkl", 'wb') as f:
    #     state = pickle.load(state, wb)
