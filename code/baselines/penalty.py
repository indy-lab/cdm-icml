import numpy as np

def penalty(params, data):
    pen = 0
    for winner, losers in data:
        choice_set = [winner] + losers
        choice_set.sort(key=lambda x: params[x], reverse=True)
        pen += choice_set.index(winner)
    return pen / len(data)

def test_penalty():
    data = [[0, [1, 2, 3]], [1, [0, 2, 3]], [2, [0, 1, 3]], [3, [0, 1, 2]]]
    params = [0.1, 0.2, 0.3, 0.4]
    print(penalty(params, data))

def test_penalty(n=10, m=1000, val = None):
    pen = []
    for _ in range(m):
        data = [[0, list(np.arange(1, n))]]
        params = np.random.rand(n)
        if val is not None:
            params[0] = val
        pen.append(penalty(params, data))
    print(np.mean(pen))

if __name__ == "__main__":
    test_penalty(val = 0.5)