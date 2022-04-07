# encoding=utf-8
import numpy as np
import random
import time

random.seed(0)

zero_threshold = 0.0000001
INF = 100000000
DEPTH = -1


class KMNode(object):
    def __init__(self, idx, exception=0, match=None, visit=False):
        self.id = idx
        self.exception = exception
        self.match = match
        self.visit = visit
        self.slack = INF

    def __repr__(self):
        return "idx:" + str(self.id) + " tag: " + str(self.exception) + " match: " + \
               str(self.match) + " vis: " + str(self.visit) + " slack: " + str(self.slack)


class KuhnMunkres(object):
    def __init__(self):
        self.matrix = None
        self.x_nodes = []
        self.y_nodes = []
        self.x_length = 0
        self.y_length = 0
        self.index_x = 0
        self.index_y = 1

    def set_matrix(self, x_y_values):
        xs = set()
        ys = set()
        for x, y, w in x_y_values:
            xs.add(x)
            ys.add(y)

        # 选取较小的作为x
        if len(xs) <= len(ys):
            self.index_x = 0
            self.index_y = 1
        else:
            self.index_x = 1
            self.index_y = 0
            xs, ys = ys, xs

        x_dic = {x: i for i, x in enumerate(xs)}
        y_dic = {y: j for j, y in enumerate(ys)}
        self.x_nodes = [KMNode(x) for x in xs]
        self.y_nodes = [KMNode(y) for y in ys]
        self.x_length = len(xs)
        self.y_length = len(ys)

        self.matrix = np.zeros((self.x_length, self.y_length))
        for row in x_y_values:
            x = row[self.index_x]
            y = row[self.index_y]
            w = row[2]
            x_index = x_dic[x]
            y_index = y_dic[y]
            self.matrix[x_index, y_index] = w

        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i, :])

    def km(self):
        for i in range(self.x_length):
            for node in self.y_nodes:
                node.slack = INF
            while True:
                for node in self.x_nodes:
                    node.visit = False
                for node in self.y_nodes:
                    node.visit = False
                if self.dfs(i, 0):
                    break
                d = INF
                for node in self.y_nodes:
                    if (not node.visit) and d > node.slack:
                        d = node.slack
                if d == INF or d < zero_threshold:
                    break
                for node in self.x_nodes:
                    if node.visit:
                        node.exception -= d
                for node in self.y_nodes:
                    if node.visit:
                        node.exception += d
                    else:
                        node.slack -= d

    def dfs(self, x, depth):
        global DEPTH
        if depth > DEPTH and depth % 20 == 0:
            print(depth)
        DEPTH = max(DEPTH, depth)
        if DEPTH > 500:
            return False
        x_node = self.x_nodes[x]
        x_node.visit = True
        for y in range(self.y_length):
            y_node = self.y_nodes[y]
            if y_node.visit:
                continue
            t = x_node.exception + y_node.exception - self.matrix[x][y]
            if abs(t) < zero_threshold and int("8888" + str(x_node.id)) != int(y_node.id) \
                    and int("8888" + str(y_node.id)) != int(x_node.id):
                y_node.visit = True
                if y_node.match is None or self.dfs(y_node.match, depth + 1):
                    y_node.match = x
                    x_node.match = y
                    return True
            elif y_node.slack > t:
                y_node.slack = t
        return False

    def set_match_list(self, match_list):
        for i, j in match_list:
            x_node = self.x_nodes[i]
            y_node = self.y_nodes[j]
            x_node.match = j
            y_node.match = i

    def get_connect_result(self):
        ret = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            if j is None:
                continue
            y_node = self.y_nodes[j]
            x_id = x_node.id
            y_id = y_node.id
            w = self.matrix[i][j]

            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            ret.append((x_id, y_id, w))
        return ret

    def get_max_value_result(self):
        ret = 0
        for i in range(self.x_length):
            j = self.x_nodes[i].match
            if j is None:
                continue
            ret += self.matrix[i][j]
        return ret


def find_max_match(x_y_values):
    solver = KuhnMunkres()
    solver.set_matrix(x_y_values)
    solver.km()
    return solver.get_max_value_result(), solver.get_connect_result()


if __name__ == '__main__':
    values = []
    random.seed(0)
    for i in range(50):
        for j in range(60):
            if i // 100 == j // 1000:
                value = random.random()
                values.append((i, j, value))
    print("begin")
    s_time = time.time()
    print(find_max_match(values))
    print("time usage: %s " % str(time.time() - s_time))
