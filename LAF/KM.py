# encoding=utf-8

import numpy as np
import random
import time
from collections import deque, defaultdict
from multiprocessing import Process, Queue, cpu_count

random.seed(0)

zero_threshold = 0.0000001
INF = 100000000
CPU_CNT = cpu_count()


def bfs_split(values):
    left_name_idx = dict()
    left_idx_name = []
    right_name_idx = dict()
    right_idx_name = []
    left_cnt = 0
    right_cnt = 0
    left_right = defaultdict(list)
    right_left = defaultdict(list)
    for x, y, w in values:
        if x not in left_name_idx:
            left_name_idx[x] = left_cnt
            left_cnt += 1
            left_idx_name.append(x)
        if y not in right_name_idx:
            right_name_idx[y] = right_cnt
            right_cnt += 1
            right_idx_name.append(y)
        left_right[left_name_idx[x]].append((right_name_idx[y], w))
        right_left[right_name_idx[y]].append((left_name_idx[x], w))
    left_visit = [False] * left_cnt
    right_visit = [False] * right_cnt

    blocks = []
    for x in left_right:
        if left_visit[x]:
            continue
        block_x = [x]
        left_visit[x] = True
        q = deque([(x, 'l')])
        while q:
            src, side = q.popleft()
            if side == 'l':
                for dst, w in left_right[src]:
                    if right_visit[dst]:
                        continue
                    right_visit[dst] = True
                    q.append((dst, 'r'))
            else:
                for dst, w in right_left[src]:
                    if left_visit[dst]:
                        continue
                    block_x.append(dst)
                    left_visit[dst] = True
                    q.append((dst, 'l'))
        # convert blocks back to values
        values_block = []
        for x_1 in block_x:
            for y, w in left_right[x_1]:
                values_block.append((left_idx_name[x_1], right_idx_name[y], w))
        blocks.append(values_block)
    return blocks


class KMNode(object):
    def __init__(self, idx, no, exception=0, match=None, visit=False):
        self.id = idx
        self.no = no
        self.exception = exception
        self.match = match
        self.visit = visit
        self.slack = INF

    def __repr__(self):
        return "idx:" + str(self.id) + " tag: " + str(self.exception) + " match: " + \
               str(self.match) + " vis: " + str(self.visit) + " slack: " + str(self.slack)


class KuhnMunkres(object):

    def __init__(self, interval):
        self.matrix = None
        self.x_nodes = []
        self.y_nodes = []
        self.x_length = 0
        self.y_length = 0
        self.index_x = 0
        self.index_y = 1
        self.online_time = None
        self.income = None
        self.order_price_dur = None
        self.interval = interval
        self.quick_sol = None

    def set_matrix(self, x_y_values, online_time, income, order_price_dur):
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
        self.x_nodes = [KMNode(x, x_dic[x]) for x in xs]
        self.y_nodes = [KMNode(y, y_dic[y]) for y in ys]
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
        if self.x_length == 1:
            best_choice = int(np.argmax(self.matrix[0]))
            max_val = self.matrix[0][best_choice]
            left_id = self.x_nodes[0].id
            right_id = self.y_nodes[best_choice].id
            if self.index_x == 1:
                left_id, right_id = right_id, left_id
            match = [(left_id, right_id, max_val)]
            self.quick_sol = (max_val, match)
            return
        # left is worker
        if self.index_x == 0:
            self.online_time = [online_time[node.id] for node in self.x_nodes]
            self.income = [income[node.id] for node in self.x_nodes]
            self.order_price_dur = [order_price_dur[node.id] for node in self.y_nodes]
        # right is worker
        else:
            self.online_time = [online_time[node.id] for node in self.y_nodes]
            self.income = [income[node.id] for node in self.y_nodes]
            self.order_price_dur = [order_price_dur[node.id] for node in self.x_nodes]

        for i in range(self.x_length):
            self.x_nodes[i].exception = max(self.matrix[i, :])

    def km(self):
        if self.quick_sol is not None:
            return
        for i in range(self.x_length):
            for node in self.y_nodes:
                node.slack = INF
            while True:
                for node in self.x_nodes:
                    node.visit = False
                for node in self.y_nodes:
                    node.visit = False
                if self.dfs(i):
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
        # remain order is not matching
        if self.index_x == 1:
            remain_orders = [(self.order_price_dur[x.no][0] / self.order_price_dur[x.no][1], x.no) for x in self.x_nodes if x.match is None]
            if len(remain_orders) == 0:
                return
            remain_drivers = [(self.income[y.no] / (self.online_time[y.no] + 0.1), y.no) for y in self.y_nodes if y.match is None]
            remain_drivers.sort()
            remain_orders.sort(reverse=True)
            idx = 0
            for _, order_no in remain_orders:
                driver_ratio, driver_no = remain_drivers[idx]
                idx += 1
                self.x_nodes[order_no].match = driver_no
                self.y_nodes[driver_no].match = order_no
        return

    def check_fairness(self, x_node, y_node):
        driver = x_node
        order = y_node
        impact_driver = None if order.match is None else self.x_nodes[order.match]
        impact_order = None
        if impact_driver is not None:
            for y in range(self.y_length):
                y_order = self.y_nodes[y]
                if y_order.visit:
                    continue
                t = impact_driver.exception + y_order.exception - self.matrix[impact_driver.no][y]
                if abs(t) < zero_threshold:
                    impact_order = y_order
                    break
        if self.index_x == 1:
            driver, order = order, driver
            impact_driver, impact_order = impact_order, impact_driver
        # judge two drivers gap
        if impact_driver is None:
            return True
        if impact_order is None:
            price, duration = self.order_price_dur[order.no]
            order_ratio = price / duration
            driver_ratio = self.income[driver.no] / (self.online_time[driver.no] + 0.1)
            impact_driver_ratio = self.income[impact_driver.no] / (self.online_time[impact_driver.no] + 0.1)
            if order_ratio < min(driver_ratio, impact_driver_ratio):
                return driver_ratio > impact_driver_ratio
            else:
                return driver_ratio < impact_driver_ratio
        # both impact order and impact driver is not None
        ratio1 = (self.income[driver.no] + self.order_price_dur[order.no][0]) / \
                 ((0.1 + self.online_time[driver.no] + self.order_price_dur[order.no][1]) / 3600)
        ratio2 = (self.income[impact_driver.no] + self.order_price_dur[impact_order.no][0]) / \
                 ((0.1 + self.online_time[impact_driver.no] + self.order_price_dur[impact_order.no][1]) / 3600)
        # return self.interval[1] < ratio1 < self.interval[3] and self.interval[1] < ratio2 < self.interval[3]
        return abs(ratio1 - ratio2) < self.interval[3] - self.interval[2] and max(ratio1, ratio2) < self.interval[4]

    def dfs(self, x):
        x_node = self.x_nodes[x]
        x_node.visit = True
        for y in range(self.y_length):
            y_node = self.y_nodes[y]
            if y_node.visit:
                continue
            t = x_node.exception + y_node.exception - self.matrix[x][y]
            if abs(t) < zero_threshold:
                y_node.visit = True
                if y_node.match is None or (self.check_fairness(x_node, y_node) and self.dfs(y_node.match)):
                    y_node.match = x
                    x_node.match = y
                    return True
            elif y_node.slack > t:
                y_node.slack = t
        return False

    def get_connect_result(self):
        if self.quick_sol is not None:
            return self.quick_sol[1]
        ret = []
        for i in range(self.x_length):
            x_node = self.x_nodes[i]
            j = x_node.match
            if j is None:
                continue
            # TODO: handle those unmatched orders
            y_node = self.y_nodes[j]
            x_id = x_node.id
            y_id = y_node.id
            w = self.matrix[i][j]
            if self.index_x == 1 and self.index_y == 0:
                x_id, y_id = y_id, x_id
            ret.append((x_id, y_id, w))
        return ret

    def get_max_value_result(self):
        if self.quick_sol is not None:
            return self.quick_sol[0]
        ret = 0
        for i in range(self.x_length):
            j = self.x_nodes[i].match
            if j is None:
                continue
            ret += self.matrix[i][j]
        return ret


def find_part_block(part_block_value, online_time, income, order_price_dur, interval, q=None):
    solvers = [KuhnMunkres(interval) for _ in range(len(part_block_value))]
    for i, solver in enumerate(solvers):
        solver.set_matrix(part_block_value[i], online_time, income, order_price_dur)
        solver.km()
    val = 0
    for solver in solvers:
        val += solver.get_max_value_result()
    matches = [solver.get_connect_result() for solver in solvers]
    match_all = [match[i] for match in matches for i in range(len(match))]
    if q:
        q.put((val, match_all))
    return val, match_all


def find_max_match(x_y_values, online_time, income, order_price_dur, interval, split=True, mult_process=False):
    if (not split) or (len(x_y_values) < 1):
       return find_part_block([x_y_values], online_time, income, order_price_dur, interval)
    block_values = bfs_split(x_y_values)
    # print("split into %d blocks" % len(block_values))
    if mult_process:
        random.shuffle(block_values)
        STEP = len(block_values) // CPU_CNT
        max_val = 0
        matches = []
        # print("multi process")
        q = Queue()
        processes = [Process(target=find_part_block, args=(block_values[i * STEP:
                                                                        len(block_values)
                                                                        if i == CPU_CNT - 1 else (i + 1) * STEP],
                                                           online_time, income, order_price_dur, interval,
                                                           q)) for i in range(CPU_CNT)]
        for p in processes:
            p.start()
            p.join()
        for _ in range(len(processes)):
            val, match = q.get()
            max_val += val
            matches.extend(match)
    else:
        max_val, matches = find_part_block(block_values, online_time, income, order_price_dur, interval)
    return max_val, matches


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
    print(find_max_match(values, split=False, mult_process=False))
    print("time usage: %s " % str(time.time() - s_time))
