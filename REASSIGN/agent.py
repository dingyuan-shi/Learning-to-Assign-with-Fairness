from recorder import Recorder
from typing import List, Dict, Any
import time
import sys
from collections import defaultdict
if sys.platform == 'darwin':
    from model.KM import find_max_match
    from model.utils import cal_income, cal_online_seconds
else:
    from KM import find_max_match
    from utils import cal_income, cal_online_seconds

IS_PER_HOUR = True


class Agent(Recorder):
    def __init__(self, **kwargs):
        super().__init__()

    def dispatch(self, dispatch_observ: List[Dict[str, Any]], index2hash=None) -> List[Dict[str, int]]:
        """ Compute the assignment between drivers and passengers at each time step
        :param dispatch_observ: a list of dict, the key in the dict includes:
                order_id, int
                driver_id, int
                order_driver_distance, float
                order_start_location, a list as [lng, lat], float
                order_finish_location, a list as [lng, lat], float
                driver_location, a list as [lng, lat], float
                timestamp, int
                order_finish_timestamp, int
                day_of_week, int
                reward_units, float
                pick_up_eta, float
        :param index2hash: driver_id to driver_hash
        :return: a list of dict, the key in the dict includes:
                order_id and driver_id, the pair indicating the assignment
        """
        if len(dispatch_observ) == 0:
            return []
        cur_time = time.localtime(dispatch_observ[0]['timestamp'])
        cur_sec = (cur_time.tm_hour - 4) * 3600 + cur_time.tm_min * 60 + cur_time.tm_sec
        drivers = set(each['driver_id'] for each in dispatch_observ)
        orders = set(each['order_id'] for each in dispatch_observ)
        order_to_dur = defaultdict(int)
        for each in dispatch_observ:
            order_to_dur[each['order_id']] = each['order_finish_timestamp'] - each['timestamp']
        order_to_pri = {each['order_id']: each['reward_units'] for each in dispatch_observ}
        fake_edges = [(driver, int("8888" + str(driver)), 0) for driver in drivers]
        edges = [(each['driver_id'], each['order_id'], each['reward_units']) for each in dispatch_observ]
        edge_plus = edges + fake_edges
        # get M_old
        v, match_old = find_max_match(edges)
        match_old_dic = {each[0]: each[1] for each in match_old}
        # get M_fair  bi search for edge weights
        match_fair = match_old
        lo, hi = 0, 50
        while abs(lo - hi) > 0.001:
            f = (lo + hi) / 2
            edge_f = [edge for edge in edge_plus if (cal_income(index2hash[edge[0]], cur_time.tm_hour,
                                                                self.drivers_total_income,
                                                                self.drivers_income_per_hour, IS_PER_HOUR)
                                                     + edge[2]) /
                      (cal_online_seconds(self.drivers_log_on_off[index2hash[edge[0]]][0],
                                         cur_sec, order_to_dur[edge[1]], IS_PER_HOUR) + 0.1) > f]
            v_f, match_fair = find_max_match(edge_f)
            perfect_match = True
            if len(match_fair) < min(len(order_to_dur), len(drivers)):
                perfect_match = False
            if perfect_match:
                lo = f
            else:
                hi = f
        match_fair_dic = {each[0]: each[1] for each in match_fair if each[2] > 0.000001}
        f_opt = lo
        # get f_threshold
        driver_incomes = [cal_income(index2hash[driver], cur_time.tm_hour,
                                     self.drivers_total_income, self.drivers_income_per_hour, IS_PER_HOUR)
                          for driver in drivers]
        driver_incomes.sort()
        f_thresh = driver_incomes[int(len(driver_incomes) * 0.1)]
        if f_thresh < 0.00001:
            match_new_dic = match_old_dic
        elif f_thresh > f_opt:
            match_new_dic = match_fair_dic
        else:
            # reassign
            match_new_dic = match_old_dic
            break_loop = 0
            while True:
                break_loop += 1
                for driver in match_new_dic:
                    order = match_new_dic[driver]
                    price = order_to_pri[order]
                    if (cal_income(index2hash[driver], cur_time.tm_hour,
                                self.drivers_total_income, self.drivers_income_per_hour, IS_PER_HOUR) + price) / \
                            (cal_online_seconds(self.drivers_log_on_off[index2hash[driver]][0],
                                                cur_sec, order_to_dur[order], IS_PER_HOUR) + 0.1) < f_thresh:
                        v = driver
                        break
                else:
                    break
                match_new_dic.pop(v)
                if v not in match_fair_dic:
                    continue
                while True:
                    break_loop += 1
                    for driver in match_new_dic:
                        if match_new_dic[driver] == match_fair_dic[v]:
                            vp = driver
                            break
                    else:
                        break
                    match_new_dic.pop(vp)
                    match_new_dic[v] = match_fair_dic[v]
                    v = vp
                    if break_loop > 1000000:
                        print("may cause dead loop")
                        break
                match_new_dic[v] = match_fair_dic[v]
                if break_loop > 1000000:
                    print("may cause dead loop")
                    break
        res = []
        assigned_orders = set()
        assigned_drivers = set()
        for driver in drivers:
            if driver in match_new_dic and (len(str(match_new_dic[driver])) < 4 or
                                            str(match_new_dic[driver])[0:4] != "8888"):
                res.append(dict(driver_id=driver, order_id=match_new_dic[driver]))
                assigned_drivers.add(driver)
                assigned_orders.add(match_new_dic[driver])
        dispatch_observ.sort(key=lambda x: x['order_driver_distance'])
        for each in dispatch_observ:
            if each['driver_id'] not in assigned_drivers and each['order_id'] not in assigned_orders:
                res.append(dict(driver_id=each['driver_id'], order_id=each['order_id']))
                assigned_orders.add(each['order_id'])
                assigned_drivers.add(each['driver_id'])
        return res

    def reposition(self, repo_observ):
        return []
