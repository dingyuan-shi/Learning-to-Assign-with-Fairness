from collections import defaultdict
from pulp import LpProblem, LpVariable, LpMinimize, LpBinary, PULP_CBC_CMD
from recorder import Recorder
from typing import Dict, List, Set, Any
import math
import time


def dist(x1, y1, x2, y2):
    return math.sqrt((x2 - x1) * (x2 - x1) + (y2 - y1) * (y2 - y1)) * 110000


topK = 50000


class Agent(Recorder):
    """ Agent for dispatching and reposition """

    def __init__(self, **kwargs):
        super().__init__()
        """ Load your trained model and initialize the parameters """
        self.drivers_utility = defaultdict(float)       # type: Dict[int, float]
        self.driver_online_rounds = defaultdict(int)    # type: Dict[int, int]
        self.driver_max_last_round = 5000

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
        timestamp = dispatch_observ[0]['timestamp']
        cur_local = time.localtime(timestamp)
        od_decision = dict()                # type: Dict[str, Any]
        if_match_utility = dict()           # type: Dict[str, float]
        drivers_cur_round = set()           # type: Set[int]
        orders_cur_round = set()            # type: Set[int]
        order_driver_cand = defaultdict(int)
        dispatch_observ.sort(key=lambda x: x['order_driver_distance'])
        for od in dispatch_observ:
            driver_id = od['driver_id']     # type: int
            order_id = od['order_id']       # type: int
            orders_cur_round.add(order_id)
            if order_driver_cand[order_id] < topK:
                order_driver_cand[order_id] += 1
                drivers_cur_round.add(driver_id)
                driver_id_order_id = str(driver_id) + "_" + str(order_id)  # type: str
                od_decision[driver_id_order_id] = LpVariable(cat=LpBinary, name=driver_id_order_id)
                lng1, lat1 = od['order_start_location']
                lng2, lat2 = od['order_finish_location']
                if_match_utility[driver_id_order_id] = dist(lng1, lat1, lng2, lat2) - od['order_driver_distance']

        # Create a new model
        m = LpProblem(name="ILP_Model", sense=LpMinimize)
        # each driver should only have at most one order
        # each order should only have at most one driver
        driver_constrains = defaultdict(int)
        order_constrains = defaultdict(int)
        goal = 0
        for driver_id_order_id in od_decision:
            driver_id, order_id = driver_id_order_id.split('_')
            driver_constrains[driver_id] += od_decision[driver_id_order_id]
            order_constrains[order_id] += od_decision[driver_id_order_id]
            one_driver = self.driver_max_last_round - \
                         (self.drivers_utility[driver_id] / (self.driver_online_rounds[driver_id] + 1)
                          + od_decision[driver_id_order_id] * if_match_utility[driver_id_order_id] /
                          (self.driver_online_rounds[driver_id] + 1))
            one_driver_abs = LpVariable(name="abs_driver" + driver_id_order_id)
            m += (one_driver_abs >= one_driver)
            m += (one_driver_abs >= -one_driver)
            goal += one_driver_abs
        for driver_id in driver_constrains:
            m += (driver_constrains[driver_id] <= 1)
        for order_id in order_constrains:
            m += (order_constrains[order_id] <= 1)
        m += goal
        m.solve(PULP_CBC_CMD(msg=False))
        # update the online rounds
        if cur_local.tm_min == 0 and cur_local.tm_min == 0:
            for driver_id in drivers_cur_round:
                self.driver_online_rounds[driver_id] = 0
            for driver_id in self.drivers_utility:
                self.drivers_utility[driver_id] = 0
        for driver_id in drivers_cur_round:
            self.driver_online_rounds[driver_id] += 1
        dispatch_action = []    # type: List[Dict[str, int]]
        for v in m.variables():
            if v.varValue == 1:
                driver_id_str, order_id_str = v.name.split('_')
                # print("#####", driver_id, order_id)
                dispatch_action.append(dict(order_id=int(order_id_str), driver_id=int(driver_id_str)))
                # update the utility
                self.drivers_utility[int(driver_id_str)] += if_match_utility[driver_id_str + '_' + order_id_str]
                self.driver_max_last_round = max(self.driver_max_last_round,
                                                 self.drivers_utility[int(driver_id_str)] / self.driver_online_rounds[int(driver_id_str)])
        return dispatch_action

    def reposition(self, repo_observ):
        """ Compute the reposition action for the given drivers
        :param repo_observ: a dict, the key in the dict includes:
                timestamp: int
                driver_info: a list of dict, the key in the dict includes:
                        driver_id: driver_id of the idle driver in the treatment group, int
                        grid_id: id of the grid the driver is located at, str
                day_of_week: int
        :return: a list of dict, the key in the dict includes:
                driver_id: corresponding to the driver_id in the od_list
                destination: id of the grid the driver is repositioned to, str
        """
        return []
