from collections import defaultdict
import random
from recorder import Recorder, acc_dist
from typing import Dict, List, Any
import time


class Agent(Recorder):
    """ Agent for dispatching and reposition """

    def __init__(self, **kwargs):
        super().__init__()
        """ Load your trained model and initialize the parameters """
        # this utility is different from the one of Recorder, this one is simply a distance substraction
        self.driver_to_utility = defaultdict(float)

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
        # record orders of each driver, add driver into utility recorder
        if len(dispatch_observ) == 0:
            return []
        timestamp = dispatch_observ[0]['timestamp']
        cur_local = time.localtime(timestamp)
        driver_to_orders = defaultdict(list)
        for od in dispatch_observ:
            lng1, lat1 = od['order_start_location']
            lng2, lat2 = od['order_finish_location']
            pref = acc_dist(lng1, lat1, lng2, lat2) - od['order_driver_distance']
            driver_to_orders[od['driver_id']].append((pref, od['order_id']))

        # make the right order of drivers based on utility
        # caution: only consider drivers in this round
        utility_driver_worst_first = [(self.drivers_income_per_hour[driver_id][cur_local.tm_hour], driver_id)
                                      for driver_id in driver_to_orders]
        utility_driver_worst_first.sort()
        # same utility need shuffling
        i = 0
        while i < len(utility_driver_worst_first) - 1:
            j = i + 1
            while j < len(utility_driver_worst_first) and \
                    abs(utility_driver_worst_first[j][0] - utility_driver_worst_first[i][0]) < 0.000005:
                j += 1
            if j - i > 1:
                copy = utility_driver_worst_first[i:j]
                random.shuffle(copy)
                utility_driver_worst_first[i:j] = copy
            i = j

        # worst first matching
        assigned_orders = set()
        dispatch_action = []
        for utility, driver_id in utility_driver_worst_first:
            # sort based on pref from high to low
            driver_to_orders[driver_id].sort(reverse=True)
            for pref, order_id in driver_to_orders[driver_id]:
                if order_id in assigned_orders:
                    continue
                assigned_orders.add(order_id)
                dispatch_action.append(dict(order_id=order_id, driver_id=driver_id))
                self.driver_to_utility[driver_id] += pref
                break
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
