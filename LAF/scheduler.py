import math
from typing import Dict, List, Any, Tuple
from grid import Grid
from matcher import Matcher
from global_var import SPEED, REPO_NAIVE
import time
import sys
if sys.platform == 'darwin':
    from model.utils import cal_online_seconds
else:
    from utils import cal_online_seconds


class Scheduler:
    def __init__(self, gamma: float):
        self.gamma = gamma

    def reposition(self, matcher: Matcher, repo_observ: Dict[str, Any],
                   drivers_total_income, drivers_log_on_off, drivers_income_per_hour) -> List[Dict[str, str]]:
        if len(repo_observ['driver_info']) == 0:
            return []
        cur_hour, cur_time, day_of_week, drivers = Scheduler.parse_repo(repo_observ, drivers_total_income,
                                                                         drivers_log_on_off, drivers_income_per_hour)
        grid_ids = Grid.get_grid_ids()
        drivers.sort(key=lambda x: x[2])  # sort from small to large based on ratio
        median = drivers[int(len(drivers) * 0.50)][2]
        reposition = []  # type: List[Dict[str, str]]
        for driver_id, current_grid_id, ratio in drivers:
            if 6 <= ratio <= 12:
                reposition.append(dict(driver_id=driver_id, destination=current_grid_id))
            else:
                best_grid_id, best_value = current_grid_id, -100
                current_value = matcher.get_grid_value(current_grid_id)
                for grid_id in grid_ids:
                    duration = Grid.mahattan_distance(current_grid_id, grid_id) / SPEED
                    if ratio < 6:
                        discount = math.pow(0.999, duration)
                        proposed_value = matcher.get_grid_value(grid_id)
                        incremental_value = discount * proposed_value - current_value
                    else:
                        incremental_value = -abs(median -
                                                 drivers_income_per_hour[driver_id][cur_hour] /
                                                 ((cal_online_seconds(
                                                     drivers_log_on_off[driver_id][0], cur_time, duration,
                                                     per_hour=False) + 0.1)
                                                  / 3600))
                    if incremental_value > best_value:
                        best_grid_id, best_value = grid_id, incremental_value
                reposition.append(dict(driver_id=driver_id, destination=best_grid_id))
        return reposition

    @staticmethod
    def parse_repo(repo_observ, drivers_total_income, drivers_log_on_off, drivers_income_per_hour):
        timestamp = repo_observ['timestamp']  # type: int
        cur_local = time.localtime(timestamp)
        cur_time = cur_local.tm_hour * 3600 + cur_local.tm_min * 60 + cur_local.tm_sec - 4 * 3600
        day_of_week = repo_observ['day_of_week']  # type: int
        drivers = [(driver['driver_id'], driver['grid_id'],
                    drivers_income_per_hour[driver['driver_id']][cur_local.tm_hour] /
                    ((cal_online_seconds(drivers_log_on_off[driver['driver_id']][0], cur_time, per_hour=False) + 0.1) / 3600))
                   for driver in repo_observ['driver_info']]  # type: List[Tuple[str, str, float]]
        return cur_local.tm_hour, cur_time, day_of_week, drivers
