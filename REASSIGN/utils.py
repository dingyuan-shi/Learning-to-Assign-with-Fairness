

def cal_online_seconds(log_on, cur, dur=0, per_hour=True):
    return (cur - log_on + dur) if not per_hour or cur < (log_on + 3599) // 3600 * 3600 else cur % 3600 + dur


def cal_income(driver_id, cur_hour, drivers_total_income, drivers_income_per_hour, per_hour=True):
    return drivers_income_per_hour[driver_id][cur_hour] if per_hour else drivers_total_income[driver_id]

