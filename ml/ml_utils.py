import time


def current_time_in_millis():
    return int(round(time.time() * 1000))

print(current_time_in_millis())
