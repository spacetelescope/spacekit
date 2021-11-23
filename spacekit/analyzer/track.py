import sys
import numpy as np
import datetime as dt


def proc_time(start, end, prcname=""):
    duration = np.round((end - start), 2)
    proc_time = np.round((duration / 60), 2)
    if duration > 3600:
        t = f"{np.round((proc_time / 60), 2)} hours."
    elif duration > 60:
        t = f"{proc_time} minutes."
    else:
        t = f"{duration} seconds."
    print(f"\nProcess [{prcname}] : {t}\n")


# TODO: record laps (eg. for loading train test val image sets)
def stopwatch(prcname, t0=None, t1=None, out=".", log=True):
    lap = 0
    if t1 is not None:
        info = "COMPLETED"
        t = t1
        if t0 is not None:
            lap = 1
    else:
        info = "STARTED"
        t = t0
    timestring = dt.datetime.fromtimestamp(t).strftime("%m/%d/%Y - %H:%M:%S")
    message = f"{timestring} [i] {info} [{prcname}]"
    print(message)

    if log is True:
        sysout = sys.stdout
        with open(f"{out}/clocktime.txt", "a") as timelog:
            sys.stdout = timelog
            print(message)
            if lap:
                proc_time(t0, t1)
            sys.stdout = sysout
