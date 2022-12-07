import sys
import numpy as np
import datetime as dt
import os
import glob
import time
from spacekit.logger.log import Logger


class Stopwatch:
    def __init__(self, func, log=None, out="."):
        self.func = func
        self.log = log
        self.out = out
        self.lap = 0
        self.laps = {}
        self.ps = self.func.__name__
        self.t0 = None
        self.p0 = None
        self.t1 = None
        self.p1 = None
        self.walltime = None
        self.clocktime = None
        self.delta = None
        if self.log is None:
            self.log = Logger(self.ps, console_log_level="info").setup_logger()

    def clockit(self):
        def wrap(*args, **kwargs):
            self.start()
            result = self.func(*args, **kwargs)
            self.stop()
            return result

        return wrap

    def start(self):
        self.t0, self.p0 = time.time(), time.process_time()
        self.record(self.t0, "STARTED")

    def stop(self):
        self.t1, self.p1 = time.time(), time.process_time()
        self.record(self.t1, "COMPLETED")
        self.duration()
        self.log.info(f"\nWALL [{self.ps}] : {self.walltime}")
        self.log.info(f"\nCLOCK [{self.ps}] : {self.clocktime}")

    def record(self, t, info):
        timestring = dt.datetime.fromtimestamp(t).strftime("%m/%d/%Y - %H:%M:%S")
        self.log(f"{timestring} [i] {info} [{self.ps}]")
        if self.delta:
            self.log.info(f"\nDuration [{self.ps}] : {self.delta[0]} {self.delta[1]}\n")
            self.lap += 1
            self.laps[self.lap] = self.delta
            self.reset()

    def reset(self):
        self.t0 = None
        self.t1 = None
        self.p0 = None
        self.p1 = None
        self.delta = None
        self.walltime = None
        self.clocktime = None

    def duration(self, wall=True):
        """calculates total duration from start to end of a process that finished running.

        Parameters
        ----------
        start : int
            starting interval clocktime
        end : int
            end interval clocktime
        prcname : str, optional
            name of running process, by default ""
        """
        self.walltime = np.round((self.t1 - self.t0), 2)
        self.clocktime = np.round((self.p1 - self.p0), 2)

        s = self.walltime if wall is True else self.clocktime
        m = np.round((s / 60), 2)
        h = np.round((m / 60), 2)
        if s > 3600:
            self.delta = (h, "HOURS")
        elif s > 60:
            self.delta = (m, "MINUTES")
        else:
            self.delta = (s, "SECONDS")


def proc_time(start, end, prcname=""):
    """calculates total duration from start to end of a process that finished running.

    Parameters
    ----------
    start : int
        starting interval clocktime
    end : int
        end interval clocktime
    prcname : str, optional
        name of running process, by default ""
    """
    duration = np.round((end - start), 2)
    proc_time = np.round((duration / 60), 2)
    if duration > 3600:
        t = f"{np.round((proc_time / 60), 2)} hours."
    elif duration > 60:
        t = f"{proc_time} minutes."
    else:
        t = f"{duration} seconds."
    print(f"\nProcess [{prcname}] : {t}\n")
    return t


# TODO: record laps (eg. for loading train test val image sets)
def stopwatch(prcname, t0=None, t1=None, out=".", log=True, subset_name=None):
    """Times a process from start to finish and (optionally) records the intervals and total duration in a text file on disk.

    Parameters
    ----------
    prcname : str
        name of running process
    t0 : int, optional
        time.time timestamp start interval, by default None
    t1 : int, optional
        time.time timestamp end interval], by default None
    out : str, optional
        location to save recorded clocktimes, by default "."
    log : bool, optional
        record process clocktimes in a text file on disk, by default True
    """
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

    fname = "clocktime.txt" if subset_name is None else f"clocktime{subset_name}.txt"

    if log is True:
        sysout = sys.stdout
        with open(f"{out}/{fname}", "a") as timelog:
            sys.stdout = timelog
            print(message)
            if lap:
                proc_time(t0, t1, prcname=prcname)
            sys.stdout = sysout


def get_file_metrics(visit_path):
    fpaths = glob.glob(f"{visit_path}/*.fits")
    n_files = len(fpaths)
    if n_files > 0:
        total_size = sum([os.stat(fname).st_size for fname in fpaths])
        return n_files, total_size
    else:
        return 0, 0


def timer(t0=None, clock=None):
    if t0 is None:
        return time.time(), time.process_time()
    else:
        walltime = time.time() - t0
        clocktime = time.process_time() - clock
        return walltime, clocktime


def clockit(func):
    ps = func.__name__

    def wrap(*args, **kwargs):
        start = time.time()
        start(ps, t0=start)
        result = func(*args, **kwargs)
        end = time.time()
        stopwatch(ps, t0=start, t1=end)
        print(end - start)
        return result

    return wrap


def record_metrics(
    log_dir, visit, wall, clock, ps="svm", n_files=None, total_size=None
):
    log_file = os.path.join(log_dir, f"{ps}-stats.txt")
    metrics = {"visit": visit, "walltime": wall, "clocktime": clock}
    if n_files:
        metrics.update({"n_files": n_files, "total_size": total_size})

    with open(log_file, "w") as lf:
        for k, v in metrics.items():
            lf.write(f"{k}: {v}\n")
