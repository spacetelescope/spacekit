#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# -------------------------------------------------------------------------------
#                                                                               -
#  Python dual-logging setup (console and log file),                            -
#  supporting different log levels and colorized output                         -
#                                                                               -
#  LogFormatter credit: on Fonic <https://github.com/fonic>                     -
#  https://stackoverflow.com/a/13733863/1976617                                 -
#  https://uran198.github.io/en/python/2016/07/12/colorful-python-logging.html  -
#  https://en.wikipedia.org/wiki/ANSI_escape_code#Colors                        -
#                                                                               -
# -------------------------------------------------------------------------------

# Imports
import os
import sys
import logging
from stsci.tools import logutil


# Logging formatter supporting colorized output
class LogFormatter(logging.Formatter):

    COLOR_CODES = {
        logging.CRITICAL: "\033[1;35m",  # bright/bold magenta
        logging.ERROR: "\033[1;31m",  # bright/bold red
        logging.WARNING: "\033[1;33m",  # bright/bold yellow
        logging.INFO: "\033[0;37m",  # white / light gray
        logging.DEBUG: "\033[1;30m",  # bright/bold black / dark gray
    }

    RESET_CODE = "\033[0m"

    def __init__(self, color, *args, **kwargs):
        super(LogFormatter, self).__init__(*args, **kwargs)
        self.color = color

    def format(self, record, *args, **kwargs):
        if self.color is True and record.levelno in self.COLOR_CODES:
            record.color_on = self.COLOR_CODES[record.levelno]
            record.color_off = self.RESET_CODE
        else:
            record.color_on = ""
            record.color_off = ""
        return super(LogFormatter, self).format(record, *args, **kwargs)


class Logger:
    def __init__(
        self,
        script_name,
        console_log_output="stdout",
        console_log_level="warning",
        console_log_color=True,
        logfile_log_level="debug",
        logfile_log_color=False,
        asctime=True,
        threadname=False,
        splunk=False,
    ):
        self.__name__ = script_name  # "diagnostic_json_harvester"
        self.log_level = logging.DEBUG
        self.console_log_output = console_log_output.lower()
        self.console_log_level = console_log_level.upper()
        self.console_log_color = console_log_color
        self.console_formatter = None
        self.logfile_file = self.__name__ + ".log"
        self.logfile_log_level = logfile_log_level.upper()
        self.logfile_log_color = logfile_log_color
        self.logfile_formatter = None
        self.asctime = asctime
        self.threadname = threadname
        self.splunk = splunk
        # self.log_line_template = "%(color_on)s[%(created)d] [%(threadName)s] [%(levelname)-8s] %(message)s%(color_off)s"
        self.log_line_template = "%(color_on)s[%(created)d] [%(threadName)s - %(name)s] [%(levelname)-8s] %(message)s%(color_off)s"
        self.console_handler = None
        self.logfile_handler = None
        # "%(asctime)s %(levelname)s src=%(name)s- %(message)s"

    def set_formatters(self):
        start_template = "%(color_on)s"
        timestamp = "[%(asctime)s]" if self.asctime is True else "[%(created)d]"
        name = (
            " [%(threadName)s - %(name)s]" if self.threadname is True else " [%(name)s]"
        )
        levelname = " [%(levelname)-8s]"
        end_template = " %(message)s%(color_off)s"
        self.log_line_template = (
            start_template + timestamp + name + levelname + end_template
        )

        self.console_formatter = LogFormatter(
            fmt=self.log_line_template, color=self.console_log_color
        )
        self.logfile_formatter = LogFormatter(
            fmt=self.log_line_template, color=self.logfile_log_color
        )

    def add_console_handler(self):
        # Create console handler
        if self.console_log_output == "stdout":
            self.console_log_output = sys.stdout
        elif self.console_log_output == "stderr":
            self.console_log_output = sys.stderr
        else:
            print(
                "Failed to set console output: invalid output: '%s'"
                % self.console_log_output
            )
            return False
        self.console_handler = logging.StreamHandler(self.console_log_output)

        # Set console log level
        try:
            self.console_handler.setLevel(self.console_log_level)
        except Exception:
            print(
                "Failed to set console log level: invalid level: '%s'"
                % self.console_log_level
            )
            return False

        # Create and set formatter, add console handler to logger
        self.console_handler.setFormatter(self.console_formatter)

    def add_file_handler(self):
        try:
            self.logfile_handler = logging.FileHandler(self.logfile_file)
        except Exception as e:
            print("Failed to set up log file: %s" % str(e))
            return False
        # Set log file log level
        try:
            self.logfile_handler.setLevel(self.logfile_log_level)
        except Exception:
            print(
                "Failed to set log file log level: invalid level: '%s'"
                % self.logfile_log_level
            )
            return False
        # Create and set formatter, add log file handler to logger
        self.logfile_handler.setFormatter(self.logfile_formatter)

    def setup_logger(self, console=True, logfile=True):
        # console_log_color, logfile_file, logfile_log_level, logfile_log_color, log_line_template
        # Create logger
        # For simplicity, we use the root logger, i.e. call 'logging.getLogger()'
        # without name argument. This way we can simply use module methods for
        # logging throughout the script. An alternative would be exporting
        # the logger, i.e. 'global logger; logger = logging.getLogger("<name>")'
        self.set_formatters()
        logger = logging.getLogger(self.__name__)
        # Set global log level to 'debug' (required for handler levels to work)
        logger.setLevel(self.log_level)

        # add console handler
        if console is True:
            self.add_console_handler()
            logger.addHandler(self.console_handler)

        # Create log file handler
        if logfile is True:
            self.add_file_handler()
            logger.addHandler(self.logfile_handler)

        # Success
        return logger


def splunk_logger(__name__, log_level="info"):
    """Initializes a logging object which logs process info to sys.stdout

    Returns
    -------
    logutil.log object
        logs process info to sys.stdout
    """
    levels = {
        "info": logutil.logging.INFO,
        "debug": logutil.logging.DEBUG,
        "error": logutil.logging.ERROR,
        "warning": logutil.logging.WARNING,
    }
    template = dict(
        level=levels[log_level],
        stream=sys.stdout,
        format="%(asctime)s %(levelname)s src=%(name)s- %(message)s",
        datefmt="%Y%j%H%M%S",
    )
    logger = logutil.create_logger(__name__, **template)
    logger.log.setLevel(log_level)
    return logger


# Command line test
def log_test():
    name = None
    loglevel = "warning"
    args = sys.argv
    if len(args) > 1:
        name = args[1]
    if len(args) > 2:
        loglevel = args[2]

    script_name = (
        name if name is not None else os.path.splitext(os.path.basename(sys.argv[0]))[0]
    )
    # Setup logging
    # script_name = os.path.splitext(os.path.basename(sys.argv[0]))[0]
    log = Logger(script_name, console_log_level=loglevel).setup_logger()
    if not log:
        print("Failed to setup logging, aborting.")
        return 1

    # Log some messages
    log.debug("Debug message")
    log.info("Info message")
    log.warning("Warning message")
    log.error("Error message")
    log.critical("Critical message")


# Call main function
if __name__ == "__main__":
    sys.exit(log_test())
