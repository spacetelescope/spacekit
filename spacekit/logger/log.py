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
        console=True,
        logfile=True,
        console_log_output="stdout",
        console_log_level="info",
        console_log_color=True,
        logfile_log_level="debug",
        logfile_log_color=False,
        logdir=".",
        asctime=True,
        threadname=False,
        verbose=False,
        log=None,
    ):
        self.__name__ = (
            f"spacekit.{script_name}" if script_name != "spacekit" else script_name
        )
        self.short_name = script_name
        self.console = console
        self.logfile = logfile
        self.log_level = logging.INFO
        self.console_log_output = console_log_output.lower()
        self.console_log_level = console_log_level.upper()
        self.console_log_color = console_log_color
        self.console_formatter = None
        self.logdir = logdir
        self.logfile_file = os.path.join(self.logdir, "spacekit" + ".log")
        self.logfile_log_level = logfile_log_level.upper()
        self.logfile_log_color = logfile_log_color
        self.logfile_formatter = None
        self.asctime = asctime
        self.threadname = threadname
        self.console_handler = None
        self.logfile_handler = None
        self.verbose = verbose
        self.log = log
        self.log_line_template = self.set_log_line_template()

    def set_name(self):
        if self.verbose is True:
            name = (
                " [%(threadName)s - %(name)-16s]"
                if self.threadname is True
                else " [%(name)-16s]"
            )
        else:
            name = (
                " [%(threadName)s - %(name)s]"
                if self.threadname is True
                else " [%(name)s]"
            )
        return name

    def set_log_line_template(self):
        start_template = "%(color_on)s"
        timestamp = "[%(asctime)s]" if self.asctime is True else "[%(created)d]"
        name = self.set_name()
        levelname = " [%(levelname)-8s]"
        end_template = " %(message)s%(color_off)s"
        self.log_line_template = (
            start_template + timestamp + name + levelname + end_template
        )

    def set_formatters(self):
        self.set_log_line_template()

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

    def add_handlers(self, logger):
        # add console handler
        if self.console is True:
            self.add_console_handler()
            logger.addHandler(self.console_handler)

        # Create log file handler
        if self.logfile is True:
            os.makedirs(self.logdir, exist_ok=True)
            self.add_file_handler()
            logger.addHandler(self.logfile_handler)
        return logger

    def update_handlers(self, logger):
        try:
            for handler in logger.handlers:
                if isinstance(handler, logging.FileHandler):
                    handler.setLevel(self.logfile_log_level)
                    handler.setFormatter(self.logfile_formatter)
                elif isinstance(handler, logging.StreamHandler):
                    handler.setLevel(self.console_log_level)
                    handler.setFormatter(self.console_formatter)
        except Exception:
            print("Failed to modify handlers.")
        return logger

    def setup_logger(self, logger=None):
        self.set_formatters()
        # Create logger
        if logger is None:
            logger = logging.getLogger("spacekit")

        # Set global log level to 'debug' (required for handler levels to work)
        logger.setLevel(self.log_level)

        # don't add additional handlers if they already exist
        if logger.hasHandlers() is True:
            logger = self.update_handlers(logger)
        else:
            logger = self.add_handlers(logger)

        if self.verbose:  # identify source module in each log statement
            logger = logger.getChild(self.short_name)
            logger.name = self.__name__[:16]

        # Success
        return logger

    def spacekit_logger(self):
        # inherit settings from script via `log` attr
        if self.log:
            logger = self.setup_logger(logger=self.log)
        else:
            # Called by submodules (not scripts)
            logger = self.setup_logger()
        return logger


global SPACEKIT_LOG
SPACEKIT_LOG = Logger("spacekit").setup_logger()


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
