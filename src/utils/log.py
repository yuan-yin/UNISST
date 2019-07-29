import logging


class MyFormatter(logging.Formatter):
    dbg_fmt = "%(asctime)s DEBUG  - %(module)s: [%(funcName)s:%(lineno)d] %(msg)s"
    info_fmt = "%(asctime)s LOG    - %(msg)s"
    error_fmt = "%(asctime)s ERROR - %(module)s: [%(funcName)s:%(lineno)d] %(msg)s"

    def __init__(self):
        super().__init__(datefmt='%m-%d %H:%M:%S', style='%')

    def format(self, record):

        # Save the original format configured by the user
        # when the logger formatter was instantiated
        format_orig = self._style._fmt

        # Replace the original format with one customized by logging level
        if record.levelno == 31:
            self._style._fmt = MyFormatter.dbg_fmt
        elif record.levelno == 32:
            self._style._fmt = MyFormatter.info_fmt
        else:
            self._style._fmt = format_orig
        # elif record.levelno == 40:
        #     self._style._fmt = MyFormatter.error_fmt

        # Call the original formatter class to do the grunt work
        result = logging.Formatter.format(self, record)

        # Restore the original format configured by the user
        return result


def setup_custom_logger(name):
    fmt = MyFormatter()

    handler = logging.StreamHandler()
    handler.setFormatter(fmt)

    logger = logging.getLogger(name)
    logger.addHandler(handler)
    return logger
