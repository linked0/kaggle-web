import logging, sys

logging.basicConfig(format="%(filename)s:%(funcName)s:%(message)s",level=logging.DEBUG,stream=sys.stderr)

def log(msg):
    logging.debug(msg)