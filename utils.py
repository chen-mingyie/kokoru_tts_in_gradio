import logging

# setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
formatter = logging.Formatter(fmt='%(asctime)s - %(levelname)s - %(message)s', datefmt="%Y-%m-%d %H:%M:%S")
handler.setFormatter(formatter)
logger.addHandler(handler)