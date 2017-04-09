import argparse
from xworld import xworld_args
from learning import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parser():
    xworld_parser = xworld_args.parser()
    learning_parser = learning_args.parser()
    xworld_learning_parser = argparse.ArgumentParser(parents=[xworld_parser, learning_parser])
    return xworld_learning_parser
