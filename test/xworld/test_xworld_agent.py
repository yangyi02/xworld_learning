from xworld import xworld_args, xworld_agent
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info("test xworld agent functions")
    args = xworld_args.parser().parse_args()
    agent = xworld_agent.XWorldAgent(args)
    for i in range(10):
        action = agent.random_action()
        logging.info(action)
    logging.info("test xworld agent functions done")

if __name__ == '__main__':
    main()
