import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parser():
    parser = argparse.ArgumentParser(description='PyTorch reinforcement learning', add_help=False)
    parser.add_argument('--method', default='actor_critic',
                        help='reinforce, actor_critic, q_learn, ... (default: reinforce)')
    parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                        help='discount factor for rewards (default: 0.99)')
    parser.add_argument('--seed', type=int, default=432, metavar='N',
                        help='random seed (default: 1)')
    parser.add_argument('--render', action='store_true',
                        help='render the environment')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='interval between training status logs (default: 10)')
    parser.add_argument('--opt_method', default='adam', choices=('sgd', 'adam', 'rmsprop'),
                        help='optimization method (default: adam)')
    parser.add_argument('--num_games', type=int, default=10000,
                        help='total number of games')
    parser.add_argument('--max_episode_length', type=int, default=100,
                        help='maximum length of an episode (default: 100)')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch size')
    parser.add_argument('--use_gpu', action='store_true',
                        help='use gpu for training and testing')
    parser.add_argument('--gpu_id', type=int, default=1,
                        help='use which gpu')
    parser.add_argument('--replay_memory_size', type=int, default=100000,
                        help='replay memory size')
    parser.add_argument('--learn_start', type=int, default=0,
                        help='number of games before learning start')
    parser.add_argument('--save_dir', default='./models',
                        help='model save directory')
    parser.add_argument('--log_dir', default='./log',
                        help='log save directory')
    parser.add_argument('--init_model', default='',
                        help='initial model path')
    parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--train', action='store_true',
                        help='training mode')
    parser.add_argument('--test', action='store_true',
                        help='testing mode (default mode)')
    parser.add_argument('--save_interval', type=int, default=1000,
                        help='save model intervals (default: 1000)')

    return parser


def main():
    arg_parser = parser()
    args = arg_parser.parse_args()
    logging.info(args.method)
    logging.info(args.gamma)
    logging.info(args.seed)
    logging.info(args.render)
    logging.info(args.log_interval)
    logging.info(args.opt_method)
    logging.info(args.num_games)
    logging.info(args.batch_size)
    logging.info(args.use_gpu)
    logging.info(args.gpu_id)
    logging.info(args.replay_memory_size)
    logging.info(args.learn_start)
    logging.info(args.save_dir)
    logging.info(args.log_dir)
    logging.info(args.init_model)
    logging.info(args.learning_rate)
    logging.info(args.max_episode_length)
    logging.info(args.train)
    logging.info(args.test)

if __name__ == '__main__':
    main()
