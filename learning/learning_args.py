import argparse
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def parser():
    arg_parser = argparse.ArgumentParser(description='Reinforcement learning', add_help=False)
    arg_parser.add_argument('--method', default='actor_critic',
                            help='reinforce, actor_critic, q_learn, ... (default: reinforce)')
    arg_parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                            help='discount factor for rewards (default: 0.99)')
    arg_parser.add_argument('--seed', type=int, default=432, metavar='N',
                            help='random seed (default: 1)')
    arg_parser.add_argument('--render', action='store_true',
                            help='render the environment')
    arg_parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                            help='interval between training status logs (default: 10)')
    arg_parser.add_argument('--opt_method', default='adam', choices=('sgd', 'adam', 'rmsprop'),
                            help='optimization method (default: adam)')
    arg_parser.add_argument('--num_games', type=int, default=10000,
                            help='total number of games')
    arg_parser.add_argument('--max_episode_length', type=int, default=100,
                            help='maximum length of an episode (default: 100)')
    arg_parser.add_argument('--batch_size', type=int, default=64,
                            help='batch size')
    arg_parser.add_argument('--use_gpu', action='store_true',
                            help='use gpu for training and testing')
    arg_parser.add_argument('--gpu_id', type=int, default=1,
                            help='use which gpu')
    arg_parser.add_argument('--replay_memory_size', type=int, default=100000,
                            help='replay memory size')
    arg_parser.add_argument('--learn_start', type=int, default=0,
                            help='number of games before learning start')
    arg_parser.add_argument('--save_dir', default='./models',
                            help='model save directory')
    arg_parser.add_argument('--log_dir', default='./log',
                            help='log save directory')
    arg_parser.add_argument('--init_model_path', default='',
                            help='initial model path')
    arg_parser.add_argument('--learning_rate', type=float, default=0.01, metavar='LR',
                            help='learning rate (default: 0.01)')
    arg_parser.add_argument('--train', action='store_true',
                            help='training mode')
    arg_parser.add_argument('--test', action='store_true',
                            help='testing mode (default mode)')
    arg_parser.add_argument('--save_interval', type=int, default=1000,
                            help='save model intervals (default: 1000)')
    arg_parser.add_argument('--num_processes', type=int, default=4,
                            help='how many training processes to use (default: 4)')

    return arg_parser
