from learning import learning_args
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                    level=logging.INFO)


def main():
    logging.info('test learning args functions')
    args = learning_args.parser().parse_args()
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
    logging.info(args.num_processes)
    logging.info('test learning args functions done')

if __name__ == '__main__':
    main()
