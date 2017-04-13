from . import reinforce, actor_critic, async_actor_critic, cuda
import logging
logging.basicConfig(format='[%(levelname)s %(asctime)s %(filename)s:%(lineno)s] %(message)s',
                            level=logging.INFO)


def test(args, env, model):
    logging.info('testing')
    env.seed(args.seed)
    model.load_state_dict(torch.load(args.init_model_path))

    if args.method == 'reinforce':
        reinforcement_model = reinforce.Reinforce(args.gamma, model, optimizer)
    elif args.method == 'actor_critic':
        reinforcement_model = actor_critic.ActorCritic(args.gamma, model, optimizer)
    elif args.method == 'async_actor_critic':
        reinforcement_model = actor_critic.ActorCritic(args.gamma, model, optimizer)
    elif args.method == 'q_learn':
        reinforcement_model = q_learn.QLearn(args.gamma, model, optimizer)

    cumulative_rewards = []
    for i_episode in range(args.num_games):
        state, teacher = env.reset()
        cumulative_reward = []
        discount = 1.0
        for t in range(args.max_episode_length):  # Don't infinite loop while learning
            state_input = state.onehot_state.swapaxes(0,2).swapaxes(1,2)
            action = reinforcement_model.select_action(state_input)
            next_state, teacher, done = env.step(action[0, 0])
            reward = teacher.reward
            reinforcement_model.rewards.append(reward)
            cumulative_reward.append(reward * discount)
            discount *= args.gamma
            if done:
                break
        cumulative_rewards.append(numpy.sum(numpy.asarray(cumulative_reward)))
        if i_episode % args.log_interval == 0:
            logging.info('Episode {}\taverage cumulative reward: {:.2f}'.format(
                i_episode, numpy.mean(numpy.asarray(cumulative_rewards))))
            cumulative_rewards = []

