from contest.sonic_util import make_env, AllowBacktracking

def main():
    env = make_env(False, False)
    env = AllowBacktracking(env)
    for _ in range(30):
        env.reset()
        while True:
            _obs, _rew, done, _info = env.step(env.action_space.sample())
            env.render()
            if done:
                print('done')
                break

if __name__ == '__main__':
    main()
