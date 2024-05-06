from os.path import exists
from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback

from redpitayaenv import RedPitayaEnv
from rpscope import RedPitayaScope


def create_env(skip: int = 15):
    #env = RedPitayaEnv(RedPitayaScope('169.254.167.128'))
    #env = SkipSteps(env, skip)
    return #env


def ppo_model(env, verbose: int = 1, n_steps: int = 2048 * 8,
              batch_size: int = 64, n_epochs: int = 10, gamma: float = 0.999,
              device: str = 'cpu', file_name=None):
    assert device in ['cuda', 'mps', 'cpu']
    set_random_seed(42, 'cuda' == device)
    if file_name is not None and exists(file_name + '.zip'):
        print('\nLoading checkpoint')
        model = PPO.load(file_name)
        model.set_env(env)
        model.rollout_buffer.reset()
    else:
        model = PPO('MlpPolicy',
                    env,
                    verbose=verbose,
                    n_steps=n_steps,
                    batch_size=batch_size,
                    n_epochs=n_epochs,
                    gamma=gamma,
                    device=device)
    return model


def run():
    RedPitayaScope(hostname='169.254.167.128')
    """model = ppo_model(
        create_env(),
        verbose=1,
        n_steps=256,
        n_epochs=3,
        device='cpu',
        batch_size=16
    )
    checkpoint_callback = CheckpointCallback(
        save_freq=1024*10,
        save_path='models',
        name_prefix='pid_n512_e10_b64',
        verbose=1
    )
    model.learn(total_timesteps=1024 * 50_000, callback=checkpoint_callback)   # six days"""


if __name__ == '__main__':
    run()
