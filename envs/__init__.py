from envs.smax_env_wrapper import SMAXEnvWrapper, VecSMAXEnvWrapper


def make_env_wrapper(env_name: str, num_agents: int, max_steps: int):
    """Instantiate a single-env SMAX wrapper for the given scenario string
    (e.g. "3m", "2s3z", "8m")."""
    return SMAXEnvWrapper(env_name, num_agents, max_steps)


def make_vec_env_wrapper(env_name: str, num_agents: int, max_steps: int, num_envs: int):
    """Instantiate a vectorized SMAX wrapper for the given scenario string."""
    return VecSMAXEnvWrapper(env_name, num_agents, max_steps, num_envs)
