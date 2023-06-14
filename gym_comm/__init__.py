from gym.envs.registration import register

register(
    id='OvercookedMultiCommEnv-v0',
    entry_point='gym_comm.envs:OvercookedMultiEnv',
)