import numpy as np

def objective(yaws, models, freewind):
    X = np.concatenate([yaws, freewind]).reshape(1, -1)
    return -sum(m.predict(X)[0] for m in models)

def step_policy(i, env):
    '''Randmoly samples actions'''
    joint_action = {"yaw": np.zeros(env.num_turbines)}
    mask = np.random.random(env.num_turbines) < 0.3 # proba of a turbine to get controled
    joint_action["yaw"][mask] = np.random.uniform(-5, 5, size=mask.sum())
    return joint_action

def obs_to_row(obs, reward, power, step):
    '''Writes all episode results into a dict'''
    row = {"step": step, "reward": reward[0]}
    for key, val in obs.items():
        val = np.atleast_1d(val)
        if len(val) == 1:
            row[key] = val[0]
        else:
            for i, v in enumerate(val):
                row[f"{key}_{i}"] = v
    for i in range(len(power)):
        row[f"power_{i}"] = power[i]
    return row