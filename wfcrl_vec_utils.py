"""
Parallel WFCRL environments for PPO rollout collection.

Must be a separate importable module (not defined in the notebook) so that
multiprocessing 'spawn' on Windows can find the worker function by module path.
"""
import multiprocessing as mp
import numpy as np


def _env_worker(seed, pipe, env_id):
    """Worker process: creates its own env from a seed, serves commands over a Pipe.

    Defined at module level so spawn can import it by path - no closure, no pickling issues.
    """
    from wfcrl import environments as envs  # import inside worker after spawn

    e = envs.make(
        env_id,
        max_num_steps=150,
        controls={"yaw": (-45, 45, 5)},
        continuous_control=True,
        log=False,
    )
    pipe.send("ready")

    while True:
        cmd, data = pipe.recv()

        if cmd == "reset":
            result = e.reset(**data)
            obs = result[0] if isinstance(result, tuple) else result
            pipe.send(obs)

        elif cmd == "step":
            obs, reward, terminated, truncated, info = e.step(data)
            pipe.send((obs, float(np.squeeze(reward)), bool(terminated), bool(truncated), info))

        elif cmd == "close":
            e.close()
            break


class ParallelVecEnv:
    """Parallel WFCRL environments using multiprocessing.

    Works on Windows (spawn) and Linux/Colab (spawn explicitly, or fork implicitly).
    The API mirrors gymnasium VectorEnv: reset() and step() return batched arrays.
    """

    def __init__(self, seeds, env_id):
        ctx = mp.get_context("spawn")
        self.n = len(seeds)
        self.env_id = env_id
        self.pipes = []
        self.procs = []

        for seed in seeds:
            parent, child = ctx.Pipe()
            proc = ctx.Process(target=_env_worker, args=(seed, child, env_id), daemon=True)
            proc.start()
            child.close()  # parent doesn't need the child end
            assert parent.recv() == "ready", f"Worker with seed {seed} failed to start"
            self.pipes.append(parent)
            self.procs.append(proc)

    def reset(self, seed=None, options=None):
        seeds = seed if isinstance(seed, list) else [seed] * self.n
        for i, pipe in enumerate(self.pipes):
            pipe.send(("reset", {"seed": seeds[i], "options": options}))
        obs_list = [pipe.recv() for pipe in self.pipes]
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}, {}

    def step(self, action_dict):
        for i, pipe in enumerate(self.pipes):
            pipe.send(("step", {k: v[i] for k, v in action_dict.items()}))
        results = [pipe.recv() for pipe in self.pipes]
        obs_list = [r[0] for r in results]
        return (
            {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]},
            np.array([r[1] for r in results]),
            np.array([r[2] for r in results]),
            np.array([r[3] for r in results]),
            [r[4] for r in results],
        )

    def close(self):
        for pipe in self.pipes:
            try:
                pipe.send(("close", None))
            except OSError:
                pass
        for proc in self.procs:
            proc.join(timeout=5)
