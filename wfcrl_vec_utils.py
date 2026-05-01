"""
Parallel WFCRL environments for PPO rollout collection.

Must be a separate importable module (not defined in the notebook) so that
multiprocessing 'spawn' on Windows can find the worker function by module path.
"""
import multiprocessing as mp
import numpy as np
import os
import pickle
import shutil
import socket
import subprocess
import sys
from multiprocessing.connection import Listener


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


class SequentialVecEnv:
    """Single-process vectorized env for simulators that require MPI (e.g. Fastfarm).

    Runs all N envs in the calling process sequentially - no multiprocessing,
    so MPI.COMM_SELF.Spawn() works correctly.
    """

    def __init__(self, seeds, env_id):
        from wfcrl import environments as envs
        self.n = len(seeds)
        self.envs = [
            envs.make(
                env_id,
                max_num_steps=150,
                controls={"yaw": (-45, 45, 5)},
                continuous_control=True,
                log=False,
            )
            for _ in seeds
        ]

    def reset(self, seed=None, options=None):
        seeds = seed if isinstance(seed, list) else [seed] * self.n
        obs_list = [
            self.envs[i].reset(seed=seeds[i], options=options)[0]
            for i in range(self.n)
        ]
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}, {}

    def step(self, action_dict):
        results = []
        for i, env in enumerate(self.envs):
            obs, reward, terminated, truncated, info = env.step(
                {k: v[i] for k, v in action_dict.items()}
            )
            results.append((obs, float(np.squeeze(reward)), bool(terminated), bool(truncated), info))
        obs_list = [r[0] for r in results]
        return (
            {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]},
            np.array([r[1] for r in results]),
            np.array([r[2] for r in results]),
            np.array([r[3] for r in results]),
            [r[4] for r in results],
        )

    def close(self):
        for env in self.envs:
            try:
                env.close()
            except Exception:
                pass


class MpiSequentialVecEnv:
    """Sequential VecEnv for MPI-based simulators (e.g. Fastfarm).

    Worker runs under `mpiexec -n 1` so MPI.COMM_SELF.Spawn works without
    launching the entire Jupyter kernel under mpiexec.
    """

    def __init__(self, seeds, env_id):
        mpiexec = shutil.which('mpiexec') or shutil.which('msmpiexec') or 'mpiexec'
        worker = os.path.join(os.path.dirname(os.path.abspath(__file__)), '_mpi_vec_worker.py')

        self._listener = Listener(('localhost', 0), authkey=b'mpi-worker')
        port = self._listener.address[1]
        self._proc = subprocess.Popen([mpiexec, '-n', '1', sys.executable, worker, str(port)])
        self._conn = self._listener.accept()
        self._conn.send_bytes(pickle.dumps((seeds, env_id)))
        assert self._conn.recv_bytes() == b'ready', 'MPI worker failed to initialize'
        self.n = len(seeds)

    def reset(self, seed=None, options=None):
        seeds = seed if isinstance(seed, list) else [seed] * self.n
        self._conn.send_bytes(pickle.dumps({'cmd': 'reset', 'seeds': seeds, 'options': options}))
        results = pickle.loads(self._conn.recv_bytes())
        obs_list = [r[0] for r in results]
        return {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]}, {}

    def step(self, action_dict):
        self._conn.send_bytes(pickle.dumps({'cmd': 'step', 'action': action_dict}))
        results = pickle.loads(self._conn.recv_bytes())
        obs_list = [r[0] for r in results]
        return (
            {k: np.stack([o[k] for o in obs_list]) for k in obs_list[0]},
            np.array([r[1] for r in results]),
            np.array([r[2] for r in results]),
            np.array([r[3] for r in results]),
            [r[4] for r in results],
        )

    def close(self):
        try:
            self._conn.send_bytes(pickle.dumps({'cmd': 'close'}))
            self._conn.recv_bytes()
        except Exception:
            pass
        self._conn.close()
        self._listener.close()
        self._proc.wait(timeout=10)
