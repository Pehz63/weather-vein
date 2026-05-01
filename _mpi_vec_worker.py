"""Worker launched via `mpiexec -n 1` so MPI.COMM_SELF.Spawn works."""
import sys
import pickle
from multiprocessing.connection import Client


def main():
    port = int(sys.argv[1])
    conn = Client(('localhost', port), authkey=b'mpi-worker')

    seeds, env_id = pickle.loads(conn.recv_bytes())

    from wfcrl import environments as envs
    _envs = [
        envs.make(
            env_id,
            max_num_steps=150,
            controls={"yaw": (-45, 45, 5)},
            continuous_control=True,
            log=False,
        )
        for _ in seeds
    ]
    conn.send_bytes(b'ready')

    while True:
        msg = pickle.loads(conn.recv_bytes())
        cmd = msg['cmd']

        if cmd == 'reset':
            results = []
            for i, e in enumerate(_envs):
                obs, info = e.reset(seed=msg['seeds'][i], options=msg.get('options'))
                results.append((obs, info))
            conn.send_bytes(pickle.dumps(results))

        elif cmd == 'step':
            import numpy as np
            results = []
            for i, e in enumerate(_envs):
                action = {k: v[i] for k, v in msg['action'].items()}
                obs, reward, terminated, truncated, info = e.step(action)
                results.append((obs, float(np.squeeze(reward)), bool(terminated), bool(truncated), info))
            conn.send_bytes(pickle.dumps(results))

        elif cmd == 'close':
            for e in _envs:
                try:
                    e.close()
                except Exception:
                    pass
            conn.send_bytes(b'closed')
            break


if __name__ == '__main__':
    main()
