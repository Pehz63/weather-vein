"""Worker launched via `mpiexec -n 1` so MPI.COMM_SELF.Spawn works."""
import sys
import pickle
import traceback
from multiprocessing.connection import Client


def _send_ok(conn, payload):
    conn.send_bytes(pickle.dumps(('ok', payload)))


def _send_err(conn):
    conn.send_bytes(pickle.dumps(('error', traceback.format_exc())))


def main():
    port = int(sys.argv[1])
    conn = Client(('localhost', port), authkey=b'mpi-worker')

    seeds, env_id = pickle.loads(conn.recv_bytes())

    try:
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
    except Exception:
        _send_err(conn)
        return

    while True:
        msg = pickle.loads(conn.recv_bytes())
        cmd = msg['cmd']

        if cmd == 'reset':
            try:
                results = []
                for i, e in enumerate(_envs):
                    obs, info = e.reset(seed=msg['seeds'][i], options=msg.get('options'))
                    results.append((obs, info))
                _send_ok(conn, results)
            except Exception:
                _send_err(conn)

        elif cmd == 'step':
            import numpy as np
            try:
                results = []
                for i, e in enumerate(_envs):
                    action = {k: v[i] for k, v in msg['action'].items()}
                    obs, reward, terminated, truncated, info = e.step(action)
                    results.append((obs, float(np.squeeze(reward)), bool(terminated), bool(truncated), info))
                _send_ok(conn, results)
            except Exception:
                _send_err(conn)

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
