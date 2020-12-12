import copy
import numpy as np
from multiprocessing import Process, Pipe, Value


def worker(worker_id, env, master_end, worker_end):
    master_end.close()  # Forbid worker to use the master end for messaging
    env.seed(worker_id)

    while True:
        cmd, data = worker_end.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)

            if done:
                ob = env.reset()
            worker_end.send((ob, reward, done, info))
        elif cmd == 'reset':
            ob = env.reset()
            worker_end.send(ob)
        elif cmd == 'render':
            #env.render()
            worker_end.send(env.render(mode='rgb_array'))
        elif cmd == 'close':
            worker_end.close()
            break
        elif cmd == 'get_episode_rewards':
            episode_rewards = env.get_episode_rewards()
            worker_end.send(episode_rewards)
        else:
            raise NotImplementedError

def _flatten_list(l):
    assert isinstance(l, (list, tuple))
    assert len(l) > 0
    assert all([len(l_) > 0 for l_ in l])

    return [l__ for l_ in l for l__ in l_]

class ParallelEnv(object):
    """
    This class
    """

    def __init__(self, num_processes, env):
        self.nenvs = num_processes
        self.waiting = False
        self.closed = False
        self.workers = []
        self.action_space = env.action_space
        self.observation_space = env.observation_space

        self.master_ends, self.send_ends = zip(*[Pipe() for _ in range(self.nenvs)])
        for worker_id, (master_end, send_end) in enumerate(zip(self.master_ends, self.send_ends)):
            p = Process(target=worker, args=(worker_id, copy.deepcopy(env), master_end, send_end))
            p.start()
            self.workers.append(p)

    def render(self):
        for pipe in self.master_ends:
            pipe.send(('render', None))
        imgs = [pipe.recv() for pipe in self.master_ends]
        imgs = _flatten_list(imgs)
        return imgs




    def step(self, actions):
        """
        Perform step for each environment and return the stacked transitions
        """
        for master_end, action in zip(self.master_ends, actions):
            master_end.send(('step', action))
        self.waiting = True

        results = [master_end.recv() for master_end in self.master_ends]
        self.waiting = False
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), np.stack(infos)

    def reset(self):
        for master_end in self.master_ends:
            master_end.send(('reset', None))
        results = [master_end.recv() for master_end in self.master_ends]
        return np.stack(results)

    def close(self):
        if self.closed:
            return
        if self.waiting:
            [master_end.recv() for master_end in self.master_ends]
        for master_end in self.master_ends:
            master_end.send(('close', None))
        for worker in self.workers:
            worker.join()
        self.closed = True

    def get_episode_rewards(self):
        for master_end in self.master_ends:
            master_end.send(('get_episode_rewards', None))
        ep_per_worker = 100 // self.nenvs
        results = [master_end.recv()[-ep_per_worker:] for master_end in self.master_ends]
        return np.concatenate(results)
