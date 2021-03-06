
import numpy as np

from rlpyt.samplers.collectors import BaseEvalCollector
from rlpyt.agents.base import AgentInputs
from rlpyt.utils.buffer import buffer_from_example, torchify_buffer, numpify_buffer
from rlpyt.utils.logging import logger
from rlpyt.utils.quick_args import save__init__args

# For sampling, serial sampler can use Cpu collectors.


class SerialEvalCollector(BaseEvalCollector):
    """Does not record intermediate data."""

    def __init__(
            self,
            envs,
            agent,
            TrajInfoCls,
            max_T,
            max_trajectories=None,
            ):
        save__init__args(locals())

    def collect_evaluation(self, itr):
        traj_infos = [self.TrajInfoCls() for _ in range(len(self.envs))]
        completed_traj_infos = list()
        observations = list()
        should_visualize = False 
        use_env = False
        if should_visualize:
            vis_frames = [[] for _ in range(len(self.envs))]
        for env in self.envs:
            observations.append(env.reset())
        observation = buffer_from_example(observations[0], len(self.envs))
        for b, o in enumerate(observations):
            observation[b] = o
        # from IPython import embed; embed()
        action = buffer_from_example(self.envs[0].action_space.null_value(),
            len(self.envs))
        reward = np.zeros(len(self.envs), dtype="float32")
        obs_pyt, act_pyt, rew_pyt = torchify_buffer((observation, action, reward))
        self.agent.reset()
        self.agent.eval_mode(itr)

        action_bag = []

        for t in range(self.max_T):
            # if t < 100:
            #     pass
                # print(act_pyt, "act_pyt")
                # print(obs_pyt, act_pyt, rew_pyt)
            act_pyt, agent_info = self.agent.step(obs_pyt, act_pyt, rew_pyt)
            action = numpify_buffer(act_pyt)

            action_bag.append(action)

            for b, env in enumerate(self.envs):
                # print("action", action)
                # print("action[b]", action[b])
                o, r, d, env_info = env.step(action[b])
                if should_visualize:
                    if use_env:
                        vis_frames[b].append(np.transpose(env.render("rgb_array"), (2,0,1)))
                    else:
                        try:
                            vis_frames[b].append(o.observation)
                        except:
                            vis_frames[b].append(o)
                traj_infos[b].step(observation[b], action[b], r, d,
                    agent_info[b], env_info)
                if getattr(env_info, "traj_done", d):
                    completed_traj_infos.append(traj_infos[b].terminate(o))
                    traj_infos[b] = self.TrajInfoCls()
                    o = env.reset()
                if d:
                    action[b] = 0  # Prev_action for next step.
                    r = 0
                    self.agent.reset_one(idx=b)
                observation[b] = o
                reward[b] = r
            if (self.max_trajectories is not None and
                    len(completed_traj_infos) >= self.max_trajectories):
                logger.log("Evaluation reached max num trajectories "
                    f"({self.max_trajectories}).")
                print(action_bag[::10])
                break
        if t == self.max_T - 1:
            logger.log("Evaluation reached max num time steps "
                f"({self.max_T}).")
        if should_visualize:
            # temporarily only save the first environment
            logger.save_rollout_vis([vis_frames[0]], itr)
        return completed_traj_infos
