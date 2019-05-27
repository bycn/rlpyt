
import torch

from rlpyt.algos.base import RlAlgorithm
from rlpyt.agents.base import AgentInputs
from rlpyt.agents.base_recurrent import AgentTrainInputs
from rlpyt.utils.quick_args import save__init__args
from rlpyt.utils.logging import logger
from rlpyt.replays.uniform import UniformReplayBuffer
from rlpyt.replays.prioritized import PrioritizedReplayBuffer
from rlpyt.utils.collections import namedarraytuple
from rlpyt.utils.tensor import select_at_indexes

OptInfo = namedarraytuple("OptInfo", ["loss", "gradNorm", "tdAbsError"])
OptData = None  # TODO
ReplaySamples = namedarraytuple("ReplaySamples",
    ["observation", "action", "reward", "done"])


class DQN(RlAlgorithm):
    """DQN with options for: Double-DQN, Dueling Architecture, n-step returns,
    prioritized replay."""

    def __init__(
            self,
            discount=0.99,
            batch_size=32,
            min_steps_learn=int(5e4),
            delta_clip=1,
            replay_size=int(1e6),
            training_intensity=8,  # Average number training uses per datum.
            target_update_steps=int(1e4),
            n_step_return=1,
            learning_rate=2.5e-4,
            OptimCls=torch.optim.Adam,
            optim_kwargs=None,
            initial_optim_state_dict=None,
            grad_norm_clip=10.,
            eps_init=1,
            eps_final=0.01,
            eps_final_min=None,  # set < eps_final to use vector-valued eps.
            eps_steps=int(1e6),
            eps_eval=0.001,
            dueling_dqn=False,
            priotized_replay=False,
            pri_alpha=0.6,
            pri_beta_init=0.4,
            pri_beta_final=1.,
            pri_beta_steps=int(50e6),
            default_priority=None,
            ):
        if optim_kwargs is None:
            optim_kwargs = dict()
        if default_priority is None:
            default_priority = delta_clip
        save__init__args(locals())

    def initialize(self, agent, n_itr, batch_spec, mid_batch_reset, examples):
        self.agent = agent
        if (self.eps_final_min is not None and
                self.eps_final_min != self.eps_final):  # vector-valued epsilon
            self.eps_init = self.eps_init * torch.ones(batch_spec.B)
            self.eps_final = torch.logspace(
                torch.log10(torch.tensor(self.eps_final_min)),
                torch.log10(torch.tensor(self.eps_final)),
                batch_spec.B)
        agent.set_epsilon_greedy(self.eps_init)
        agent.give_eval_epsilon_greedy(self.eps_eval)
        self.n_itr = n_itr
        self.optimizer = self.OptimCls(agent.parameters(),
            lr=self.learning_rate, **self.optim_kwargs)
        if self.initial_optim_state_dict is not None:
            self.optimizer.load_state_dict(self.initial_optim_state_dict)
        sample_bs = batch_spec.size
        train_bs = self.batch_size

        assert (self.training_intensity * sample_bs) % train_bs == 0
        self.updates_per_optimize = int((self.training_intensity * sample_bs) //
            train_bs)
        logger.log(f"From sampler batch size {sample_bs}, DQN "
            f"batch size {train_bs}, and training intensity "
            f"{self.training_intensity}, computed {self.updates_per_optimize} "
            f"updates per iteration.")

        self.eps_itr = max(1, self.eps_steps // sample_bs)
        self.target_update_itr = max(1, self.target_update_steps // sample_bs)
        self.min_itr_learn = self.min_steps_learn // sample_bs
        if self.prioritized_replay:
            self.pri_beta_itr = max(1, self.pri_beta_steps // sample_bs)

        # Annoying TODO: make example--get from sampler?  Eat first.
        replay_example = ReplaySamples(
            observation=examples["observation"],
            action=examples["action"],
            reward=examples["reward"],
            done=examples["done"],
        )
        replay_kwargs = dict(
            example=replay_example,
            size=self.replay_size,
            B=batch_spec.B,
            discount=self.dicount,
            n_step_return=self.n_step_return,
        )
        if self.prioritize_replay:
            replay_kwargs.update(dict(
                alpha=self.pri_alpha,
                beta=self.pri_beta_init,
                default_priority=self.default_priority,
            ))
            ReplayCls = PrioritizedReplayBuffer
        else:
            ReplayCls = UniformReplayBuffer
        self.replay_buffer = ReplayCls(**replay_kwargs)

    def optimize_agent(self, samples, itr):
        replay_samples = ReplaySamples(
            observation=samples.env.observation,
            action=samples.agent.action,
            reward=samples.env.reward,
            done=samples.env.done,
        )
        self.replay_buffer.append_samples(replay_samples)
        if itr < self.min_itr_learn:
            return OptData(), OptInfo()  # TODO fix for empty
        opt_info = OptInfo(loss=[], gradNorm=[], tdAbsError=[])
        for _ in range(self.updates_per_optimize):
            mb_samples = self.replay_buffer.sample(self.batch_size)
            self.optimizer.zero_grad()
            loss, td_abs_errors = self.loss(mb_samples)
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                self.agent.model.parameters(), self.clip_grad_norm)
            self.optimizer.step()
            if self.prioritized_replay:
                self.replay_buffer.update_batch_priorities(td_abs_errors)
            opt_info.loss.append(loss.item())
            opt_info.gradNorm.append(grad_norm)
            opt_info.tdAbsError.extend(td_abs_errors[::8])  # Downsample for stats.
        if itr % self.update_target_itr == 0:
            self.agent.update_target()
        self.update_itr_hyperparams(itr)
        return OptData(), opt_info  # TODO: fix opt_data

    def loss(self, samples):
        if self.agent.recurrent:
            raise NotImplementedError
        qs = self.agent(*samples.agent_inputs)
        q = select_at_indexes(samples.action, qs)
        with torch.no_grad():
            if self.double_dqn:
                next_qs = self.agent(*samples.next_agent_inputs)
                next_a = torch.argmax(next_qs, dim=-1)
                next_qs = self.agent.target_q(*next_agent_inputs)
                next_q = select_at_indexes(next_a, next_qs)
            else:
                next_qs = self.agent.target_q(*next_agent_inputs)
                next_q = torch.argmax(next_qs, dim=-1)
        disc_next_q = (self.discount ** self.n_step_return) * next_q
        y = samples.return_ + (1 - samples.done_n) * disc_next_q
        delta = y - q
        losses = 0.5 * delta ** 2
        abs_delta = abs(delta)
        if self.delta_clip is not None:  # Huber loss.
            b = self.delta_clip * (abs_delta - self.delta_clip / 2)
            losses = torch.where(abs_delta <= self.delta_clip, losses, b)
        if self.prioritized_replay:
            losses *= samples.is_weights
        loss = torch.mean(losses)
        td_abs_errors = torch.clamp(abs_delta, 0, self.delta_clip)

        return loss, td_abs_errors

    def update_itr_hyperparams(self, itr):
        if itr < self.eps_itr:  # Epsilon can be vector-valued.
            prog = min(1, itr / self.eps_itr)
            new_eps = prog * self.eps_final + (1 - prog) * self.eps_init
            self.agent.set_eps_greedy(new_eps)
        if self.prioritized_replay and itr < self.pri_beta_itr:
            prog = min(1, itr / self.pri_beta_itr)
            new_beta = (prog * self.pri_beta_final +
                (1 - prog) * self.pri_beta_init)
            self.replay_buffer.set_beta(new_beta)
