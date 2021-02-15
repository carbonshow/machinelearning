import torch
from torch.nn import functional as nnf
import torch.optim as optim
from replay_buffer import Transition


class QLearn(object):
    BATCH_SIZE = 128
    GAMMA = 0.999
    TARGET_UPDATE = 10

    def __init__(self, policy_nn, memory):
        super(QLearn, self).__init__()
        self.optimizer = optim.RMSprop(policy_nn.parameters())
        self.memory = memory
        self.policy_nn = policy_nn

    def optimize_model(self, target_nn, device):
        if len(self.memory) < QLearn.BATCH_SIZE:
            return
        transitions = self.memory.sample(QLearn.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                           if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_nn(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(QLearn.BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_nn(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * QLearn.GAMMA) + reward_batch

        # Compute Huber loss
        loss = nnf.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_nn.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
