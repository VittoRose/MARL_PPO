import torch
import numpy as np

from .parameters import *

def get_advantages(agent, buffer, next_obs, next_done, device) -> tuple[torch.tensor, torch.tensor]:
    with torch.no_grad():
        next_value = agent.get_value(next_obs).reshape(1, -1)
        advantages = torch.zeros_like(buffer.rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(n_step)):
            if t == n_step- 1:
                nextnonterminal = 1.0 - next_done.int()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - buffer.dones[t + 1].int()
                nextvalues = buffer.values[t + 1]
            delta = buffer.rewards[t] + GAMMA * nextvalues * nextnonterminal - buffer.values[t]
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + buffer.values

    return advantages, returns

def update_network(agent, optimizer, buffer, b_advantages, b_returns, logger):
    """
    Update network K times with the same experience divided in mini batches
    """
    index = np.arange(BATCH_SIZE)

    # Update networks K times
    for i in range(K_EPOCHS):

        # Shuffle index to break correlations
        np.random.shuffle(index)
        
        update_minibatch(agent, optimizer, buffer, b_advantages, b_returns, logger, index)


def update_minibatch(agent, optimizer, buffer, b_advantages, b_returns, logger, index) -> None:
    """
    Update actor and critic network using mini_batches from buffer 
    """
    
    b_obs, b_logprobs, b_actions, b_values = buffer.get_batch()

    for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):
        
        end = start + MINI_BATCH_SIZE

        min_batch_idx = index[start:end]

        _, newlogprob, entropy, newval = agent.get_action_and_value(b_obs[min_batch_idx], b_actions.long()[min_batch_idx])
        logratio = newlogprob - b_logprobs[min_batch_idx]
        ratio = logratio.exp()

        mb_advantages = b_advantages[min_batch_idx]

        # Normalize advantages
        mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

        # Normalize value 
        if VALUE_NORM:
            newval = (newval - newval.mean()) / (newval.std() + 1e-8)
            b_values[min_batch_idx] = (b_values[min_batch_idx] - b_values[min_batch_idx].mean()) / (b_values[min_batch_idx].std() + 1e-8)
        
        # Policy loss
        surr1 = -mb_advantages * ratio
        surr2 = -mb_advantages * torch.clamp(ratio, 1-CLIP, 1+CLIP)
        pg_loss = torch.max(surr1, surr2).mean()

        # Value loss
        if VALUE_CLIP:
            v_clip = b_values[min_batch_idx] + torch.clamp(newval.squeeze()-b_values[min_batch_idx], 1-CLIP, 1+CLIP)
            v_losses = torch.nn.functional.mse_loss(newval.squeeze(), b_returns[min_batch_idx])
            v_loss_max = torch.max(v_clip, v_losses)
            v_loss = 0.5*v_loss_max.mean()
        else:
            v_losses = torch.nn.functional.mse_loss(newval.squeeze(), b_returns[min_batch_idx])
            v_loss = v_losses.mean()

        # Entropy loss
        entropy_loss = entropy.mean()

        # Global loss function
        loss = pg_loss - ENTROPY_COEF*entropy_loss + VALUE_COEFF*v_loss
        logger.add_loss(loss.item())
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

def test_network(update, agent, test_env, logger, device):
    """
    Execute n complete run in a test enviroment without exploration
    """
    if update % TEST_INTERVAL:
        
        rew_data = np.zeros(TEST_RESET)
        len_data = np.zeros(TEST_RESET)
        
        # Collect data for 3 episode of test and log the mean reward and ep_lenght
        for i in range(TEST_RESET):
            stop_test = False
            test_reward = 0
            test_state, _ = test_env.reset(seed = SEED)
            ep_len = 0
            
            while not stop_test:
                # Get action with argmax
                with torch.no_grad():
                    test_state_tensor = torch.tensor(test_state).to(device)
                    action = agent.get_action_test(test_state_tensor)
                    
                ns, rew, ter, trun, _ = test_env.step(action)
                test_reward += rew
                test_state = ns
                ep_len +=1

                if ter or trun:
                    rew_data[i] = test_reward
                    len_data[i] = ep_len
                    stop_test = True

        if ter or trun:
            logger.add_test(np.mean(rew_data), np.mean(ep_len))
            stop_test = True