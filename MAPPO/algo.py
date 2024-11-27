import torch
import numpy as np
from grid_env.coverage import encode_action, decode_reward, get_critic_state

from .parameters import *

def get_advantages(agent, buffer, next_obs, next_done) -> tuple[torch.tensor, torch.tensor]:
    """
    Get advantage and return value based on the interaction with the environment
    """
    
    with torch.no_grad():
        critic_state = get_critic_state(next_obs, N_ENV)
        next_value = agent.get_value(critic_state)
        advantages = torch.zeros_like(buffer.rewards)
        lastgaelam = 0
        for t in reversed(range(N_STEP)):
            if t == N_STEP - 1:
                nextnonterminal = 1.0 - next_done.int()
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - buffer.dones[t + 1].int()
                nextvalues = buffer.value_pred[t + 1]

            # Reshape tensor from [N_ENV] to [N_ENV, N_AGENTS]
            nextvalues = nextvalues.unsqueeze(dim=-1).expand(-1,2)
            nextnonterminal = nextnonterminal.unsqueeze(dim=-1).expand(-1,2)
            value_pred = buffer.value_pred[t].unsqueeze(dim=-1).expand(-1,2)

            # delta.shape -> [N_ENV, N_AGENTS]
            delta = buffer.rewards[t] + GAMMA * nextvalues * nextnonterminal - value_pred   
            advantages[t] = lastgaelam = delta + GAMMA * GAE_LAMBDA * nextnonterminal * lastgaelam
        returns = advantages + buffer.value_pred

    return advantages, returns

def update_network(agent, optimizer, buffer, b_advantages, b_returns, logger):
    """
    Update network K times with the same experience divided in mini batches
    """
    index = np.arange(BATCH_SIZE)

    # Update networks K times
    for _ in range(K_EPOCHS):

        # Shuffle index to break correlations
        np.random.shuffle(index)
        
        # Update the network using minibatches 
        update_minibatch(agent, optimizer, buffer, b_advantages, b_returns, logger, index)


def update_minibatch(agent, optimizer, buffer, b_advantages, b_returns, logger, index, agent_id = 0) -> None:
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
        logger.add_loss(loss.item(), agent_id)
        
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
        optimizer.step()

def test_network(update, agent0, agent1, test_env, logger):
    """
    Execute n complete run in a test enviroment without exploration
    """
    
    if update % TEST_INTERVAL == 0:
        
        rew_data = np.zeros((2, TEST_RESET))
        len_data = np.zeros(TEST_RESET)
        
        # Collect data for 3 episode of test and log the mean reward and ep_lenght
        for i in range(TEST_RESET):
            stop_test = False
            test_reward = [0, 0]
            test_state, _ = test_env.reset(seed = SEED)
            ep_len = 0
            
            while not stop_test:
                # Get action with argmax
                with torch.no_grad():
                    test_state_tensor = torch.tensor(test_state)
                    action = encode_action(agent0.get_action_test(test_state_tensor[0]).cpu(),
                                            agent1.get_action_test(test_state_tensor[1]).cpu())
                    
                ns, rew, ter, trun, _ = test_env.step(action)
                
                rew0, rew1, = decode_reward(rew)
                
                test_reward[0] += rew0
                test_reward[1] += rew1
                test_state = ns
                ep_len +=1

                if ter or trun:
                    rew_data[:,i] = test_reward
                    len_data[i] = ep_len
                    stop_test = True

        if ter or trun:
            logger.add_test([np.mean(rew_data[0]), np.mean(rew_data[1])], np.mean(ep_len))
            stop_test = True