import torch
import numpy as np
from grid_env.coverage import encode_action, decode_reward, get_critic_state

from .parameters import *

def get_advantages(agent, buffer, next_obs, next_done) -> tuple[torch.tensor, torch.tensor]:
    """
    Get advantage and return value based on the interaction with the environment
    
    :param agent: Actor_critic class, used to evaluate last obs value
    :param next_obs: Last observation for the N_STEPth timestamp
    :param next_done: Last done check
    
    :return advantages: estimation using GAE, [N_STEP, N_ENV, N_AGENT]
    :return values: target for critic network, [N_STEP, N_ENV, N_AGENT]
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
        
        # Add a dimension to value to get shape: [N_STEP, N_ENV, N_AGENT]
        value = buffer.value_pred.unsqueeze(dim=-1).repeat(1,1,2)
        
        returns = advantages + value

    return advantages, returns

def update_network(agent, buffer, advantages, returns, logger):
    """
    Update network K times with the same experience divided in mini batches
    
    :param agent: Actor_critic wrapper to update
    :param buffer: Rollout Buffer with experience 
    :param advantage: Advantage data, shape: [N_STEP, N_ENV, N_AGENT]
    :param returns: returns for critic network, shape [N_STEP, N_ENV, N_AGENT]
    """
    
    # From [N_STEP, N_ENV, N_AGENT] to [BATCH_SIZE] = [N_STEP*N_ENV*N_AGENT]
    b_advantages = advantages.flatten()
    b_returns = returns.flatten()
    
    # Update networks K times
    for _ in range(K_EPOCHS):

        # Shuffle index to break correlations 
        actor_i = torch.randperm(BATCH_SIZE)
        critic_i = torch.randperm(N_STEP*N_ENV)
        
        # Update the network using minibatches 
        update_minibatch(agent, buffer, b_advantages, b_returns, logger, actor_i, critic_i)


def update_minibatch(agent, buffer, b_advantages, b_returns, logger, actor_i, critic_i) -> None:
    """
    Update actor and critic network using mini_batches from buffer 
    """
        
    # Get the data in the buffer with the proper shape
    b_obs, b_logprobs, b_actions, b_critic, b_values = buffer.get_batch()

    for start in range(0, BATCH_SIZE, MINI_BATCH_SIZE):
        
        # Index for minibatch
        end = start + MINI_BATCH_SIZE
        idx_a = actor_i[start:end]
        idx_c = critic_i[start//N_AGENT:end//N_AGENT]

        # Create minibatch for actor 
        mb_obs = b_obs[idx_a]
        mb_actions = b_actions[idx_a]
        mb_logprobs = b_logprobs[idx_a]
        mb_advantages = b_advantages[idx_a]

        actor_loss = update_actor(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages)

        # Prepare the minibatches for critic
        mb_values = b_values[idx_c]
        mb_critic = b_critic[idx_c]
        mb_returns = b_returns[idx_c]
        
        critic_loss = update_critic(agent, mb_critic, mb_values, mb_returns)
        
        logger.add_loss(actor_loss, critic_loss)
        
        
def update_actor(agent, mb_obs, mb_actions, mb_logprobs, mb_advantages) -> float:
    """
    Update actor with clip and entropy loss from a minibatch
    
    :param agent: Actor_critic wrapper to update
    :param mb_obs: Minibatch of observation. Shape [MINIBATCH, obs]
    :param mb_actions: Minibatch of actions. Shape [MINIBATCH]
    :param mb_logprobs: Minibatch of logprob. Shape [MINIBATCH]
    :param mb_advantages: Minibatch of advantages. Shape [MINIBATCH]
    
    :return loss: loss value
    """
    newlogprob, entropy = agent.evaluate_action(mb_obs, mb_actions)
    
    # Evaluation for actor loss
    logratio = newlogprob - mb_logprobs
    ratio = logratio.exp()

    # Normalize advantages
    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

    # Policy loss
    surr1 = -mb_advantages * ratio
    surr2 = -mb_advantages * torch.clamp(ratio, 1-CLIP, 1+CLIP)
    l1 = torch.max(surr1, surr2).mean()
    entropy_loss = entropy.mean()
    
    # Clip loss + entropy bonus
    loss = l1 - ENTROPY_COEF*entropy_loss
    
    # Backward
    agent.actor_optim.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
    agent.actor_optim.step()
    
    return loss.item()

def update_critic(agent, mb_critic, mb_values, mb_returns):
    """
    Update actor with clip and entropy loss from a minibatch
    Minibatches size: [N_STEP*N_ENV]
    
    :param agent: Actor_critic wrapper to update
    :param mb_obs: Minibatch of observation. Shape [N_STEP*N_ENV, obs]
    :param mb_obs: Minibatch of observation. Shape [N_STEP*N_ENV]
    """
    
    newval = agent.get_value(mb_critic)
    
    # Normalize value 
    if VALUE_NORM:
        newval = (newval - newval.mean()) / (newval.std() + 1e-8)
        mb_values = (mb_values - mb_values.mean()) / (mb_values.std() + 1e-8)

    # Value loss
    if VALUE_CLIP:
        v_clip = mb_values + torch.clamp(newval-mb_values, 1-CLIP, 1+CLIP)
        v_losses = torch.nn.functional.mse_loss(newval, mb_returns)
        v_loss_max = torch.max(v_clip, v_losses)
        v_loss = 0.5*v_loss_max.mean()
    else:
        v_losses = torch.nn.functional.mse_loss(newval, mb_returns)
        v_loss = v_losses.mean()
    
    # Backpropagation
    agent.critic_optim.zero_grad()
    v_loss.backward()
    torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
    agent.critic_optim.step()
    
    return v_loss.item()
        
def test_network(update, agent, test_env, logger):
    """
    Execute n complete run in a test environment using the action with the maximum probability from the policy
    
    :param update: Current epoch number
    :param agent: Actor_critic wrapper
    :param test_env: Single gym GridCoverage environment
    :param logger: InfoPlot object
    """
    
    if update % TEST_INTERVAL == 0:
        
        rew_data = np.zeros((2, TEST_RESET))
        len_data = np.zeros(TEST_RESET)
        
        # Collect data for 3 episode of test and log the mean reward and ep_length
        for i in range(TEST_RESET):
            stop_test = False
            test_reward = [0, 0]
            test_state, _ = test_env.reset(seed = SEED)
            ep_len = 0
            
            while not stop_test:
                # Get action with argmax
                with torch.no_grad():
                    test_state_tensor = torch.tensor(test_state)
                    action = encode_action(agent.get_action_test(test_state_tensor[0]).cpu(),
                                            agent.get_action_test(test_state_tensor[1]).cpu())
                    
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