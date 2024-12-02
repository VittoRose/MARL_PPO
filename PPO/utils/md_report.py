from PPO.parameters import *
from PPO.ActorCritic import neurons, activation_fn, hidden_layer
from time import time

def create_md_summary(gym_id: str, name: str, folder: str, seed: float, device: str)-> None:
    """
    Function that create a MarkDown report for parameters used during training
    """
    report = folder + name + ".md"
    
    with open(report, 'w') as file:
        file.write("# Environment: " + gym_id + "\n\n")
        
        file.write("Executed on " + device + "\n")
        if int(seed) == SEED:
            file.write(f"Seed: {seed}, (deterministic)\n")
        else:
            file.write(f"Seed: {seed}, (random)\n")

        file.write("\n## Training parameters\n\n")

        file.write(f"- Total epoch: {MAX_EPOCH}\n")
        file.write(f"- Number of environments: {n_env}\n")
        file.write(f"- Timestep for collecting data T = {n_step}\n")
        file.write(f"- Epoch for test: {TEST_INTERVAL} with {TEST_RESET} tests each time\n")
        file.write(f"- Total data for each loop: {BATCH_SIZE}\n")
        file.write(f"- Update epoch K = {K_EPOCHS}\n")
        file.write(f"- Mini-batch size {MINI_BATCH_SIZE}\n\n")

        file.write("## Hyperparameters\n\n")
        file.write(f"- Discount factor: {GAMMA}\n")
        file.write(f"- GAE lambda: {GAE_LAMBDA}\n")
        file.write(f"- Learning rate: {LR}\n")
        file.write(f"- Clipping factor: {CLIP}\n")
        file.write(f"- Loss: c1 = {VALUE_COEFF}; c2 = {ENTROPY_COEF}\n")

        file.write(f"\nClipping loss function: {VALUE_CLIP}\n\n")
        file.write(f"Value normalization: {VALUE_NORM}\n\n")

        file.write(f"## Network\n\n")
        file.write(f"- Number of neurons for hidden layer: {neurons}\n")
        file.write(f"- Activation function: {activation_fn}\n")
        file.write(f"- Number of hidden layer: {hidden_layer}\n")

def complete_md_summary(path: str, starting_time: float) -> None:
    """
    Complete the run summary with execution time
    """
    end_time = time()
    min = (end_time - starting_time)/60
    sec = (end_time - starting_time)%60
    with open(path, 'a') as file:
        file.write(f"\nTotal time: {min:.0f} min {sec:.0f} sec\n")
