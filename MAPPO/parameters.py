from torch.nn import Tanh

# Network parameters
N_LAYER = 2                     # Number of hidden layer
N_NEURONS = 64                  # Number of neurons for each layer
ACT_FN = Tanh()                 # Activation function


# Training parameters
N_ENV = 4
N_STEP = 128                    # Number of step in the environment between each update
BATCH_SIZE = N_ENV*N_STEP*2     # Data collected for each update
MAX_EPOCH = 8_000               # Total epoch for training

# Hyperparameters
LR = 2.5e-4                     # Optimizer learning rate
GAMMA = 0.99                    # Discount factor
GAE_LAMBDA = 0.99               # TD(lambda) factor: 1 -> Monte Carlo; 0 -> TD error
K_EPOCHS = 4                    # Number of update at the end data collection

CLIP = 0.2                      # Clipping factor in policy loss
ENTROPY_COEF = 0.01             # Entropy coefficient for loss calculation
VALUE_COEFF = 0.5               # Value coefficient for loss calculation

VALUE_CLIP = False
VALUE_NORM = False

if BATCH_SIZE % K_EPOCHS != 0:
    raise ValueError("Batch size and K_epochs are not compatible")

MINI_BATCH_SIZE = BATCH_SIZE // K_EPOCHS      # Be careful here

SEED = 0

# Test parameters
TEST_INTERVAL = 120
TEST_RESET = 3
