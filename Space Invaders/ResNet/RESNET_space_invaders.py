import gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import os
import csv
import time
import torchvision.models as models

#first section is same as other games
BASE_DIR = "space_invaders_base_dir"
os.makedirs(BASE_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'space_invaders_resnet_checkpoint.pth')
BEST_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'space_invaders_resnet_best.pth')
BEST_AVG_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'space_invaders_resnet_best_avg.pth')
METRICS_PATH = os.path.join(BASE_DIR, 'space_invaders_metrics_resnet.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
MEMORY_CAPACITY = 100_000
GAMMA = 0.99
LR_BACKBONE = 0.0001
LR_HEAD = 0.0005
EPSILON_START = 1.0
EPSILON_END = 0.1
EPSILON_DECAY = 0.9995
TRAIN_START = 10000
TARGET_UPDATE_FREQ = 1000
REWARD_WINDOW_SIZE = 10

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

#envirenment setup
def make_env():
    env = gym.make('SpaceInvadersNoFrameskip-v4')
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4, #we skip 4 frames to speed up training
        screen_size=84,
        grayscale_obs=True,
        scale_obs=False
    )
    env = gym.wrappers.FrameStack(env, 4)
    return env

env = make_env()
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def preprocess_state(state):
    return (np.array(state) / 255.0).astype(np.float32)

#basic ResNet architecture
class DQNResNet(nn.Module):
    def __init__(self, num_actions):
        super(DQNResNet, self).__init__()
        self.resnet = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1) # Load pre-trained ResNet-18, initialized with ImageNet weights as a backbone
        original_conv1 = self.resnet.conv1 # Change the first convolutional layer to accept 4 input channels (for FrameStack) and keep the rest of the architecture intact, four channels are used for FrameStack
        self.resnet.conv1 = nn.Conv2d(4, original_conv1.out_channels, #output channels are the same as original conv1 (original_conv1.out_channels)
                                      kernel_size=original_conv1.kernel_size, #kernel size is conv1's kernel size
                                      stride=original_conv1.stride,
                                      padding=original_conv1.padding,
                                      bias=original_conv1.bias)
        num_ftrs = self.resnet.fc.in_features # Change the final fully connected layer to output the number of actions
        self.resnet.fc = nn.Linear(num_ftrs, num_actions) #this is the final layer that outputs the Q-values for each action
    
    def forward(self, x): # x is the input state
        return self.resnet(x)

#thid replay buffer is used to store transitions and sample them for training, different from the standard replay buffer, this one uses priorities to sample transitions
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity #maximum capacity of the buffer whcih is used to store transitions
        self.buffer = deque(maxlen=capacity) 
        self.priorities = deque(maxlen=capacity) #priorities for each transition, used to sample transitions based on their importance
        self.alpha = alpha #alpha is used to control the prioritization of transitions, higher alpha means more prioritization
        self.beta = beta #beta is used to control the importance of the sampling weights
        self.max_priority = 1.0 #initial maximum priority, used to initialize the priorities of new transitions
        
    def push(self, transition): #transition is a tuple of (state, action, reward, next_state, done)
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)
        
    def sample(self, batch_size): #samples a batch of transitions based on their priorities
        priorities = np.array(self.priorities) ** self.alpha
        probs = priorities / priorities.sum()
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta)
        weights /= weights.max()
        return samples, indices, weights
        
    def update_priorities(self, indices, priorities):
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.buffer)

# --- DQN Agent ---
class DQNAgent:
    def __init__(self, load_checkpoint=False):
        self.policy_net = DQNResNet(env.action_space.n).to(DEVICE)
        self.target_net = DQNResNet(env.action_space.n).to(DEVICE)
        
        optimizer_params = [
            {'params': self.policy_net.resnet.fc.parameters(), 'lr': LR_HEAD},
            {'params': self.policy_net.resnet.conv1.parameters(), 'lr': LR_HEAD},
            {'params': self.policy_net.resnet.layer4.parameters(), 'lr': LR_BACKBONE},
            {'params': self.policy_net.resnet.layer3.parameters(), 'lr': LR_BACKBONE},
        ]
        self.optimizer = optim.Adam(optimizer_params)
        self.memory = PrioritizedReplayBuffer(MEMORY_CAPACITY)
        
        self.epsilon = EPSILON_START
        self.start_episode = 0
        self.frame_idx = 0
        self.best_reward = -np.inf
        self.best_avg_reward = -np.inf
        self.reward_window = deque(maxlen=REWARD_WINDOW_SIZE)

        if load_checkpoint and os.path.exists(CHECKPOINT_PATH):
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.start_episode = checkpoint['episode'] + 1
            self.frame_idx = checkpoint['frame_idx']
            self.best_reward = checkpoint.get('best_reward', -np.inf)
            self.best_avg_reward = checkpoint.get('best_avg_reward', -np.inf)
            self.reward_window = deque(checkpoint.get('reward_window', [0]*REWARD_WINDOW_SIZE),
                                      maxlen=REWARD_WINDOW_SIZE)
            print(f"Loaded checkpoint from episode {self.start_episode}.")
        else:
            self.target_net.load_state_dict(self.policy_net.state_dict())
        
        self.target_net.eval()
            
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                return self.policy_net(state_tensor).argmax().item()
        return env.action_space.sample()
    
    def update_model(self):
        if len(self.memory) < TRAIN_START:
            return
        
        transitions, indices, weights = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE)

        states = torch.from_numpy(np.array(batch.state)).to(DEVICE)
        next_states = torch.from_numpy(np.array(batch.next_state)).to(DEVICE)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(batch.reward).to(DEVICE)
        dones = torch.FloatTensor(batch.done).to(DEVICE)
        
        q_values = self.policy_net(states).gather(1, actions)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1))
            expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * GAMMA * next_q_values
        
        td_errors = torch.abs(q_values - expected_q_values).squeeze().detach().cpu().numpy()
        loss = (weights * F.mse_loss(q_values.squeeze(), expected_q_values.squeeze(), reduction='none')).mean()
        
        self.memory.update_priorities(indices, td_errors + 1e-5)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, episode, path=CHECKPOINT_PATH):
        torch.save({
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'frame_idx': self.frame_idx,
            'best_reward': self.best_reward,
            'best_avg_reward': self.best_avg_reward,
            'reward_window': list(self.reward_window)
        }, path)
        print(f"[✓] Checkpoint saved at episode {episode}")


def save_metrics(episode, reward, epsilon, best_reward, best_avg_reward, file_path=METRICS_PATH):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Episode', 'Reward', 'Epsilon', 'Best_Reward', 'Best_Avg_Reward'])
        writer.writerow([episode, reward, epsilon, best_reward, best_avg_reward])


try:
    agent = DQNAgent(load_checkpoint=False)
    print("Training Space Invaders with ResNet DQN...")
    for episode in range(agent.start_episode, 1000):
        obs, _ = env.reset(seed=np.random.randint(1, 10000))
        state = preprocess_state(obs)
        episode_reward = 0
        done = False
        
        while not done:
            agent.frame_idx += 1
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state = preprocess_state(next_obs)
            agent.memory.push(Transition(state, action, reward, next_state, done))
            agent.update_model()
            state = next_state
            episode_reward += reward
            
            if agent.frame_idx % TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
            agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
        
        agent.reward_window.append(episode_reward)
        avg_reward = np.mean(agent.reward_window)
        
        if episode_reward > agent.best_reward:
            agent.best_reward = episode_reward
            agent.save_checkpoint(episode, BEST_CHECKPOINT_PATH)
        if avg_reward > agent.best_avg_reward:
            agent.best_avg_reward = avg_reward
            agent.save_checkpoint(episode, BEST_AVG_CHECKPOINT_PATH)
        
        save_metrics(episode, episode_reward, agent.epsilon, agent.best_reward, agent.best_avg_reward)
        print(f"Ep {episode} | Reward: {episode_reward:.1f} | Avg({REWARD_WINDOW_SIZE}): {avg_reward:.1f} | ε: {agent.epsilon:.4f}")
        
        if episode % 10 == 0:
            agent.save_checkpoint(episode)

except KeyboardInterrupt:
    print("\nTraining manually stopped.")
finally:
    if 'agent' in locals():
        agent.save_checkpoint(episode)
        print("Final checkpoint saved.")
    env.close()
