#sin comentario, virtualmente mismo que enterno de pong.
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
import math

BASE_DIR = "base_dir"
os.makedirs(BASE_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'space_invaders_dqn_checkpoint.pth')
BEST_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'space_invaders_dqn_best.pth')
METRICS_PATH = os.path.join(BASE_DIR, 'space_invaders_metrics_dqn.csv')

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 64
MEMORY_CAPACITY = 200_000
GAMMA = 0.99
LR = 0.00025
EPSILON_START = 1.0
EPSILON_END = 0.1  
EPSILON_DECAY = 0.9998  
TRAIN_START = 10_000
TARGET_UPDATE_FREQ = 10_000
REWARD_WINDOW_SIZE = 10
EXPLORATION_BURST_EVERY = 50  
BURST_EPSILON = 0.3  
REWARD_SCALE = 0.01

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def make_env():
    env = gym.make('SpaceInvadersNoFrameskip-v4')
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=True,
        grayscale_obs=True,
        scale_obs=False
    )
    env = gym.wrappers.FrameStack(env, 4)
    return env

env = make_env()

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))

def preprocess_state(state):
    state_array = np.array(state, dtype=np.float32) / 255.0
    return state_array

class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)  
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)  
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)  
        
        self.flattened_size = 64 * 7 * 7
        
        self.fc1 = nn.Linear(self.flattened_size, 512)
        self.fc2 = nn.Linear(512, num_actions)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
                
    def forward(self, x):
        
        x = F.relu(self.conv1(x))   
        x = F.relu(self.conv2(x))   
        x = F.relu(self.conv3(x))   
        
        x = x.view(x.size(0), -1)   
        
        x = F.relu(self.fc1(x))     
        return self.fc2(x)          

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, transition):
        self.buffer.append(transition)
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    def __init__(self, load_checkpoint=False):
        self.num_actions = env.action_space.n
        self.policy_net = DQN(self.num_actions).to(DEVICE)
        self.target_net = DQN(self.num_actions).to(DEVICE)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=LR, alpha=0.95, eps=0.01)
        self.memory = ReplayBuffer(MEMORY_CAPACITY)
        
        self.epsilon = EPSILON_START
        self.start_episode = 0
        self.frame_idx = 0
        self.episode_count = 0
        
        self.best_reward = -np.inf
        self.best_avg_reward = -np.inf
        self.reward_window = deque(maxlen=REWARD_WINDOW_SIZE)
        [self.reward_window.append(0) for _ in range(REWARD_WINDOW_SIZE)]

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
            print(f"Checkpoint loaded. Resuming from episode {self.start_episode}, frame {self.frame_idx}.")
            print(f"Best reward: {self.best_reward:.2f}, Best avg reward: {self.best_avg_reward:.2f}")
            
    def select_action(self, state):
        #cambio de epsilon para burst de exploracion
        if self.episode_count % EXPLORATION_BURST_EVERY == 0:
            current_epsilon = BURST_EPSILON
            burst_flag = True
        else:
            current_epsilon = self.epsilon
            burst_flag = False
            
        if random.random() > current_epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item(), burst_flag
        else:
            return env.action_space.sample(), burst_flag
    
    def update_model(self):
        if len(self.memory) < TRAIN_START:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(DEVICE)

        current_q = self.policy_net(states).gather(1, actions).squeeze()

        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze()
            target_q = rewards + (1 - dones) * GAMMA * next_q

  
        reward_min = rewards.min().item()
        reward_max = rewards.max().item()
        reward_range = reward_max - reward_min + 1e-5 
        reward_factor = (rewards - reward_min) / reward_range

        base_loss = F.mse_loss(current_q, target_q, reduction='none')

        scaled_loss = base_loss * reward_factor
        loss = scaled_loss.mean()

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)

        self.optimizer.step()


    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def save_checkpoint(self, episode, path=CHECKPOINT_PATH):
        checkpoint = {
            'policy_net': self.policy_net.state_dict(),
            'target_net': self.target_net.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'episode': episode,
            'frame_idx': self.frame_idx,
            'best_reward': self.best_reward,
            'best_avg_reward': self.best_avg_reward,
            'reward_window': list(self.reward_window)
        }
        torch.save(checkpoint, path)
        print(f"[✓] Checkpoint saved at episode {episode} to {path}")

def save_metrics(episode, reward, epsilon, best_reward, best_avg_reward, file_path=METRICS_PATH):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(['Episode', 'Reward', 'Epsilon', 'Best_Reward', 'Best_Avg_Reward'])
        writer.writerow([episode, reward, epsilon, best_reward, best_avg_reward])

if __name__ == "__main__":
    
    try:
        agent = DQNAgent(load_checkpoint=True)
        
        if not os.path.exists(METRICS_PATH) or os.stat(METRICS_PATH).st_size == 0:
            with open(METRICS_PATH, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Episode', 'Reward', 'Epsilon', 'Best_Reward', 'Best_Avg_Reward'])
        
        print("Starting training with enhanced exploration...")
        print(f"Device: {DEVICE}")
        print(f"Number of actions: {env.action_space.n}")
        print(f"Exploration bursts every {EXPLORATION_BURST_EVERY} episodes")
        
        for episode in range(agent.start_episode, 1000):  
            agent.episode_count = episode
            seed = SEED + episode
            obs, _ = env.reset(seed=seed)
            state = preprocess_state(obs)
            episode_reward = 0
            done = False
            episode_start_time = time.time()
            burst_active = False
            
            while not done:
                agent.frame_idx += 1
                action, is_burst = agent.select_action(state)
                if is_burst:
                    burst_active = True
                
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                
                next_state = preprocess_state(next_obs)
                scaled_reward = reward * REWARD_SCALE
                
                agent.memory.push(Transition(state, action, scaled_reward, next_state, done))
                
                agent.update_model()
                
                state = next_state
                episode_reward += reward
                
                if agent.frame_idx % TARGET_UPDATE_FREQ == 0:
                    agent.update_target_net()
                    print(f"Target network updated at frame {agent.frame_idx}")
                
                if not burst_active:
                    agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)
                
                if done:
                    agent.reward_window.append(episode_reward)
                    current_avg_reward = np.mean(agent.reward_window)
                    
                    save_best = False
                    if episode_reward > agent.best_reward:
                        agent.best_reward = episode_reward
                        save_best = True
                    
                    save_best_avg = False
                    if current_avg_reward > agent.best_avg_reward:
                        agent.best_avg_reward = current_avg_reward
                        save_best_avg = True
                    
                    episode_duration = time.time() - episode_start_time
                    
                    burst_status = " [EXPLORATION BURST]" if burst_active else ""
                    
                    print(f"Ep {episode}: Reward {episode_reward:.1f} | "
                        f"Best {agent.best_reward:.1f} | "
                        f"Avg({REWARD_WINDOW_SIZE}) {current_avg_reward:.1f} | "
                        f"Best Avg {agent.best_avg_reward:.1f} | "
                        f"ε {agent.epsilon:.4f}{burst_status} | "
                        f"Frames {agent.frame_idx} | "
                        f"Time {episode_duration:.1f}s")
                    
                    save_metrics(episode, episode_reward, agent.epsilon, 
                                agent.best_reward, agent.best_avg_reward)
                    
                    if episode % 10 == 0:
                        agent.save_checkpoint(episode, CHECKPOINT_PATH)
                    
                    if save_best:
                        agent.save_checkpoint(episode, BEST_CHECKPOINT_PATH)
                        print(f"-----------> New BEST individual model saved with reward: {agent.best_reward:.2f}")
                    
                    if save_best_avg:
                        agent.save_checkpoint(episode, BEST_CHECKPOINT_PATH)
                        print(f"-----------> New BEST AVERAGE model saved with avg reward: {agent.best_avg_reward:.2f}")
                    
                    break
    except KeyboardInterrupt:
        print("\nTraining stopped manually.")

    finally:
        if 'agent' in locals():
            final_episode = episode if 'episode' in locals() else 0
            agent.save_checkpoint(final_episode, CHECKPOINT_PATH)
            print("Final checkpoint saved.")
        env.close()