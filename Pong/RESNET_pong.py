#Usa arquitectura Resnet18 pre-entrenada para jugar al juego Pong.
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

#Directorio base y para guardar checkpoints y métricas
BASE_DIR = "pong_base_dir"
os.makedirs(BASE_DIR, exist_ok=True)
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pong_resnet_checkpoint.pth')
BEST_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pong_resnet_best.pth')
BEST_AVG_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pong_resnet_best_avg.pth')
METRICS_PATH = os.path.join(BASE_DIR, 'pong_metrics_resnet.csv')

#Cuda (nvidia gpu) si está disponible, sino CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#Batch size, memoria, gamma, tasas de aprendizaje y epsilon
BATCH_SIZE = 32 
MEMORY_CAPACITY = 100_000
GAMMA = 0.99
LR_BACKBONE = 0.0001
LR_HEAD = 0.0005
EPSILON_START = 1.0 #Diferentes epsilons para ir decayendo 
EPSILON_END = 0.01  
EPSILON_DECAY = 0.9999  # Decay mas lento para Pong
TRAIN_START = 10000 #Número de experiencias para empezar a entrenar
TARGET_UPDATE_FREQ = 1000 # Frecuencia de actualizacion del target network
REWARD_WINDOW_SIZE = 10 # Tamaño de la ventana para calcular la recompensa promedio

#Seed usada para el enorno de reproducibilidad
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Crear el entorno de Pong con preprocesamiento
def make_env():
    env = gym.make('PongNoFrameskip-v4') #Agregar "render_mode='human'" si se quiere ver el entrenamiento, real time
    env = gym.wrappers.AtariPreprocessing(
        env,
        noop_max=30, 
        frame_skip=4, # Salto de 4 frames
        screen_size=84, 
        grayscale_obs=True,
        scale_obs=False 
    )
    env = gym.wrappers.FrameStack(env, 4)
    return env


env = make_env()
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) # Transicion de experiencia

def preprocess_state(state): #
    return (np.array(state) / 255.0).astype(np.float32) # Normalizar a 0-1, ResNet espera imágenes en rango 0-1

# --- ResNet DQN Architecture ---
class DQNResNet(nn.Module):
    def __init__(self, num_actions):
        super(DQNResNet, self).__init__()
        self.resnet = models.resnet18(weights=None) #importar ResNet18 sin pesos preentrenados para adaptarlo a Pong
        original_conv1 = self.resnet.conv1 # Guardar la capa convolucional original
        self.resnet.conv1 = nn.Conv2d(4, original_conv1.out_channels,
                                      kernel_size=original_conv1.kernel_size,
                                      stride=original_conv1.stride,
                                      padding=original_conv1.padding,
                                      bias=original_conv1.bias)
        num_ftrs = self.resnet.fc.in_features # Guardar el número de características de la capa final
        self.resnet.fc = nn.Linear(num_ftrs, num_actions) # Cambiar la capa final para que coincida con el número de acciones del entorno Pong
    
    def forward(self, x):
        return self.resnet(x)

# Reply Buffer con Prioridades, es decir que las experiencias se almacenan con una prioridad que afecta su probabilidad de ser muestreadas
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6, beta=0.4):
        self.capacity = capacity
        self.buffer = deque(maxlen=capacity)
        self.priorities = deque(maxlen=capacity)
        self.alpha = alpha
        self.beta = beta
        self.max_priority = 1.0
        
    def push(self, transition): #Push para agregar una transicion al buffer
        self.buffer.append(transition)
        self.priorities.append(self.max_priority)
        
    def sample(self, batch_size): #Sample para muestrear experiencias del buffer
        priorities = np.array(self.priorities) ** self.alpha #prioridades elevadas a alpha para dar más peso a las experiencias con alta prioridad
        probs = priorities / priorities.sum() 
        indices = np.random.choice(len(self.buffer), batch_size, p=probs) # Muestreo de índices basado en prioridades
        samples = [self.buffer[i] for i in indices]
        weights = (len(self.buffer) * probs[indices]) ** (-self.beta) # Calcular pesos inversamente proporcionales a las prioridades
        weights /= weights.max()
        return samples, indices, weights #regresa las muestras, los indices y los pesos
        
    def update_priorities(self, indices, priorities): #actualizar prioridades de las experiencias muestreadas
        for idx, priority in zip(indices, priorities):
            self.priorities[idx] = priority
            self.max_priority = max(self.max_priority, priority)
            
    def __len__(self):
        return len(self.buffer)

# DQN Agent con ResNet Backbone
class DQNAgent:
    def __init__(self, load_checkpoint=False):
        self.num_actions = env.action_space.n # Número de acciones en el entorno
        self.policy_net = DQNResNet(self.num_actions).to(DEVICE) #polict net es para tomar decisiones
        self.target_net = DQNResNet(self.num_actions).to(DEVICE) #target net es para calcular el valor de las acciones
        
        # Diferentes tasas de aprendizaje para la cabeza y el backbone
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

        if load_checkpoint and os.path.exists(CHECKPOINT_PATH): # Cargar desde un checkpoint si existe, true si se quiere cargar un checkpoint
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
            
    # Metodo para seleccionar una accion basado en la politica epsilon-greedy
    def select_action(self, state):
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                return self.policy_net(state_tensor).argmax().item()
        return env.action_space.sample()
    
    # Metodo para actualizar el modelo
    def update_model(self):
        if len(self.memory) < TRAIN_START: #si no hay suficientes experiencias, no entrenar
            return

        transitions, indices, weights = self.memory.sample(BATCH_SIZE) # Muestreo de experiencias del buffer
        batch = Transition(*zip(*transitions)) 
        weights = torch.tensor(weights, dtype=torch.float32).to(DEVICE) # Convertir pesos a tensor y mover a DEVICE, ei. GPU o CPU
        states = torch.from_numpy(np.array(batch.state)).to(DEVICE)
        next_states = torch.from_numpy(np.array(batch.next_state)).to(DEVICE)
        actions = torch.LongTensor(batch.action).unsqueeze(1).to(DEVICE)
        rewards = torch.FloatTensor(batch.reward).to(DEVICE)
        dones = torch.FloatTensor(batch.done).to(DEVICE)

        q_values = self.policy_net(states).gather(1, actions) # Q-values de la politica actual
        with torch.no_grad(): #
            next_actions = self.policy_net(next_states).argmax(1) # Acciones de la politica actual para los siguientes estados
            next_q_values = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)) # Q-values de la target network para los siguientes estados
            expected_q_values = rewards.unsqueeze(1) + (1 - dones.unsqueeze(1)) * GAMMA * next_q_values # Calcular los valores esperados de Q usados formula de Bellman
        
        td_errors = torch.abs(q_values - expected_q_values).squeeze().detach().cpu().numpy() #td_errors es el error temporal de la diferencia entre los Q-values actuales y los esperados
        loss = (weights * F.mse_loss(q_values.squeeze(), expected_q_values.squeeze(), reduction='none')).mean() # Calcular la perdida usando MSE y ponderar por los pesos
        
        self.memory.update_priorities(indices, td_errors + 1e-5)
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters(): # Clipping de gradientes esto es para evitar explosiones de gradientes, clipping los gradientes entre -1 y 1
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
    
    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def save_checkpoint(self, episode, path=CHECKPOINT_PATH): # Guardar un checkpoint
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

# Oara guardar las metricas de entrenamiento
def save_metrics(episode, reward, epsilon, best_reward, best_avg_reward, file_path=METRICS_PATH):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['Episode', 'Reward', 'Epsilon', 'Best_Reward', 'Best_Avg_Reward'])
        writer.writerow([episode, reward, epsilon, best_reward, best_avg_reward])


try:
    agent = DQNAgent(load_checkpoint=False)
    print("Training Pong with ResNet DQN :)")
    print(f"Action space: {env.action_space.n} actions")
    
    for episode in range(agent.start_episode, 1000):  #Configurar el numero de episodios a entrenar, usamos 1000 episodios
        obs, _ = env.reset()
        state = preprocess_state(obs)
        episode_reward = 0
        done = False
        lives = 5  # Inicializar vidas, Pong tiene 5 vidas al inicio
        
        while not done:
            agent.frame_idx += 1
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            #Clipping de recompensa, Pong tiene recompensas de -1, 0 o 1, esto es para evitar que las recompensas negativas afecten el entrenamiento
            clipped_reward = np.sign(reward)
            
            # Actualizar vidas, Pong tiene un sistema de vidas que se pierde al recibir un punto
            if 'lives' in info:
                if info['lives'] < lives:
                    lives = info['lives']
            
            #
            next_state = preprocess_state(next_obs)
            agent.memory.push(Transition(state, action, clipped_reward, next_state, done))
            agent.update_model()
            state = next_state
            episode_reward += clipped_reward
            
            # periodicamente actualizar la red objetivo
            if agent.frame_idx % TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
            
            # Actualizar epsilon
            agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

        #Al final del episodio, actualizar la ventana de recompensas y calcular la recompensa promedio
        agent.reward_window.append(episode_reward)
        avg_reward = np.mean(agent.reward_window)
        
        # Guardar el mejor checkpoint si se supera la mejor recompensa o la mejor recompensa promedio
        if episode_reward > agent.best_reward:
            agent.best_reward = episode_reward
            agent.save_checkpoint(episode, BEST_CHECKPOINT_PATH)
            print(f" New BEST reward: {episode_reward:.1f} ")
            
        if avg_reward > agent.best_avg_reward:
            agent.best_avg_reward = avg_reward
            agent.save_checkpoint(episode, BEST_AVG_CHECKPOINT_PATH)
            print(f" New BEST average reward: {avg_reward:.1f} ")

        # Guardar las métricas del episodio
        save_metrics(episode, episode_reward, agent.epsilon, agent.best_reward, agent.best_avg_reward)
        
        # Imprimir estadísticas del episodio, mientras se entrena
        print(f"Ep {episode} | "
              f"Reward: {episode_reward:.1f} | "
              f"Avg({REWARD_WINDOW_SIZE}): {avg_reward:.1f} | "
              f"ε: {agent.epsilon:.4f} | "
              f"Frames: {agent.frame_idx}")
        
        # Guardar checkpoint cada 10 episodios
        if episode % 10 == 0:
            agent.save_checkpoint(episode)

except KeyboardInterrupt:
    print("\nTraining manually stopped.")
finally:
    if 'agent' in locals():
        agent.save_checkpoint(episode)
        print("Final checkpoint saved.")
    env.close()