#Comentarios solo en modelos Pong, todos los modelos de DQN - Resnet son iguales, solo se cambio unos valores para mejorar el rendimiento del juego especifico
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

# Directorios para guardar los checkpoints y métricas
BASE_DIR = "pong_base_dir"
os.makedirs(BASE_DIR, exist_ok=True) 
CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pong_DQN_checkpoint.pth')
BEST_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pong_DQN_best.pth')
BEST_AVG_CHECKPOINT_PATH = os.path.join(BASE_DIR, 'pong_DQN_best_avg.pth') 
METRICS_PATH = os.path.join(BASE_DIR, 'pong_DQN_metrics.csv')

# Cuda (nvidia GPU), de lo contrario CPU
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 128  # El tamaño del batch puede cambiar dependiendo del juego
MEMORY_CAPACITY = 100_000  # Tamaño del buffer de replay
GAMMA = 0.99
LR = 0.0001  # Tasa de aprendizaje más baja para mayor estabilidad
EPSILON_START = 1.0
EPSILON_END = 0.01  # Epsilon final más bajo
EPSILON_DECAY = 0.995  # Decaimiento más rápido
EPSILON_DECAY_START_FRAME = 10_000  # Comenzar a decaer después de algunos frames
TARGET_UPDATE_FREQ = 1000
REWARD_WINDOW_SIZE = 20 
REWARD_SCALE = 1.0  # No escalar las recompensas de Pong
MIN_REPLAY_SIZE = 5000  # Tamaño mínimo del buffer antes de entrenar

# Semilla para reproducibilidad, entorno gym y torch
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# Crear el entorno de Pong
def make_env():
    env = gym.make('PongNoFrameskip-v4') #Agregar parametro "render_mode= 'human' " para ver el juego
    env = gym.wrappers.AtariPreprocessing(
        env, 
        noop_max=30,
        frame_skip=4,
        screen_size=84,
        terminal_on_life_loss=False,
        grayscale_obs=True,
        scale_obs=False
    )
    env = gym.wrappers.FrameStack(env, 4)
    return env

env = make_env()
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done')) # Transición de experiencia

def preprocess_state(state):
    return (np.array(state) / 255.0).astype(np.float32) # Normalizar el estado a [0, 1] para mejorar la estabilidad del entrenamiento

# Modelo DQN
class DQN(nn.Module):
    def __init__(self, num_actions):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4) # 4 canales de entrada (stack de frames)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2) # 32 canales de entrada
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1) # 64 canales de entrada
        self.fc1 = nn.Linear(64 * 7 * 7, 512) # 64 canales de salida, tamaño de imagen reducido a 7x7, 512 neuronas en la capa oculta
        self.fc2 = nn.Linear(512, num_actions) # Capa de salida con número de acciones posibles

        # Para inicializacion de pesos y bias
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x): #Foward pass del modelo
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1) # Aplanar la salida de las capas convolucionales
        x = F.relu(self.fc1(x))
        return self.fc2(x)

# Replay bufffer mas simple que el de Resnet, no se usa Prioritized Experience Replay
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)

    def __len__(self):
        return len(self.buffer)

# Agente DQN, que maneja la selección de acciones, el entrenamiento del modelo y el almacenamiento de experiencias
class DQNAgent:
    def __init__(self, load_checkpoint=False): 
        self.num_actions = env.action_space.n #num de acciones del entorno Pong
        self.policy_net = DQN(self.num_actions).to(DEVICE) #policy net es para seleccionar acciones
        self.target_net = DQN(self.num_actions).to(DEVICE) #target net es para calcular el valor objetivo de las acciones
        self.target_net.load_state_dict(self.policy_net.state_dict()) # Inicializar target net con los mismos pesos que policy net
        self.target_net.eval()
        
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LR)  # Use Adam optimizer
        self.memory = ReplayBuffer(MEMORY_CAPACITY)

        self.epsilon = EPSILON_START
        self.start_episode = 0
        self.frame_idx = 0
        self.best_reward = -np.inf
        self.best_avg_reward = -np.inf
        self.reward_window = deque([0] * REWARD_WINDOW_SIZE, maxlen=REWARD_WINDOW_SIZE)

        if load_checkpoint and os.path.exists(CHECKPOINT_PATH): # Cargar el checkpoint si existe
            checkpoint = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
            self.policy_net.load_state_dict(checkpoint['policy_net'])
            self.target_net.load_state_dict(checkpoint['target_net'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.epsilon = checkpoint['epsilon']
            self.start_episode = checkpoint['episode'] + 1
            self.frame_idx = checkpoint['frame_idx']
            self.best_reward = checkpoint.get('best_reward', -np.inf)
            self.best_avg_reward = checkpoint.get('best_avg_reward', -np.inf)
            self.reward_window = deque(checkpoint.get('reward_window', [0]*REWARD_WINDOW_SIZE), maxlen=REWARD_WINDOW_SIZE)
            print(f"Checkpoint loaded. Resuming from episode {self.start_episode}.")

    def select_action(self, state): #seleccionar una acción usando la política epsilon-greedy
        if random.random() > self.epsilon:
            with torch.no_grad():
                state_tensor = torch.from_numpy(state).unsqueeze(0).to(DEVICE)
                q_values = self.policy_net(state_tensor)
                return q_values.argmax().item()
        return env.action_space.sample()

    def update_model(self): # Actualizar el modelo DQN usando el buffer de replay
        if len(self.memory) < MIN_REPLAY_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE) #obtener una muestra aleatoria del buffer, esto es para romper la correlación entre las experiencias
        batch = Transition(*zip(*transitions)) 

        states = torch.tensor(np.array(batch.state), dtype=torch.float32).to(DEVICE)
        next_states = torch.tensor(np.array(batch.next_state), dtype=torch.float32).to(DEVICE)
        actions = torch.tensor(batch.action, dtype=torch.int64).unsqueeze(1).to(DEVICE)
        rewards = torch.tensor(batch.reward, dtype=torch.float32).to(DEVICE)
        dones = torch.tensor(batch.done, dtype=torch.float32).to(DEVICE)

        # Calcular Q-values actuales y objetivos
        current_q = self.policy_net(states).gather(1, actions).squeeze()
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + (1 - dones) * GAMMA * next_q

        # calcular perdida
        loss = F.mse_loss(current_q, target_q)
        
        # Optimizar
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)  # Gradient clipping
        self.optimizer.step()

    # Actualizar la red objetivo
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
        print(f"[✓] Checkpoint saved at episode {episode}")

# Ir guardando las metricas de entrenamiento en un archivo CSV
def save_metrics(episode, reward, epsilon, best_reward, best_avg_reward, file_path=METRICS_PATH):
    file_exists = os.path.isfile(file_path)
    with open(file_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists or os.stat(file_path).st_size == 0:
            writer.writerow(['Episode', 'Reward', 'Epsilon', 'Best_Reward', 'Best_Avg_Reward'])
        writer.writerow([episode, reward, epsilon, best_reward, best_avg_reward])

# Loop de entrenamiento
try:
    agent = DQNAgent(load_checkpoint=False)
    print("Training Pong with DQN...")
    print(f"Action space: {env.action_space.n} actions")
    
    for episode in range(agent.start_episode, 1000):  # num de episodios a entrenar, 1000 es lo que se utiliza en el paper
        obs, _ = env.reset()
        state = preprocess_state(obs)
        episode_reward = 0
        done = False
        step_count = 0 
        
        while not done: #
            agent.frame_idx += 1 #este contador es para el decaimiento de epsilon y actualizacion de la red objetivo
            step_count += 1 #step_count es para entrenar el modelo cada 4 pasos0
            
            # actualizar epsilon
            if agent.frame_idx > EPSILON_DECAY_START_FRAME:
                agent.epsilon = max(EPSILON_END, agent.epsilon * EPSILON_DECAY)

            # seleccionar accion
            action = agent.select_action(state)
            next_obs, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            # Clip Pong rewards (-1, 0, 1)
            clipped_reward = np.sign(reward)
            
            next_state = preprocess_state(next_obs)
            agent.memory.push(Transition(state, action, clipped_reward, next_state, done))
            state = next_state
            episode_reward += clipped_reward
            
            # entrenamiento del modelo mas frecuente
            if step_count % 4 == 0:
                agent.update_model()
            
            # actualizar la red objetivo cada TARGET_UPDATE_FREQ frames
            if agent.frame_idx % TARGET_UPDATE_FREQ == 0:
                agent.update_target_net()
                print(f"Target network updated at frame {agent.frame_idx}")

        # actualizar la ventana de recompensas
        agent.reward_window.append(episode_reward)
        avg_reward = np.mean(agent.reward_window)
        # Guardar las métricas
        save_metrics(episode, episode_reward, agent.epsilon, agent.best_reward, agent.best_avg_reward)

        # Actualizar mejores recompensas
        if episode_reward > agent.best_reward:
            agent.best_reward = episode_reward
            agent.save_checkpoint(episode, BEST_CHECKPOINT_PATH)
            print(f"*** New BEST reward: {episode_reward:.1f} ***")
            
        # Actualizar mejor recompensa promedio  
        if avg_reward > agent.best_avg_reward:
            agent.best_avg_reward = avg_reward
            agent.save_checkpoint(episode, BEST_AVG_CHECKPOINT_PATH)
            print(f"*** New BEST average reward: {avg_reward:.1f} ***")
        
        # Imprimir estadísticas del episodio, esto es para ver el progreso del entrenamiento
        print(f"Ep {episode} | "
              f"Reward: {episode_reward:.1f} | "
              f"Best: {agent.best_reward:.1f} | "
              f"Avg({REWARD_WINDOW_SIZE}): {avg_reward:.1f} | "
              f"ε: {agent.epsilon:.4f} | "
              f"Frames: {agent.frame_idx}")
        
        # Guardar checkpoint cada 10 episodios, puede cambiar con el juego
        if episode % 10 == 0:
            agent.save_checkpoint(episode)

except KeyboardInterrupt: #para detener el entrenamiento manualmente
    print("\nTraining manually stopped.")
finally:
    if 'agent' in locals():
        agent.save_checkpoint(episode)
        print("Final checkpoint saved.")
    env.close()