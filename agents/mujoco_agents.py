import itertools
import torch
import random
from torch import nn
from torch import optim
import numpy as np
from tqdm import tqdm
import pdb
from torch.distributions.multivariate_normal import MultivariateNormal
import matplotlib.pyplot as plt
import os

from utils.replay_buffer import ReplayBuffer
import utils.utils as utils
from agents.base_agent import BaseAgent
import utils.pytorch_util as ptu
from policies.experts import load_expert_policy

#device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')

class ImitationAgent(BaseAgent):
    '''
    Please implement an Imitation Learning agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: 1) You may explore the files in utils to see what helper functions are available for you.
          2) You can add extra functions or modify existing functions. Dont modify the function signature of __init__ and train_iteration.  
          3) The hyperparameters dictionary contains all the parameters you have set for your agent. You can find the details of parameters in config.py.  
    
    Usage of Expert policy:
        Use self.expert_policy.get_action(observation:torch.Tensor) to get expert action for any given observation. 
        Expert policy expects a CPU tensors. If your input observations are in GPU, then 
        You can explore policies/experts.py to see how this function is implemented.
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        self.replay_buffer = ReplayBuffer(100000) #you can set the max size of replay buffer if you want    
        #initialize your model and optimizer and other variables you may need
        self.model = nn.Sequential(
            nn.Linear(observation_dim, hyperparameters['hidden_dim']),
            nn.ReLU(),
            nn.Linear(hyperparameters['hidden_dim'], hyperparameters['hidden_dim']),
            nn.ReLU(),
            nn.Linear(hyperparameters['hidden_dim'], action_dim),
            nn.Tanh(),
        ).to(device)       
        self.optimizer = optim.Adam(self.model.parameters(), lr = hyperparameters['lr'])
        self.loss = nn.MSELoss() 

    def forward(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        action = self.model(observation) #change this to your action
        return action


    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        
        action = self.model(observation).detach() #change this to your action
        return action 

    
    
    def update(self, observations, actions):
        #*********YOUR CODE HERE******************

        pass
    


    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        if not hasattr(self, "expert_policy"):
            self.expert_policy, initial_expert_data = load_expert_policy(env, self.args.env_name)
            self.replay_buffer.add_rollouts(initial_expert_data)
        max_ep_len = env.spec.max_episode_steps
        if(itr_num != 0):
            p = pow(self.hyperparameters['p'],itr_num)
        else:
            p = 1
        action = np.random.choice([0,1],p = [p,1-p])
        m = self.hyperparameters['m']
        if(action == 1):
            trajs = utils.sample_n_trajectories(env, self.get_action,m,max_ep_len)
            num = sum([len(traj['reward']) for traj in trajs])
            self.replay_buffer.add_rollouts(trajs)
        else:
            trajs = utils.sample_n_trajectories(env, self.expert_policy.forward,m,max_ep_len)
            num = sum([len(traj['reward']) for traj in trajs])
            self.replay_buffer.add_rollouts(trajs)
        obs = self.replay_buffer.obs
        acs = self.replay_buffer.acs
        b_size = 64
        total_size = len(self.replay_buffer.obs)
        running_loss = 0.0
        for epoch in range(5):  # loop over the dataset multiple times
            running_loss = 0.0
            for i in range(0,total_size,b_size):
                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = obs[i:min(total_size,i+b_size)],acs[i:min(total_size,i+b_size)]
                inputs = inputs
                labels = labels
                # zero the parameter gradients
                self.optimizer.zero_grad()

                # forward + backward + optimize
                outputs = self.forward(torch.Tensor(inputs).to(device))
                loss = self.loss(outputs,torch.Tensor(labels).to(device))
                loss.backward()
                self.optimizer.step()

                # print statistics
                running_loss += loss.item()   
        
        #*********YOUR CODE HERE******************

        return {'episode_loss': running_loss/total_size , 'trajectories': self.replay_buffer.paths, 'current_train_envsteps':num} #you can return more metadata if you want to



class actor(nn.Module):
    def __init__(self, observation_dim: int, action_dim: int, hidden_layers : int, hidden_dim : int):
        super(actor, self).__init__()
        
        self.seq = ptu.build_mlp(observation_dim, hidden_dim, hidden_layers - 1, hidden_dim, 'relu', 'relu')
        self.mean = nn.Linear(hidden_dim, action_dim) 
        self.var = nn.Linear(hidden_dim, action_dim)   

        nn.init.kaiming_uniform_(self.mean.weight)
        nn.init.kaiming_uniform_(self.var.weight)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        x = self.seq(observation)
        mean = torch.tanh(self.mean(x))
        var = nn.functional.softplus(self.var(x))
        
        return mean, var



class RLAgent(BaseAgent):

    '''
    Please implement an policy gradient agent. Read scripts/train_agent.py to see how the class is used. 
    
    
    Note: Please read the note (1), (2), (3) in ImitationAgent class. 
    '''

    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super().__init__()
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        #initialize your model and optimizer and other variables you may need
        self.num_itr = 0
        self.actor = actor(observation_dim, action_dim, hyperparameters['n_layers_actor'], hyperparameters['hidden_dim']).to(device)
        self.critic = ptu.build_mlp(observation_dim, 1, hyperparameters['n_layers_critic'], hyperparameters['hidden_dim'], activation = 'relu').to(device)
        self.optimizer_actor = optim.RMSprop(self.actor.parameters(), lr = hyperparameters['lr_actor'], weight_decay = 1e-3)
        self.optimizer_critic = optim.RMSprop(self.critic.parameters(), lr = hyperparameters['lr_critic'], weight_decay = 1e-3)
        self.scheduler_actor = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_actor, mode = 'min', patience = 4, factor = 0.5,  min_lr = 1e-7)
        self.scheduler_critic = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_critic, mode = 'min', patience = 6, factor = 0.5, min_lr = 1e-7)
        self.control_loss = nn.MSELoss()
        self.critic_loss = nn.SmoothL1Loss(reduction = 'sum')
        self.max_reward = -1000
        self.running_actor_loss = 0.0
        self.running_critic_loss = 0.0
        self.plot_actor_loss = []
        self.plot_critic_loss = []

    def forward(self, observation: torch.FloatTensor, action: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        mean, var = self.actor(observation.to(device))
        var = torch.diag_embed(var)
        batch_normal = MultivariateNormal(mean, var)
        log_probs = batch_normal.log_prob(action)
        return log_probs.unsqueeze(1).to(device)

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #*********YOUR CODE HERE******************
        mean, var = self.actor(observation.to(device))
        return mean.detach().to(device)
    
    @torch.no_grad()
    def get_action_train(self, observation: torch.FloatTensor):
        mean, var = self.actor(observation.to(device))
        #var = torch.clamp(var, 1e-6, 4.)
        var = torch.diag_embed(var)
        normal = MultivariateNormal(mean, var)
        action = normal.sample()
        action = torch.clamp(action, min = -1.0, max = 1.0)
        return action.detach().to(device)

    def policy_update(self, observations, actions, advantage):
        loss = 0.0
        total_steps = 0
        for i in range(len(observations)):
            log_probs = self.forward(torch.Tensor(observations[i]).to(device), torch.Tensor(actions[i]).to(device))
            if self.hyperparameters['critic']:
                ad = torch.Tensor(advantage[i]).to(device)
            else:
                ad = torch.Tensor([advantage[i]]).to(device)
            ac = torch.Tensor(actions[i]).to(device)
            loss += torch.sum(ad * -log_probs)
            loss += self.hyperparameters['entropy_weight'] * torch.sum(-log_probs)
            if loss.isnan():
                print(log_probs[log_probs.isnan()], sep = '\n')
            total_steps += observations[i].shape[0]
        return loss / total_steps

    def critic_update(self, observations, q_values):
        critic_loss = 0.0
        total_steps = 0
        for i in range(len(observations)):
            pred_values = self.critic(torch.Tensor(observations[i]).to(device))
            critic_loss += self.critic_loss(pred_values, torch.Tensor(q_values[i].reshape(-1, 1)).to(device))
            total_steps += observations[i].shape[0]
        critic_loss /= total_steps
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 2., norm_type = 2)
        self.optimizer_critic.step()
        return critic_loss
        
    def update(self, observations, actions, advantage, rewards = None, next_observations = None, q_values = None):
        #*********YOUR CODE HERE******************
        if self.hyperparameters['critic']:
            gamma = self.hyperparameters['discount']
            advantage_critic = []
            targets = []
            for i in range(len(observations)):
                critic_loss = self.critic_update(observations, q_values)
                pred_values_curr = self.critic(torch.Tensor(observations[i]).to(device)).detach()
                pred_values_next = self.critic(torch.Tensor(next_observations[i]).to(device)).detach()
                if pred_values_next.size(0) < 1000:
                    pred_values_next[-1] = 0
                advantage_critic.append((torch.Tensor(rewards[i].reshape(-1, 1)).to(device) + gamma * pred_values_next - pred_values_curr).detach())
            actor_loss = self.policy_update(observations, actions, advantage_critic)
        else:
            loss = self.policy_update(observations, actions, advantage)
        if (self.num_itr + 1) % self.hyperparameters['log'] == 0:
            self.plot_actor_loss.append(actor_loss.item())
            self.plot_critic_loss.append(critic_loss.item())
        return actor_loss, critic_loss   
    
    def get_q_values(self, rewards, next_observations):
        q_values = []
        gamma = self.hyperparameters['discount']
        m = len(rewards)
        for i in range(m):
            reward_traj = rewards[i]
            x = 0
            if len(reward_traj) == 1000:
                x = self.critic(torch.tensor(next_observations[i][-1]).to(device))[0].item()
            q = np.zeros(len(reward_traj))
            for j in range(len(reward_traj) - 1, -1, -1):
                x = reward_traj[j] + gamma * x
                q[j] = x
            q = (q - np.mean(q)) / np.std(q)
            q_values.append(q)
        return q_values

    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #*********YOUR CODE HERE******************
        #self.train()
        max_ep_len = env.spec.max_episode_steps
        batch_size = self.hyperparameters['batch_size']
        trajs = utils.sample_n_trajectories(env, self.get_action_train, batch_size, max_ep_len)
        num = sum([len(traj['reward']) for traj in trajs])
        rewards = [traj['reward'] for traj in trajs]
        baseline = sum([np.sum(reward_traj) for reward_traj in rewards]) / batch_size
        observations = [traj['observation'] for traj in trajs]
        next_observations = [traj['next_observation'] for traj in trajs]
        actions = [traj['action'] for traj in trajs]
        advantage = [np.sum(reward_traj) - baseline for reward_traj in rewards]
        q_values = self.get_q_values(rewards, next_observations)
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        if self.hyperparameters['critic']:
            actor_loss, critic_loss = self.update(observations, actions, advantage, rewards, next_observations, q_values)
        else:
            loss = self.update(observations, actions, advantage)
        actor_loss.backward()
        #critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.1, norm_type = 2)
        #torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1., norm_type = 2)
        self.optimizer_actor.step()
        #self.optimizer_critic.step()
        curr_loss = actor_loss.item() + critic_loss.item()
        self.num_itr += 1
        schedule = self.hyperparameters['schedule']
        self.running_actor_loss += actor_loss.item()
        self.running_critic_loss += critic_loss.item()
        if self.num_itr % schedule == 0:
            eval_trajs = utils.sample_n_trajectories(
                env, self.get_action, 32, max_ep_len
            )
            reward_cum = sum(np.sum(eval_traj['reward']) for eval_traj in eval_trajs) / len(eval_trajs)
            self.running_actor_loss /= schedule
            self.running_critic_loss /= schedule
            self.scheduler_actor.step(self.running_actor_loss)
            #self.scheduler_critic.step(self.running_critic_loss)
            print("Reward and actor-critic loss respectively after %d iterations : " %self.num_itr, reward_cum, self.running_actor_loss, self.running_critic_loss)
            self.running_actor_loss = 0.0
            self.running_critic_loss = 0.0
            if reward_cum > self.max_reward:
                model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
                torch.save(self.state_dict(), os.path.join(model_save_path, "model_" + self.args.env_name + "_" + self.args.exp_name + ".pth"))
                self.max_reward = reward_cum
        if self.num_itr == self.hyperparameters['num_itr']:
            times = [i for i in range(1, len(self.plot_actor_loss) + 1)]
            plt.plot(times, self.plot_actor_loss)
            plt.savefig('actor_loss.png')
            plt.close()
            plt.plot(times, self.plot_critic_loss)
            plt.savefig('critic_loss.png')
            plt.close()
        return {'episode_loss': curr_loss, 'trajectories': trajs, 'current_train_envsteps': num} #you can return more metadata if you want to







class ImitationSeededRL(ImitationAgent):
    '''
    Implement a policy gradient agent with imitation learning initialization.
    You can use the ImitationAgent and RLAgent classes as parent classes.

    Note: We will evaluate the performance on Ant domain only. 
    If everything goes well, you might see an ant running and jumping as seen in lecture slides.
    '''
    
    def __init__(self, observation_dim:int, action_dim:int, args = None, discrete:bool = False, **hyperparameters ):
        super(ImitationSeededRL,self).__init__(observation_dim, action_dim, args, False, **hyperparameters)
        self.hyperparameters = hyperparameters
        self.action_dim  = action_dim
        self.observation_dim = observation_dim
        self.is_action_discrete = discrete
        self.args = args
        #initialize your model and optimizer and other variables you may need
        self.imi_agent = ImitationAgent(observation_dim, action_dim, args, False, **hyperparameters)
        self.covar = torch.tensor(hyperparameters['sigma']*np.ones((1,action_dim)), dtype = torch.float32)
        self.num_itr = 0
        self.max_reward = 0
        self.critic = ptu.build_mlp(observation_dim, 1, hyperparameters['n_layers_critic'], hyperparameters['hidden_dim'], activation = 'relu').to(device)
        self.optimizer_critic = optim.RMSprop(self.critic.parameters(), lr = hyperparameters['lr_critic'], weight_decay = 1e-3)
        self.scheduler_critic = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer_critic, mode = 'min', patience = 10, factor = 0.5, min_lr = 1e-7)
        self.critic_loss = nn.SmoothL1Loss(reduction = 'sum')
        self.max_reward = 0
        self.running_actor_loss = 0.0
        self.running_critic_loss = 0.0

    def forward(self, observation: torch.FloatTensor, action: torch.FloatTensor):
        #****YOUR CODE HERE*******
        mean, var = self.imi_agent.model(observation.to(device)),self.covar
        
        var = torch.diag_embed(var)
        batch_normal = MultivariateNormal(mean, var)
        log_probs = batch_normal.log_prob(action)
        return log_probs.unsqueeze(1).to(device)

    @torch.no_grad()
    def get_action(self, observation: torch.FloatTensor):
        #****YOUR CODE HERE*******
        mean = self.imi_agent.model(observation.to(device))
        return mean.detach().to(device)
    
    @torch.no_grad()
    def get_action_train(self, observation: torch.FloatTensor):
        mean, var = self.imi_agent.model(observation.to(device)),self.covar
        #var = torch.clamp(var, 1e-6, 4.)
        var = torch.diag_embed(var)
        normal = MultivariateNormal(mean, var)
        action = normal.sample()
        action = torch.clamp(action, min = -1.0, max = 1.0)
        return action.detach().to(device)

    def policy_update(self, observations, actions, advantage):
        loss = 0.0
        total_steps = 0
        for i in range(len(observations)):
            log_probs = self.forward(torch.Tensor(observations[i]).to(device), torch.Tensor(actions[i]).to(device))
            if self.hyperparameters['critic']:
                ad = torch.Tensor(advantage[i]).to(device)
            else:
                ad = torch.Tensor([advantage[i]]).to(device)
            ac = torch.Tensor(actions[i]).to(device)
            loss += torch.sum(ad * -log_probs)
            loss += self.hyperparameters['entropy_weight'] * torch.sum(-log_probs)
            if loss.isnan():
                print(log_probs[log_probs.isnan()], sep = '\n')
            total_steps += observations[i].shape[0]
        return loss / total_steps

    def critic_update(self, observations, q_values):
        critic_loss = 0.0
        total_steps = 0
        for i in range(len(observations)):
            pred_values = self.critic(torch.Tensor(observations[i]).to(device))
            critic_loss += self.critic_loss(pred_values, torch.Tensor(q_values[i].reshape(-1, 1)).to(device))
            total_steps += observations[i].shape[0]
        critic_loss /= total_steps
        return critic_loss
        
    def update(self, observations, actions, advantage, rewards = None, next_observations = None, q_values = None):
        #****YOUR CODE HERE*******
        if self.hyperparameters['critic']:
            gamma = self.hyperparameters['discount']
            advantage_critic = []
            targets = []
            for i in range(len(observations)):
                pred_values_curr = self.critic(torch.Tensor(observations[i]).to(device)).detach()
                pred_values_next = self.critic(torch.Tensor(next_observations[i]).to(device)).detach()
                if pred_values_next.size(0) < 1000:
                    pred_values_next[-1] = 0
                advantage_critic.append((torch.Tensor(rewards[i].reshape(-1, 1)).to(device) + gamma * pred_values_next - pred_values_curr).detach())
            critic_loss = self.critic_update(observations, q_values)
            actor_loss = self.policy_update(observations, actions, advantage_critic)
        else:
            loss = self.policy_update(observations, actions, advantage)
        return actor_loss, critic_loss   
    
    def get_q_values(self, rewards, next_observations):
        q_values = []
        gamma = self.hyperparameters['discount']
        m = len(rewards)
        for i in range(m):
            reward_traj = rewards[i]
            x = 0
            if len(reward_traj) == 1000:
                x = self.critic(torch.tensor(next_observations[i][-1]).to(device))[0].item()
            q = np.zeros(len(reward_traj))
            for j in range(len(reward_traj) - 1, -1, -1):
                x = reward_traj[j] + gamma * x
                q[j] = x
            q = (q - np.mean(q)) / np.std(q)
            q_values.append(q)
        return q_values
    
    def train_iteration(self, env, envsteps_so_far, render=False, itr_num=None, **kwargs):
        #****YOUR CODE HERE*******
        #self.train()
        if(itr_num<20 or (itr_num%21 == 0)):
                    train_info = self.imi_agent.train_iteration(env, envsteps_so_far , render, itr_num )
                    self.num_itr+=1
                    return train_info
        else:
            max_ep_len = env.spec.max_episode_steps
            batch_size = self.hyperparameters['batch_size']
            trajs = utils.sample_n_trajectories(env, self.get_action_train, batch_size, max_ep_len)
            num = sum([len(traj['reward']) for traj in trajs])
            rewards = [traj['reward'] for traj in trajs]
            baseline = sum([np.sum(reward_traj) for reward_traj in rewards]) / batch_size
            observations = [traj['observation'] for traj in trajs]
            next_observations = [traj['next_observation'] for traj in trajs]
            actions = [traj['action'] for traj in trajs]
            advantage = [np.sum(reward_traj) - baseline for reward_traj in rewards]
            q_values = self.get_q_values(rewards, next_observations)
            self.imi_agent.optimizer.zero_grad()
            self.optimizer_critic.zero_grad()
            if self.hyperparameters['critic']:
                actor_loss, critic_loss = self.update(observations, actions, advantage, rewards, next_observations, q_values)
            else:
                loss = self.update(observations, actions, advantage)
            actor_loss.backward()
            critic_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.imi_agent.model.parameters(), 1., norm_type = 2)
            torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1., norm_type = 2)
            self.imi_agent.optimizer.step()
            self.optimizer_critic.step()
            curr_loss = actor_loss.item() + critic_loss.item()
            self.num_itr += 1
            schedule = self.hyperparameters['schedule']
            self.running_actor_loss += actor_loss.item()
            self.running_critic_loss += critic_loss.item()
            if self.num_itr % schedule == 0:
                eval_trajs = utils.sample_n_trajectories(
                    env, self.get_action, 32, max_ep_len
                )
                reward_cum = sum(np.sum(eval_traj['reward']) for eval_traj in eval_trajs) / len(eval_trajs)
                self.running_actor_loss /= schedule
                self.running_critic_loss /= schedule
                self.scheduler_critic.step(self.running_critic_loss)
                print("Reward and actor-critic loss respectively after %d iterations : " %self.num_itr, reward_cum, self.running_actor_loss, self.running_critic_loss)
                self.running_actor_loss = 0.0
                self.running_critic_loss = 0.0
                if reward_cum > self.max_reward:
                    model_save_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../best_models")
                    torch.save(self.state_dict(), os.path.join(model_save_path, "model_" + self.args.env_name + "_" + self.args.exp_name + ".pth"))
                    self.max_reward = reward_cum
            return {'episode_loss': curr_loss, 'trajectories': trajs, 'current_train_envsteps': num} #you can return more metadata if you want to


