from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticCfg,
    RslRlPpoAlgorithmCfg,
)

#RslRLOnPolicyRunnerCfg
'''
controls the training loop and experiment management. Sets how long to train, when to save checkpoints, logging, and overall experiment settings
num_steps_per_env: Steps collected per environment before updating (e.g., 24)
max_iterations: Total training iterations (e.g., 1500)
save_interval: How often to save checkpoints (e.g., 50)
experiment_name: Name for logs/checkpoints
obs_groups: Which observation groups go to actor/critic
'''


#RslRLPpoActorCriticCfg
'''
Defines the actor critic network structure. configures the policy and value networks
actor_hidden_dims: Actor network layers (e.g., [256, 128, 64])
critic_hidden_dims: Critic network layers (e.g., [256, 128, 64])
activation: Activation function (e.g., "elu" or "relu")
init_noise_std: Initial exploration noise (e.g., 1.0)
'''


#RslRLPpoAlgorithmCfg
'''
Sets the PPO specific hyperparams
controls clipping, learning rate, entropy, GAE

'''
@configclass
class So101PPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24  
    max_iterations = 1500   
    save_interval = 50     
    experiment_name = "so101_lift"

    obs_groups = {
        "policy": ["policy"],
        "critic": ["policy"],
    }

    policy = RslRlPpoActorCriticCfg(
        actor_hidden_dims=[256, 128, 64], 
        critic_hidden_dims=[256, 128, 64], 
        activation="elu",                   
        actor_obs_normalization=False,      
        critic_obs_normalization=False,     
        init_noise_std=1.5,   
    )

    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,              
        use_clipped_value_loss=True,     
        
        # PPO clipping
        clip_param=0.2,                
        
        # Exploration
        entropy_coef=0.006,              
        
        # Training settings
        num_learning_epochs=5,            
        num_mini_batches=4,               
        learning_rate=1.0e-4,           
        schedule="adaptive",          
        
        # Advantage estimation
        gamma=0.98,                    
        lam=0.95,                       
        
        # KL divergence (for adaptive LR)
        desired_kl=0.01,               
        
        # Gradient clipping
        max_grad_norm=1.0,              
    )