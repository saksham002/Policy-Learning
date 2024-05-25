
'''
This file contains the configs used for Model creation and training.
'''
#DONT CHANGE THE OVERALL STRUCTURE OF THE DICTIONARY. 

configs = {
    'Ant-v4': {
        "imitation":{
            "hyperparameters": {
                'lr' : 1e-3,
                'hidden_dim' : 64,
                'm' : 1, 
                'p' : 0.75, 
            },
            "num_iteration": 20,

        },

        "RL":{
               "hyperparameters": {
                    'n_layers_actor' : 3,
                    'n_layers_critic': 3,
                    'hidden_dim': 128,
                    'lr_actor': 1e-5,
                    'lr_critic' : 1e-5,
                    'entropy_weight' : 1e-2,
                    'batch_size' : 1,
                    'max_ep_len': None,
                    'discount' : 0.99,
                    'critic' : True, 
                    'log' : 10,
                    'schedule' : 100,
                    'num_itr' : 100001
            },
            "num_iteration": 100001,
        },

         "imitation-RL":{
               "hyperparameters": {
                    'n_layers_critic':2,
                    'hidden_dim': 64,
                    'lr':1e-4,
                    'lr_critic' : 1e-4,
                    'entropy_weight' : 1e-3,
                    'batch_size' : 1,
                    'm': 1,
                    'max_ep_len': None,
                    'discount' : 0.99,
                    'critic' : True, 
                    'log' : 10,
                    'schedule' : 20,
                    'num_itr' : 20001,
                    'p':0.75,
                    'sigma':0.01
            },
            "num_iteration": 20001,

            
        }

    },


    'Hopper-v4': {
        "imitation":{
              "hyperparameters": {
                'lr' : 1e-3,
                'hidden_dim' : 64,
                'm' : 1, 
                'p' : 0.75, 
            },
            "num_iteration": 20,

        },

        "RL":{
               "hyperparameters": {
                    'n_layers_actor' : 1,
                    'n_layers_critic': 1,
                    'hidden_dim': 32,
                    'lr_actor': 1e-4,
                    'lr_critic' : 1e-4,
                    'entropy_weight' : 1e-2,
                    'batch_size' : 1,
                    'max_ep_len': None,
                    'discount' : 0.99,
                    'critic' : True, 
                    'log' : 10,
                    'schedule' : 500,
                    'num_itr' : 100001
            },
            "num_iteration": 100001,

        },
    },

    'HalfCheetah-v4': {
        "imitation":{
              "hyperparameters": {
                'lr' : 1e-3,
                'hidden_dim' : 64,
                'm' : 1, 
                'p' : 0.5, 
            },
            "num_iteration": 20,
        },

        "RL":{
               "hyperparameters": {
                    'n_layers_actor' : 1,
                    'n_layers_critic': 1,
                    'hidden_dim': 64,
                    'lr_actor': 1e-5,
                    'lr_critic' : 1e-5,
                    'entropy_weight' : 1e-4,
                    'batch_size' : 1,
                    'max_ep_len': None,
                    'discount' : 0.99,
                    'critic' : True, 
                    'log' : 10,
                    'schedule' : 100,
                    'num_itr' : 20001
            },
            "num_iteration": 20001,


        },
        
        "imitation-RL":{
               "hyperparameters": {
                
            },
            "num_iteration": 100,

        }

    },
}
