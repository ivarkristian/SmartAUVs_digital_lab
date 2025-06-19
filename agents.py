# %%
import numpy as np

class agent_IG():
    def __init__(self, *args, type, **kwargs):
        super().__init__(*args, **kwargs)
        self.type = type
        if self.type not in ['IG', 'UCB', 'DUCB']:
            print(f'Agent type {self.type} not recognized.')
    
    def get_action(self, obs):
        match self.type:
            case 'IG':
                # Highest entropy reduction, in practice go to location with max variance
                loc_x, loc_y = np.unravel_index(obs['map'][1].argmax(), obs['map'][1].shape)
                action = np.array([loc_x, loc_y])

            case 'UCB':
                print(f'{self.type} agent not implemented')
            case 'DUCB':
                print(f'{self.type} agent not implemented')
            case _:
                print(f'{self.type} agent not implemented')
        
        return action

