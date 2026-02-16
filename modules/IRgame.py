# Build the decision problem object from the raw file

#%% 
import numpy as np
from modules.utils import sample_from_simplex
from modules.preprocessing.check_format import check_format
from modules.preprocessing.full_info_game import construct_game_tree

#%%
class IRgame():
    def __init__(self, path_to_file):
        self.path_to_file = path_to_file
        self.descr = path_to_file.split('/')[-1].split('.')[0]
    
        if not hasattr(self, 'infosets_dict'):
            self.get_game_tree()


    def check_format(self):
        return check_format(self.path_to_file)

    def get_game_tree(self):
        self.nodes_dict, self.infosets_dict, self.infoset_keys = construct_game_tree(self.path_to_file)
        assert len(self.infosets_dict) == len(self.infoset_keys)
        assert set(self.infosets_dict) == set(self.infoset_keys)

    def get_random_point(self, method = "exponential", np_rng=None):
        if np_rng is None:
            np_rng = np.random
        
        init = []
        if method == "exponential":
            for key in self.infoset_keys:
                unnormalized_probs = np_rng.exponential(scale=1.0, size=self.infosets_dict[key].num_actions)
                init.append( unnormalized_probs / np.sum(unnormalized_probs) )
        elif method == "uniform":
            for key in self.infoset_keys:
                init.append( sample_from_simplex( self.infosets_dict[key].num_actions - 1 , np_rng=np_rng ) )
        else:
            raise ValueError("`method` must be 'exponential' or 'uniform'")
        
        return np.concatenate( init )

    def conv_pt_to_profile(self, point, round=True):
        
        if round:
            #Round to four digits
            point = np.around(point, decimals=4)

        profile = {}
        past_num_actions = 0
        for key in self.infoset_keys:
            infoset_actions = self.infosets_dict[key].actions
            mixed_action = ''
            for i in range(len(infoset_actions)):
                #Don't bother adding an action that isn't being used
                if point[ past_num_actions + i ] != 0:
                    #Case distinction so for whether we need to add a plus sign in the beginning
                    if mixed_action == '':
                        mixed_action += '{} {}'.format( point[ past_num_actions + i ], infoset_actions[i] )
                    else:
                        mixed_action += ' + {} {}'.format( point[ past_num_actions + i ], infoset_actions[i] )

            profile[key] = mixed_action
            past_num_actions += len(infoset_actions)
        
        return profile