from OEDAlgorithm import *
import numpy as np
from itertools import permutations
from time import time

class GreedyOED(OEDAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.greedy_sequence = []
        self.greedy_value_sequence = []
        self.greedy_time_sequence = []
        self.target_number_of_sensors = self.m

        # Attempt to load existing greedy output
        output_filename = "greedy_" + self.target_filename
        self.output_filename = output_filename
        # Attempt to find a stored decomposition with the same target
        for filename in listdir(self.outputs_directory):
            if isfile(join(self.outputs_directory, filename)):
                if output_filename in filename:
                    with open(self.outputs_directory + filename, "rb") as input_file:
                        obj = pickle.load(input_file)
                        self.greedy_sequence = obj["greedy_sequence"]
                        self.greedy_value_sequence = obj["greedy_value_sequence"]
                        self.greedy_time_sequence = obj["greedy_time_sequence"]
                        
    def Greedy(self, initial_design = None, target_number_of_sensors = None, verbose = False):
        
        if initial_design is None:
            search_init = len(self.greedy_sequence)
        else:
            search_init = int(np.ceil(np.sum(initial_design)))
            current_design = deepcopy(initial_design)
            current_sequence = []
            if search_init == target_number_of_sensors:
                return current_design, current_sequence
            
        for i in range(search_init, target_number_of_sensors):
            greedy_target = i + 1
            valgreedy = np.inf
            
            if initial_design is None:
                wgreedy = self.greedy_design(i)
            else:
                wgreedy = deepcopy(current_design)
                
            search_inds = np.argwhere(wgreedy == 0)
            num_inds = len(search_inds)
            greedy_time = time()
            for j, ind in enumerate(search_inds):
                if self.verbose:
                    print("\r","Finding greedy design for target ",greedy_target,": ",j,"/",num_inds, \
                      ", current value: ",valgreedy,sep="",end="")
                w = deepcopy(wgreedy)
                w[ind] = 1
                val = self.J_init.eval(w)
                if val < valgreedy:
                    valgreedy = deepcopy(val)
                    greedy_ind = deepcopy(ind)
                    
            if initial_design is None:
                self.greedy_sequence.append(greedy_ind)
                self.greedy_value_sequence.append(valgreedy)
                self.greedy_time_sequence.append(time() - greedy_time)

                #print("\nFound greedy design for target ",greedy_target," with value ",valgreedy,sep="")

                # Save output to external file
                if self.output_filename is not None:
                    obj = {"greedy_sequence": self.greedy_sequence, \
                           "greedy_value_sequence": self.greedy_value_sequence, \
                           "greedy_time_sequence": self.greedy_time_sequence}
                    with open(self.outputs_directory + self.output_filename, "wb") as output_file:
                        pickle.dump(obj, output_file)
                        
            else:
                current_sequence.append(greedy_ind)
                current_design[greedy_ind] = 1
                
        if initial_design is not None:
            return current_design, current_sequence
                
    def greedy_design(self, target_number_of_sensors, initial_design = None):
        w = np.zeros(self.m_sensors)
        if initial_design is not None:
            current_design, current_sequence = self.Greedy(target_number_of_sensors = target_number_of_sensors, initial_design = initial_design)
            return current_design, current_sequence
        else:
            if target_number_of_sensors > 0:
                if len(self.greedy_sequence) < target_number_of_sensors:
                    self.Greedy(target_number_of_sensors = target_number_of_sensors)
                w[np.array(self.greedy_sequence[:target_number_of_sensors],dtype=int)] = 1
            return w
            