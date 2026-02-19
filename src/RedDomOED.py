from OEDAlgorithm import *

class RedDomOED(OEDAlgorithm):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
    def determine_red_dom(self, w, stage = None):
        J = self.Jac(w)

        # Test the optimality criteria to see if we stepped to a global minimum.
        # Only bother if we actually have enough active sensors.
        if np.sum(w) == self.target_number_of_sensors:
            if self.test_optimality(w, J):
                return np.array([])

        if stage == 0:
            Cx = self.C0
        elif stage == 1:
            Cx = self.C1
        elif stage is None:
            Cx = self.C2
        else:
            assert "Wrong stage!"

        is_dom = np.zeros(len(self.free_indices), dtype = bool)
        is_red = np.zeros(len(self.free_indices), dtype = bool)
        for k in range(len(self.free_indices)):
            dominated_indices = J > J[k] + 2 * Cx
            dominating_indices = J < J[k] - 2 * Cx

            is_dom[k] = np.sum(dominated_indices) >= self.current_m - self.current_target
            is_red[k] = np.sum(dominating_indices) >= self.current_target

        dominant_indices = self.free_indices[is_dom]
        redundant_indices = self.free_indices[is_red]

        self.dominant_sequence.append(dominant_indices)
        self.redundant_sequence.append(redundant_indices)
        
        self.vprint("Stage " + str(stage) + " found " + str(np.sum(is_dom)) + " dominant indices and "\
                    + str(np.sum(is_red)) + " redundant indices.")

        self.update(dominant_indices = dominant_indices, redundant_indices = redundant_indices)
        return J[np.logical_not(np.logical_or(is_dom,is_red))]

    def step(self, J):
        
        w = np.zeros(self.m_sensors, dtype = int)
        w[self.free_indices[np.argpartition(-J, -self.current_target)[-self.current_target:]]] = 1
        w[self.dom_indices] = 1
        return w

    def RedDom(self, target_number_of_sensors, scale = 1, verbose = False):
        
        self.verbose = verbose
        self.out_flag = [0,0,"Failed"]
        
        assert scale > 0, "Only positive scale accepted."
        self.R *= scale
        self.Rfree *= scale
        
        self.dominant_sequence = []
        self.redundant_sequence = []
        
        # Target number can be set freely, all cases use the decomposition computed in init.
        self.target_number_of_sensors = target_number_of_sensors
        self.current_target = 1 * self.target_number_of_sensors
        # Reset free indices for this test.
        self.free_indices = np.arange(self.m_sensors, dtype = int)
        self.dom_indices = np.array([], dtype = int)
        self.red_indices = np.array([], dtype = int)
        self.update(dominant_indices = np.array([], dtype = int), redundant_indices = np.array([], dtype = int))
        filter_round = 0
        
        w0 = np.zeros(self.m_sensors, dtype = int)
        w1 = np.ones(self.m_sensors, dtype = int)
        while True:

            self.fixed_doms = 0
            self.fixed_reds = 0

            self.out_flag[1] = 0
            J0 = self.determine_red_dom(w0, stage = 0)
            if self.current_target == 0:
                self.vprint("Finished reducing in iteration " + str(self.out_flag[0]) + ", substep 0.")
                break

            self.out_flag[1] = 1
            w0w = self.step(J0)
            J0w = self.determine_red_dom(w0w, stage = None)
            if self.current_target == 0:
                self.vprint("Finished reducing in iteration " + str(self.out_flag[0]) + ", substep 1.")
                break

            self.out_flag[1] = 2
            J1 = self.determine_red_dom(w1, stage = 1)
            if self.current_target == 0:
                self.vprint("Finished reducing in iteration " + str(self.out_flag[0]) + ", substep 2.")
                break
            
            self.out_flag[1] = 3
            w1w = self.step(J1)
            J1w = self.determine_red_dom(w1w, stage = None)
            if self.current_target == 0:
                self.vprint("Finished reducing in iteration " + str(self.out_flag[0]) + ", substep 3.")
                break

            current_free = len(self.free_indices)
            current_dom = len(self.dom_indices)
            current_red = len(self.red_indices)
            
            if self.fixed_doms == 0 and self.fixed_reds == 0:
                self.vprint("No further filtering possible, terminating with " + str(current_free) + " remaining free indices out of " + str(self.m_sensors) + " originally.")
                self.vprint(str(current_dom) + " indices fixed as dominant out of target " + str(self.target_number_of_sensors) + ", " + str(current_red) + " fixed as redundant.")
                break
            else:
                self.vprint("In filtering round " + str(filter_round) + ", " + str(self.fixed_doms) + " indices were fixed as dominant and " + str(self.fixed_reds) + " indices were fixed as redundant.")
                if current_free == 0:
                    print("All indices fixed.")
                    break
                self.vprint(str(current_free) + " remaining free indices out of " + str(self.m_sensors) + " originally.")
            
            filter_round += 1
            self.out_flag[0] += 1
        
        self.vprint("In filtering round " + str(filter_round) + ", " + str(self.fixed_doms) + " indices were fixed as dominant and " + str(self.fixed_reds) + " indices were fixed as redundant.")
        self.vprint("All indices fixed.")
        
        if len(self.dom_indices) < self.target_number_of_sensors:
            if not self.silent:
                print("Failed for",self.target_number_of_sensors)
            if self.red_indices.size or self.dom_indices.size:
                self.out_flag[2] = "Partial"
            else:
                self.out_flag[2] = "Failed"
            self.design = None
        else:
            if not self.silent:
                print("Succeeded for",target_number_of_sensors)
            self.out_flag[2] = "RedDom"
            self.design = np.zeros(self.m_sensors)
            self.design[self.dom_indices] = 1
            
        self.R /= scale
        self.Rfree /= scale