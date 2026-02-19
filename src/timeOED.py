from runOED import *

import timeit
from functools import partial
from os import listdir
from os.path import isfile, join

times = []
t_targets = np.unique(np.array([m//100,m//50,m//5]))
repeats = 3

if m < 10000:
        number = 10000
else:
        number = 10000
        
RDOED.J_init.fac = 1

tests_folder = "tests/"
filename = str(m).zfill(5) + "_time"
try:
    mkdir(tests_folder)
except:
    pass

RDOED.silent = True
for target in t_targets:
        t = min(timeit.Timer( \
            partial(RDOED.RedDom,target)).repeat(repeat=repeats, number=number))
        t /= number
        times.append(t)
        print("Test for",target,"/",m,"in",t,"seconds!")

obj = {"times": times, "t_targets": t_targets}
with open(tests_folder + filename, "wb") as output_file:
        pickle.dump(obj, output_file)    
print("Successfully timed with m = " + str(m) + ".")
