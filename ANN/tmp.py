from tmp2 import f
import os
import multiprocessing as mp


mp.set_start_method("spawn")
with mp.Pool(os.cpu_count()) as p:
    print(p.map(f, [1, 2, 3]))
