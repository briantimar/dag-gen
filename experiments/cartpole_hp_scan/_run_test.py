import multiprocessing
import os
import sys

def test_task(arg):
    print(f"This is a task with pid {os.getpid()}")
    return

if __name__ == "__main__":
    nproc = int(sys.argv[1])
    print(f"Now running a bunch of independent tasks in {nproc} processes...")
    args = [ None for __ in range(2 * nproc) ]
    with multiprocessing.Pool(nproc) as p:
        p.map(test_task, args)
    print(f"Python script finished")
