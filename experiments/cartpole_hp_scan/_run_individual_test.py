import multiprocessing
import os
import sys

def test_task(arg):
    print(f"This is a task with pid {os.getpid()}")
    return

if __name__ == "__main__":
    nproc = int(sys.argv[1])
    print(f"Now running a bunch of independent tasks in {nproc} processes...")
    args = [ None for __ in range(nproc) ]
    processes = []
    for i in range(nproc):
        p = multiprocessing.Process(target=test_task, args=(args[i],))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()

    print(f"Python script finished")
