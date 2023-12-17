import subprocess
import time

if __name__ == '__main__':
    import multiprocessing

    num_cores = multiprocessing.cpu_count()
    cores_per_task = 1
    max_tasks = num_cores // cores_per_task
    child_processes = []

    """ Finite horizon """
    for num_disc_mf in [50, 80, 120]:
        for variant in ["fpi",]:
            for game in ['SIS',]:
                for init_pi in ["first", "last", "unif"]:
                    p = subprocess.Popen(['python',
                                          './main_fp.py',
                                          f'--cores={cores_per_task}',
                                          f'--game={game}',
                                          f'--fp_iterations={100}',
                                          f'--variant={variant}',
                                          f'--num_disc_mf={num_disc_mf}',
                                          f'--init_pi={init_pi}',
                                          ])
                    child_processes.append(p)

                    time.sleep(5)

                    while len(child_processes) >= max_tasks:
                        for p in list(child_processes):
                            if p.poll() is not None:
                                child_processes.remove(p)
                            time.sleep(1)

    for num_disc_mf in range(10, 135, 10):
        for variant in ["fp", ]:
            for game in ['SIS',]:
                for init_pi in ["first", "last", "unif"]:
                    p = subprocess.Popen(['python',
                                          './main_fp.py',
                                          f'--cores={cores_per_task}',
                                          f'--game={game}',
                                          f'--fp_iterations={1000}',
                                          f'--variant={variant}',
                                          f'--num_disc_mf={num_disc_mf}',
                                          f'--init_pi={init_pi}',
                                          ])
                    child_processes.append(p)

                    time.sleep(5)

                    while len(child_processes) >= max_tasks:
                        for p in list(child_processes):
                            if p.poll() is not None:
                                child_processes.remove(p)
                            time.sleep(1)

    for p in child_processes:
        p.wait()
