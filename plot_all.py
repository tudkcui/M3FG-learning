import os
import subprocess

if __name__ == '__main__':

    """ Finite horizon """
    for file in os.listdir("./eval"):
        p = subprocess.Popen(['python',
                              f"./eval/{file}",
                              ])
        p.wait()
