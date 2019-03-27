import os
import fire
import subprocess
from subprocess import Popen, list2cmdline
import time
import sys

PROCESSES = []

def exec_commands(cmds):
    '''
    exec cmds in parallel in multiple processes
    '''

    if not cmds: return

    def done(p):
        return p.poll() is not None

    def success(p):
        return p.returncode == 0

    def fail():
        sys.exit(1)

    while True:
        try:
            while cmds and len(PROCESSES) < 20:
                task = cmds.pop()
                print(list2cmdline(task))
                PROCESSES.append(Popen(task))

            for p in PROCESSES:
                if done(p):
                    if success(p):
                        PROCESSES.remove(p)
                    else:
                        fail()

            if not PROCESSES and not cmds:
                break
            else:
                time.sleep(0.1)

        except KeyboardInterrupt:
            for p in PROCESSES:
                p.kill()


def start_sim(ttt_file_path=os.path.join(os.environ['CE_PATH'], 'ttt', 'cooking_env.ttt'),
              headless=False,
              operating_system='linux'):
    # set vrep path
    # os.environ["RLFPS_PATH"] = "/home/xli4217/rlfps"

    home_dir = os.path.expanduser("~")
    
    cmds = []
    if operating_system == 'linux':
        if headless:
            cmds.append(["xvfb-run", "./vrep.sh", "-h", "-s", "-q", "-gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE", str(ttt_file_path)])
        else:
            cmds.append(["./vrep.sh", "-gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE", str(ttt_file_path)])
    elif operating_system == 'mac':
        if headless:
            cmds.append(["xvfb-run", "./vrep", "-h", "-s", "-q", "-gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE", str(ttt_file_path)])
        else:
            cmds.append(["./vrep", "-gREMOTEAPISERVERSERVICE_19999_FALSE_TRUE", str(ttt_file_path)])
    else:
        raise ValueError("OS not supported")
            
    exec_commands(cmds)
        
if __name__ == "__main__":
    fire.Fire(start_sim)