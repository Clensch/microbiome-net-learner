# author: Michael Hüppe
# date: 04.03.2024
# project: biostat/fed_learn/benchmarkFederatedLearning.py

import subprocess

if __name__ == '__main__':

    # Define the command you want to run
    command = 'featurecloud test start --app-image=fed_learn --client-dirs "./data/c1,./data/c2,./data/c3,./data/c4"'

    # Loop 50 times
    for i in range(50):
        # Run the command
        subprocess.run(command, shell=True)
