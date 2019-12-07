import os
from operator import itemgetter

path = './test_fake/'  # model directory path should go here.

files = os.listdir(path)
last_modification = [(os.path.getmtime(os.path.join(path, f)), f) for f in files]

# Sort the list by last modified.
last_modification.sort(key=itemgetter(0))

# Delete everything but the last 10 files.
for time, f in last_modification[:-10]:
    os.remove(os.path.join(path, f))

