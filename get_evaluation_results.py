import pickle
import os

root = 'ssd300_120000/test/'

aps = ''

for file in os.listdir(root):
#file = 'chair_pr.pkl'
with open(root + file, 'rb') as f:
    data = pickle.load(f)
    print(data)
        #aps += file + ': ' + data['ap'] + '\n'

'''aps_txt = open(root + 'aps.txt', 'w')
aps_txt.write(aps)
'''
