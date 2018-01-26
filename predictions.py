import csv, os
import numpy as np
import scipy
from scipy.sparse import coo_matrix, csr_matrix

dataPath = "data/"
graphPath = os.path.join(dataPath, "graph")
demographyPath = os.path.join(dataPath, "trainDemography")
testUsersPath = os.path.join(dataPath, "users")
resultPath = "result"

maxTotal = 47289241
linksCount = 27261623

testUsers = set()
for line in csv.reader(open(testUsersPath)):
    testUsers.add(int(line[0]))
print("loaded test Users")


def load_csr(path):
    loaded = np.load(path + ".npz")
    return csr_matrix((loaded['data'], loaded['indices'], loaded['indptr']), shape=loaded['shape'])


testGraph = load_csr(os.path.join(resultPath, "testGraph"))
print("loaded testGraph")
birthDates = np.load(os.path.join(resultPath, "birthDates.npy"))
print("loaded dates")


if not os.path.exists(resultPath): os.makedirs(resultPath)

age = 0
mask = 0
ran = 0
friend = 0
middle = 0

with open(os.path.join(resultPath, "prediction_mask.csv"), "w") as outputMask:
    writerMask = csv.writer(outputMask, delimiter=',')

    for user in testUsers:
        fromMask = 0
        avg = 0
        row = testGraph[user - 1,:]
        ran = len(row.indices)
        for i in range(len(row.indices)):
            mask = row.data[i]
            friend = row.indices[i]
            age = birthDates[friend]
            if age == 0:
                continue
            avg += age
            fromMask += age
            if mask == 33:
                fromMask += (age * 5)
                ran += 5
            if mask == 257:
                fromMask += (age * 8)
                ran += 8
            if mask == 1025:
                fromMask += (age * 8)
                ran += 8
            if mask == 16385:
                fromMask += (age * 6)
                ran += 6
            if mask == 32769:
                fromMask += (age * 12)
                ran += 12
            if mask == 1048577:
                fromMask += (age * 7)
                ran += 7
        avg = avg / len(row.indices)
        fromMask = fromMask / ran
        writerMask.writerow([user, int(fromMask)])

print("Saved")
