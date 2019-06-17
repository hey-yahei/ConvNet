import os
import numpy as np
import struct

root = "./outputs/"

def cosine_dist(x, y):
    return np.dot(x,y)/(np.linalg.norm(x)*np.linalg.norm(y))

def evaluate(features1, features2):
    pos_sims = []
    neg_sims = []
    for i, y1 in enumerate(features1):
        for j, y2 in enumerate(features2):
            dist = cosine_dist(y1, y2)
            if i == j:
                pos_sims.append(dist)
            else:
                neg_sims.append(dist)
    neg_sims.sort(reverse=True)
    threshold_00001 = neg_sims[int(len(neg_sims) * 0.0001)]
    eval_acc = (np.array(pos_sims) >= threshold_00001).mean()
    
    return eval_acc

def load(path):
    res = []
    float_packer = struct.Struct("f")
    with open(path, 'rb') as f:
        while True:
            d = f.read(4)
            if not d:
                break
            res.append(float_packer.unpack_from(d)[0])
    return np.array(res)


if __name__ == "__main__":
    features1 = []
    features2 = []
    for fname in os.listdir(root):
        if fname.endswith("_0.dat"):
            feature1 = load(os.path.join(root, fname))
            feature2 = load(os.path.join(root, fname.replace("_0.dat", "_1.dat")))
            features1.append(feature1)
            features2.append(feature2)
    features1 = np.stack(features1, axis=0)
    features2 = np.stack(features2, axis=0)

    print(f"Total: {features1.shape[0] + features2.shape[0]} images.")
    TPR_00001 = evaluate(features1, features2)
    print(f"TPR is {TPR_00001} @FPR=0.0001.")
