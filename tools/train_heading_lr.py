import numpy as np, joblib, pathlib, random
from sklearn.linear_model import LogisticRegression

# -------- synth dataset (simple heuristics) ----------
N = 10000
X = np.zeros((N, 8), np.float32)
y = np.zeros(N, int)

for i in range(N):
    # random feature values within plausible ranges
    font_z  = np.random.normal()
    bold    = np.random.rand() < 0.4
    italic  = np.random.rand() < 0.1
    capr    = np.random.rand()
    indent  = np.random.rand() * 0.5
    y_pos   = np.random.rand()
    loglen  = np.random.rand() * 5
    numpre  = np.random.randint(0, 6)

    X[i] = [font_z, bold, italic, capr, indent, y_pos, loglen, numpre]

    # heuristic label: heading if big font OR bold & near left & y_pos<0.3 …
    score = (font_z > 0.8) + bold*0.6 + (numpre>0)*0.4 + (y_pos<0.35)*0.3
    y[i] = 1 if score > 1 else 0

clf = LogisticRegression(max_iter=300, class_weight="balanced")
clf.fit(X, y)
pth = pathlib.Path("models")
pth.mkdir(exist_ok=True)
joblib.dump(clf, pth / "lr_heading.pkl")
print("Saved model ➜ models/lr_heading.pkl")
