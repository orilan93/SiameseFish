import numpy as np
import shap
from shap.plots.colors import red_transparent_blue

import const
import models
from data import load_dataset
import matplotlib.pyplot as plt

DATASET_FLAVOUR = const.FISH_HEAD

X_train, y_train, X_test, y_test, classes = load_dataset(DATASET_FLAVOUR, preprocess=True)

model = models.triplet_network_ohnm
model.load_weights("../models/embedding")

explainer = shap.GradientExplainer(model, X_train)

selected = [0, 1, 2]#np.random.choice(len(X_test), size=3, replace=False)
print(y_test[selected])
print(list(classes[idx] for idx in y_test[selected]))

nodes = np.array([[0, 10], [1, 10], [2, 10]])

nodes = []
for s in selected:
    ids = np.where(y_train == s)[0]
    nodes.append(ids[:1])
nodes = np.array(nodes)
print(nodes)
nodes = nodes[:, :1]
print(nodes)

shap_values, indices = explainer.shap_values(X_test[selected], ranked_outputs=nodes, output_rank_order="custom")

np.save("shap", shap_values)

print(indices)

index_names = np.vectorize(lambda x: classes[x])(indices)

img = shap_values.reshape(shap_values.shape[1:])[1]
abs_val = np.abs(img)
max_val = np.nanpercentile(abs_val, 99.9)
plt.imshow(img, cmap=red_transparent_blue, vmin=-max_val, vmax=max_val)

plt.show()

np.save("values", shap_values)

shap.image_plot(shap_values, X_test[selected], show=False)
plt.savefig("shap.png")
