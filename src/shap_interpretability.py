import numpy as np
import shap

import const
import models
from data import load_dataset
import matplotlib.pyplot as plt

DATASET_FLAVOUR = const.FISH_MERGED

X_train, y_train, X_test, y_test, classes = load_dataset(DATASET_FLAVOUR, preprocess=True)

model = models.triplet_network_ohnm
model.load_weights("models/embedding")

explainer = shap.GradientExplainer(model, X_train)

selected = [0, 1, 2]#np.random.choice(len(X_test), size=3, replace=False)
print(y_test[selected])
print(list(classes[idx] for idx in y_test[selected]))

shap_values, indices = explainer.shap_values(X_test[selected], ranked_outputs=2)

print(indices)

index_names = np.vectorize(lambda x: classes[x])(indices)

shap.image_plot(shap_values, X_test[selected], show=False)
plt.savefig("shap.png")
