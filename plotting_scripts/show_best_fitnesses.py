import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

models = []
values = []
iterations = []
embeds = []
for model_name in ["Relso", "RelsoLin_k25", "RelsoLin_k60"]:
    for embed in [20, 100]:
        model = f"{model_name}_embed{embed}"
        fitnesses_path = f"results/optim_logs/{model}relso1/optimization_fitnesses.npy"
        fitnesses = np.load(fitnesses_path).squeeze()
        for iteration in range(10):
            models.append(model_name.split("_")[0])
            embeds.append(embed)
            values.append(fitnesses[iteration, -1])
            iterations.append(iteration)

merged_data = pd.DataFrame({"Model": models, "Fitness": values, "Iteration": iterations, "Embedding size": embeds})
merged_data.to_csv("merged_fitness_data.csv")
sns.boxplot(data=merged_data, x="Model", y="Fitness", hue="Embedding size")
plt.title("Distribution of fitness")
plt.show()
