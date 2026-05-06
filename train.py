import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set random seed
seed = 42

################################
########## DATA PREP ###########
################################

# Load in the data
df = pd.read_csv("wine_quality.csv")

# Split into train and test sections
y = df.pop("quality")
X_train, X_test, y_train, y_test = train_test_split(
    df,
    y,
    test_size=0.2,
    random_state=seed
)

#################################
########## MODELLING ############
#################################

# Fit a model on the train section
regr = RandomForestRegressor(max_depth=5, random_state=seed)
regr.fit(X_train, y_train)

# Report training set score
train_score = regr.score(X_train, y_train) * 100

# Report test set score
test_score = regr.score(X_test, y_test) * 100

# Write scores to a file
with open("metrics.txt", "w") as outfile:
    outfile.write("Training variance explained: %2.1f%%\n" % train_score)
    outfile.write("Test variance explained: %2.1f%%\n" % test_score)

##########################################
##### PLOT FEATURE IMPORTANCE ############
##########################################

# Calculate feature importance in random forest
importances = regr.feature_importances_
labels = df.columns

feature_df = pd.DataFrame(
    list(zip(labels, importances)),
    columns=["feature", "importance"]
)

feature_df = feature_df.sort_values(
    by="importance",
    ascending=False
)

# Professional plot formatting
sns.set_theme(
    style="whitegrid",
    context="talk",
    rc={
        "axes.edgecolor": "#C9D1D9",
        "axes.labelcolor": "#24292F",
        "figure.facecolor": "white",
    }
)

fig, ax = plt.subplots(figsize=(11, 7))

ax = sns.barplot(
    x="importance",
    y="feature",
    data=feature_df,
    color="#4C78A8",
    ax=ax
)

ax.set_title(
    "Random Forest Feature Importance",
    fontsize=22,
    weight="bold",
    pad=20
)

ax.set_xlabel("Importance Score", fontsize=16)
ax.set_ylabel("Feature", fontsize=16)
ax.tick_params(axis="both", labelsize=12)
ax.grid(axis="x", linestyle="--", linewidth=0.8, alpha=0.45)
ax.grid(axis="y", visible=False)
ax.set_xlim(0, feature_df["importance"].max() * 1.15)

# Add importance values at the end of each bar
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=5, fontsize=10)

sns.despine(left=True, bottom=True)
plt.tight_layout()
plt.savefig("feature_importance.png", dpi=150, bbox_inches="tight")
plt.close()

##########################################
############ PLOT RESIDUALS  #############
##########################################

rng = np.random.default_rng(seed)
y_pred = regr.predict(X_test) + rng.normal(0, 0.25, len(y_test))
y_jitter = y_test + rng.normal(0, 0.25, len(y_test))

res_df = pd.DataFrame(
    list(zip(y_jitter, y_pred)),
    columns=["true", "pred"]
)

fig, ax = plt.subplots(figsize=(8, 8))

ax = sns.scatterplot(
    x="true",
    y="pred",
    data=res_df,
    alpha=0.75,
    s=70,
    edgecolor="white",
    linewidth=0.5,
    color="#E45756",
    ax=ax
)

ax.set_title(
    "Predicted vs True Wine Quality",
    fontsize=22,
    weight="bold",
    pad=20
)

ax.set_xlabel("True Wine Quality", fontsize=16)
ax.set_ylabel("Predicted Wine Quality", fontsize=16)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True, linestyle="--", linewidth=0.8, alpha=0.45)

# Ideal prediction line
ax.plot(
    [2.5, 8.5],
    [2.5, 8.5],
    linestyle="--",
    linewidth=2,
    color="#2F3A4A",
    label="Perfect prediction"
)

ax.set_xlim(2.5, 8.5)
ax.set_ylim(2.5, 8.5)
ax.set_xticks(np.arange(3, 9, 1))
ax.set_yticks(np.arange(3, 9, 1))
ax.set_aspect("equal", adjustable="box")
ax.legend(loc="upper left", frameon=True)

sns.despine()
plt.tight_layout()
plt.savefig("residuals.png", dpi=150, bbox_inches="tight")
plt.close()
