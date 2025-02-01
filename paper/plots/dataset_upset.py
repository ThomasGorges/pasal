import pandas as pd
from upsetplot import UpSet
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.transforms as transforms

matplotlib.rcParams.update({'font.size': 12})

df = pd.read_json("../output/preprocessing/fusion.json")
df = df.rename({"leffingwell": "Leffingwell", "sigma_2014": "Sigma", "arctander": "Arctander", "ifra_2019": "IFRA", "dream_mean": "DREAM"})
df = df.drop(["IsomericSMILES", "dream_std"])
df = df.T.notna().astype(int)

counts_df = df.groupby(list(df.columns)).size()

print(f"Dataset size: {df.sum(axis=0)}")

facecolor = "#04316A"

upset = UpSet(counts_df, subset_size="auto", show_counts=True, min_degree=2, sort_by="degree", sort_categories_by="cardinality", facecolor=facecolor)
upset.plot()

fig = plt.gcf()

for t in fig.axes[3].texts:
    t.set_transform(t.get_transform() + transforms.ScaledTranslation(12 / fig.dpi, 0, fig.dpi_scale_trans))
    t.set_fontsize(10)
    t.set_rotation(45)

unique_mol_rows = df[df.sum(axis=1) == 1]
unique_counts = unique_mol_rows.sum(axis=0)

yticklabels = fig.axes[2].get_yticklabels()

fig.axes[1].invert_yaxis()
fig.axes[2].clear()
bars = fig.axes[2].barh([x.get_text() for x in yticklabels], [unique_counts[x.get_text()] for x in yticklabels], color=facecolor, edgecolor="black", height=0.6)
offset = 100
for b in bars:
    w = b.get_width() + b.get_x()
    fig.axes[2].text(w + offset, b.get_y() + 0.5 * b.get_height(), str(w), ha="right", va="center")

fig.axes[2].invert_xaxis()
fig.axes[2].set_xlabel("Unique set size")

plt.title("Heterogeneity of odor datasets")

plt.tight_layout()
plt.subplots_adjust(left=0.05, right=0.95)

ax2_pos = fig.axes[2].get_position()
fig.axes[2].set_position([ax2_pos.x0 + 0.05, ax2_pos.y0, ax2_pos.width, ax2_pos.height])

plt.savefig('../output/plots/fig5.pdf', dpi=1200, bbox_inches="tight", pad_inches=0)
plt.show()
