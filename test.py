import seaborn as sns, matplotlib.pyplot as plt
import pdb
sns.set(style="whitegrid")

tips = sns.load_dataset("tips")
pdb.set_trace()

sns.barplot(x="day", y="total_bill", data=tips, capsize=.1, ci="sd")
sns.swarmplot(x="day", y="total_bill", data=tips, color="0", alpha=.35)

plt.show()