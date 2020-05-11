# NOT USED
# visual acuracy test
fig, axes = plt.subplots(3, 1)
sns_heatmap_annot = False
# acuracy test
sns.heatmap(resp[-1])
plt.show()
sns.heatmap(stdevimg, ax=axes[0], annot=sns_heatmap_annot)
plt.show()
