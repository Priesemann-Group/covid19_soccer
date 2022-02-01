import matplotlib as mpl

# Matplotlib config
mpl.rcParams.update(mpl.rcParamsDefault)
mpl.rcParams["font.sans-serif"] = "Arial"
mpl.rcParams["font.family"] = "sans-serif"
# mpl.rcParams['axes.linewidth'] = 0.3
mpl.rcParams["axes.labelcolor"] = "black"
mpl.rcParams["axes.edgecolor"] = "black"
mpl.rcParams["xtick.color"] = "black"
mpl.rcParams["ytick.color"] = "black"
mpl.rcParams["axes.spines.right"] = False
mpl.rcParams["axes.spines.top"] = False
mpl.rcParams["xtick.labelsize"] = 8
mpl.rcParams["ytick.labelsize"] = 8
mpl.rcParams["axes.labelsize"] = 9
mpl.rcParams["axes.titlesize"] = 12
mpl.rcParams["legend.fontsize"] = 6

# Save figure as pdf and png
save_kwargs = {"transparent": True, "dpi": 300, "bbox_inches": "tight"}
