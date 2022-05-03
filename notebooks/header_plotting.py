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
mpl.rcParams["xtick.labelsize"] = 6.5
mpl.rcParams["ytick.labelsize"] = 6.5
mpl.rcParams["axes.labelsize"] = 6.5
mpl.rcParams["axes.titlesize"] = 6.5
mpl.rcParams["figure.titlesize"] = 6.5
mpl.rcParams["legend.fontsize"] = 5
mpl.rcParams["figure.dpi"] = 300  # this primarily affects the size on screen

# Save figure as pdf and png
save_kwargs = {"transparent": True, "dpi": 300, "bbox_inches": "tight"}
