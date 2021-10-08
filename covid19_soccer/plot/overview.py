import datetime
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from .utils import get_from_trace
from .rcParams import colors
import covid19_inference as cov19


def plot_overview_quad(traces,models,dls):
    id2_country = np.array([["England","Scotland"],["Germany","France"]])

    fig = plt.figure(figsize=(6,5))
    # Two columns/rows
    outer_grid = fig.add_gridspec(2, 2, wspace=0.35, hspace=0.25,)
    #out = fig.add_subplot(outer_grid[0:,-1])
    #out.set_ylabel("Percentage of soccer related infections\nduring the duration of the Championship")
    # Two rows
    plot_beta=False
    axes = []
    for a in range(2):
        for b in range(2):

            # gridspec inside gridspec
            if plot_beta:
                inner_grid = outer_grid[a,b].subgridspec(3, 3 , width_ratios=[1,0.3,0.3],wspace=0.5)
            else:
                inner_grid = outer_grid[a,b].subgridspec(3, 2,  width_ratios=[1,0.3])

            # Create three subplots
            # - a1: fraction
            # - a2: R_soccer
            # - a3: alpha mean
            # - a4: beta mean
            country = id2_country[a,b]

            a0 = fig.add_subplot(inner_grid[0,0])
            plot_cases(a0, traces[country], models[country], dls[country], ylims_cases[country])

            a1 = fig.add_subplot(inner_grid[1,0])
            plot_fraction(a1, traces[country], models[country], dls[country], ylims_fraction[country])

            a2 = fig.add_subplot(inner_grid[2,0])
            #plot_rsoccer(a2, traces[country], models[country], dls[country])
            plot_reproductionViolin(a2, traces[country], models[country], dls[country])

            a3 = fig.add_subplot(inner_grid[0:,-1])
            plot_relative_from_soccer(a3, traces[country], models[country], dls[country])

            if plot_beta and id2_country[a,b]!="France":
                a4 = fig.add_subplot(inner_grid[0:,-1])
                plot_relative_from_soccer(a4, traces[country], models[country], dls[country])

            # Markup
            a0.set_title(country)
            for ax in [a0,a1,a2]:
                ax.set_xlim(xlim_ts)
                #Locator
                ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
                ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
                # Hide the right and top spines
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
            #a2.set_ylim(-1,15)
            #a2.set_xlim((xlim_ts[0]-model.sim_begin).days,(xlim_ts[1]-model.sim_begin).days)
            #a2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

            # remove labels for first and second timeseries
            for ax in [a0,a1]:
                ax.set(xticklabels=[])


            # Restrain y axis of violin plots
            a3.set_yticks([-20,0,20])
            if country in ["England","Scotland"]:
                a3.set_ylim(-5,30)
            elif country in ["Germany"]:
                a3.set_ylim(-10,30)
            elif country in ["France"]:
                a3.set_ylim(-30,30)

            fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
            xticks = mtick.FormatStrFormatter(fmt)
            a3.yaxis.set_major_formatter(xticks)



            if plot_beta:
                a4.spines['top'].set_visible(False)
                a4.spines['bottom'].set_visible(False)
                a4.spines['left'].set_visible(False)
                a4.tick_params(bottom=False)       
                a4.set_xlabel("Stadium")
                a3.set_xlabel("Public\nviewing")
                a3.set_ylabel("")

                if a == 1 and b == 1:
                    a4.set_ylim(-2,12)
                    a4.set(yticks=[0,4,8])
                else: 
                    a4.set_ylim(-1,6)
                    a4.set(yticks=[0,2,4])
            if a==1:
                a2.set_ylim(-1.5,1.5)


    fig.align_ylabels()
    # Save figure as pdf and png        
    kwargs = {
        "transparent":True,
        "dpi":300,
        "bbox_inches":"tight"
    }
    fig.savefig(f"{fig_path}/fig_1.pdf", **kwargs)
    fig.savefig(f"{fig_path}/fig_1.png", **kwargs)

    plt.show()
    plt.close(fig=fig)

def plot_overview_single(
    trace,
    model,
    dl,
    ylim_cases=[0,1000],
    ylim_fraction=[0.6,1.5],
    ylim_relative=[0,100],
    xlim_ts=[datetime.datetime(2021,5,30),datetime.datetime(2021,7,23)],
    title=""
    ):
    
    fig = plt.figure(figsize=(6,5))
    
    grid = fig.add_gridspec(3, 2, wspace=0.35, hspace=0.25,width_ratios=[1,0.3])

    a0 = fig.add_subplot(grid[0,0])
    plot_cases(a0, trace, model, dl, ylim_cases)

    a1 = fig.add_subplot(grid[1,0])
    plot_fraction(a1, trace, model, dl, ylim_fraction)

    a2 = fig.add_subplot(grid[2,0])
    #plot_rsoccer(a2, traces[country], models[country], dls[country])
    plot_reproductionViolin(a2, trace, model, dl)

    a3 = fig.add_subplot(grid[0:,-1])
    plot_relative_from_soccer(a3, trace, model, dl, ylim_relative)

    # Markup
    for ax in [a0,a1,a2]:
        ax.set_xlim(xlim_ts)
        #Locator
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))
        ax.xaxis.set_minor_locator(mdates.WeekdayLocator(interval=1))
        # Hide the right and top spines
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %-d'))
    #a2.set_ylim(-1,15)
    #a2.set_xlim((xlim_ts[0]-model.sim_begin).days,(xlim_ts[1]-model.sim_begin).days)
    #a2.xaxis.set_major_locator(mdates.WeekdayLocator(interval=4))

    # remove labels for first and second timeseries
    for ax in [a0,a1]:
        ax.set(xticklabels=[])

    fmt = '%.0f%%' # Format you want the ticks, e.g. '40%'
    xticks = mtick.FormatStrFormatter(fmt)
    a3.yaxis.set_major_formatter(xticks)
    
    a0.set_title(title)

# Functions
def plot_cases(ax,trace,model,dl,ylim):
    """
    Plots number of cases
    """
    new_cases = get_from_trace("new_cases",trace)
    
    cov19.plot._timeseries(
        x = pd.date_range(model.sim_begin,model.sim_end),
        y = (new_cases[:,:,0]+new_cases[:,:,1]) / (dl.population[0,0]+dl.population[1,0]) *1e6, # incidence
        what="model",
        ax=ax,
        color=colors["cases"]
    )
    cov19.plot._timeseries(
        x = pd.date_range(model.data_begin,model.data_end),
        y = (dl.new_cases_obs[:,0,0] + dl.new_cases_obs[:,1,0] )/(dl.population[0,0]+dl.population[1,0])*1e6, # male/female
        what="data",
        ax=ax,
        color=colors["data"],
        ms=1.5,
        alpha=0.8,
    )
    begin = datetime.datetime(2021,6,11)
    end = datetime.datetime(2021,7,11)
    
    ax.fill_betweenx(np.arange(0,1000),begin,end,alpha=0.1)
    ax.set_ylim(ylim)
    ax.set_ylabel("Incidence")

def plot_fraction(ax,trace,model,dl,ylim_fraction):
    
    new_cases = get_from_trace("new_cases",trace)
    
    ## Fraction male/female
    cov19.plot._timeseries(
        x = pd.date_range(model.sim_begin,model.sim_end),
        y = (new_cases[:,:,0]/dl.population[0,0]) / (new_cases[:,:,1]/dl.population[1,0]), # male/female
        what="model",
        ax=ax,
        color=colors["fraction"],
        alpha=1,
        alpha_ci=0.3
    )
    cov19.plot._timeseries(
        x = pd.date_range(model.data_begin,model.data_end),
        y = (dl.new_cases_obs[:,0,0]/dl.population[0,0])/(dl.new_cases_obs[:,1,0]/dl.population[1,0]), # male/female
        what="data",
        ax=ax,
        color=colors["data"],
        ms=1.5,
    )

    ax.set_ylabel("Gender\nimbalance")
    ax.set_ylim(ylim_fraction)
    for label in ax.xaxis.get_ticklabels()[::2]:
        label.set_visible(False)
        
    return ax

def plot_rsoccer(ax,trace,model,dl):
    """
    Plots the base and soccer reproduction number 
    
    Parameters
    ----------
    trace:
        arviz trace of model run
    model:
        corresponding model
    dl:
        dataloader
    """
    R_soccer = get_from_trace("R_t_soccer",trace)
    C_soccer = get_from_trace("C_soccer",trace)
    
    # Plot base and soccer Reproduction number
    cov19.plot._timeseries(
        x = pd.date_range(model.sim_begin,model.sim_end),
        y = R_soccer,
        what="model",
        ax=ax,
        color=colors["Repr"]
    )
    ax.axhline(0,color="tab:gray",ls="--",zorder=-5,lw=0.5)
    ax.set_ylabel("Additive\nreproduct. number")
    ax.set_ylim(-0.85,3.5)
    
    return ax

def plot_alphaMean(ax,trace,model,dl,beta=False):
    
    if not beta:
        alpha = get_from_trace(f"alpha_mean",trace)
        R_soccer = np.exp(alpha)-1
    else:
        try:
            alpha = get_from_trace(f"beta_mean",trace)
            R_soccer = np.exp(alpha)-1
        except:
            return ax
        
    import seaborn.categorical
    seaborn.categorical._Old_Violin = seaborn.categorical._ViolinPlotter
    
    sns.violinplot(
        data=R_soccer,
        scale="width",
        inner="quartile",
        orient="v",
        ax=ax,
        color=colors["Repr"],
        linewidth=1,
        saturation=1
    )
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.collections[0].set_edgecolor(colors["Repr"]) # Set outline colors
    ax.set_xticklabels([])
    ax.set_ylabel("Mean effect of all \n played soccer games")
    ax.axhline(0,color="tab:gray",ls="--",zorder=-5)
    return ax
def plot_reproductionViolin(ax,trace,model,dl):
    """
    Violin plot for the additional R values for each game and country.
    
    """
    key = "alpha"
        
    α_mean = get_from_trace(f"{key}_mean",trace)
    σ_g = get_from_trace(f"sigma_{key}_g",trace)
    Δα_g_sparse = get_from_trace(f"Delta_{key}_g_sparse",trace)
    alpha = α_mean[:,None] + np.einsum("dg,d->dg",Δα_g_sparse,σ_g)

    
    R_soccer = np.exp(alpha)-1
    
    nGames = alpha.shape[-1]
    
    # Get date of game and participants
    games = dl.timetable.loc[(dl.alpha_prior > 0)[0]]
    ticks = [(vals["date"]-model.sim_begin).days for i,vals in games.iterrows()]
    ticks = np.array([vals["date"] for i,vals in games.iterrows()])
    R_soccer = pd.DataFrame(R_soccer,columns=ticks)
    
    p = np.percentile(R_soccer, [0.5, 50, 99.5], axis=0)
    ax.errorbar(
        x=ticks,
        y=p[1],
        yerr=p[0,2],
        #width=2,
        #ecolor="tab:gray",
        color=colors["Repr"],
        ls="",
        marker="_",
        ms=4,
        #color = 'k'
        capsize=1.5,
        #error_kw= {"alpha":1,"lw":0.8,"ecolor":colors[1]}
    )
    ax.axhline(0,color="tab:gray",lw=0.5,alpha=0.5,ls="--")
    #ax.set_xticks([vals["date"] for i,vals in games.iterrows()])
    
    R_t_soccer = get_from_trace("R_t_soccer",trace)
    # Plot base and soccer Reproduction number
    
    # Construct ylabels
    #ylabels= []
    #for i, vals in games.iterrows():
    #    label =vals["date"].strftime("%d.%-m.%y")
    #    label +=f'\n{vals["team1"]} vs {vals["team2"]}'
    #    label +=f'\nin {vals["location"]}'
    #ax.set_xlabel("Effect of game")
    ax.set_ylabel("Additive rep.\nnumber")

    return ax

def plot_relative_from_soccer(ax, trace, model, dl, ylim_relative, begin=None, end=None):
    if begin is None:
        begin = datetime.datetime(2021,6,11)
    if end is None:
        end = datetime.datetime(2021,7,11)

    
    # Get params from trace and dataloader
    new_E_t = get_from_trace('new_E_t',trace)
    S_t = get_from_trace('S_t',trace)
    new_I_t = get_from_trace('new_I_t',trace)
    R_t_base = get_from_trace('R_t_base',trace)
    C_base = get_from_trace('C_base',trace)
    C_soccer = get_from_trace('C_soccer',trace)
    R_t_soccer = get_from_trace('R_t_add_fact',trace)
    pop = model.N_population
    i_begin = (begin-model.sim_begin).days
    i_end = (end-model.sim_begin).days + 1 #inclusiv last day
    
    """ Calculate cases in agegroup because of soccer and without soccer
    """
    # d is draws
    # t is time
    # i,j is gender
    R_t_ij_base = np.einsum("dt,dij->dtij",R_t_base,C_base)
    infections_base = S_t/pop*np.einsum("dti,dtij->dti", new_I_t, R_t_ij_base)
    
    R_t_ij_soccer = np.einsum("dt,dij->dtij",R_t_soccer,C_soccer)
    infections_soccer = S_t/pop*np.einsum("dti,dtij->dtj", new_I_t, R_t_ij_soccer)

    # Sum over the choosen range (i.e. month of uefa championship) male and femal
    num_infections_base = np.sum(infections_base[...,i_begin:i_end,:], axis=-2)
    num_infections_soccer = np.sum(infections_soccer[...,i_begin:i_end,:], axis=-2)
    
    # Create pandas dataframe for easy violin plot
    ratio_soccer = num_infections_soccer/(num_infections_base+num_infections_soccer)
    male = np.stack((ratio_soccer[:,0],np.zeros(ratio_soccer[:,0].shape)),axis=1)
    female = np.stack((ratio_soccer[:,1],np.ones(ratio_soccer[:,1].shape)),axis=1)
    
    percentage = pd.DataFrame(np.concatenate((male,female)),columns=["percentage_soccer","gender"])
    percentage["gender"] = pd.cut(percentage["gender"], bins=[-1,0.5,1], labels=["male","female"])
    percentage["percentage_soccer"] = percentage["percentage_soccer"]*100
    percentage['dummy'] = 0
    
    g = sns.violinplot(
        data=percentage,
        y="percentage_soccer",
        x="dummy",
        hue="gender",
        scale="width",
        inner="quartile",
        orient="v",
        ax=ax,
        split=True,
        palette={"male": colors["male"], "female": colors["female"]},
        linewidth=1,
        saturation=1,
    )
    ax.legend([],[], frameon=False)
    ax.yaxis.set_label_position("right")
    ax.yaxis.tick_right()
    ax.collections[0].set_edgecolor(colors["male"]) # Set outline colors
    ax.collections[1].set_edgecolor(colors["female"]) # Set outline colors
    ax.set_xticklabels([])
    ax.set_ylim(ylim_relative)
    
    
    ax.set_ylabel("Percentage of soccer related\ninfections during the Championship")
    ax.set_xlabel("")
    ax.set_xticks([])
    ax.axhline(0,color="tab:gray",ls="--",zorder=-5)
    
    # Remove spines
    ax.spines['bottom'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    print(f"CI [50,2.5,97.5] {dl.countries}:")
    print(f"\tmale {np.percentile(ratio_soccer[:,0], [50,2.5,97.5])}")
    print(f"\tfemale {np.percentile(ratio_soccer[:,1], [50,2.5,97.5])}")

    return ax