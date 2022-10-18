# Using gender disparities to identify EURO2020 contributions to the COVID-19 spread

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Supplementary code for our **Using gender disparities to identify EURO2020 contributions to the COVID-19 spread** publication. **This is an ongoing project!** Read the latest draft of the publication [here](./Draft_covid19_soccer_211014.pdf) (updated on 14th October 2021).

## Usage

Clone with

```bash
git clone --recurse-submodules git@github.com:Priesemann-Group/covid19_soccer.git
```

We also advice you to create an conda environment for this project as we fixed the version of some
packages which could interfer with your previously installed packages.

```bash
conda create -n "covid19_soccer"  python=3.9
conda activate covid19_soccer
pip install -r requirements.txt
```

The easiest way to start with our projects is to have a look at the [getting started](./notebooks/getting_started.ipynb) notebook. Please also read our publication!

To fully reproduce our results you may want to download the data too. The data is available [here](https://gin.g-node.org/semohr/covid19_soccer_data). Alternatively you can also run the analysis on your own this may take some time tho. We do not supply the data for the robustness checks, if you are looking for something specifc feel free to create an issue and ask.

## Orientation

Lost? Don't fear, we got your back. If you want to find code to reproduce a specific figure or table from our publication check the Table below:

Most of the figures have one distinct notebooks, the table below gives you a way to find them:

| Figure  | Title                                                                                                                                                  | Notebook                                                                                             |
| ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | ---------------------------------------------------------------------------------------------------- |
| 1       | Quantifying the impact of the Euro 2020 on COVID-19 spread                                                                                             | [overview_figure](./notebooks/overview_figure.ipynb)                                                 |
| 2       | Example cases illustrate that the spread associated with the Euro 2020 can encompass a substantial fraction of the observed cases                      | [overview_figure](./notebooks/overview_figure.ipynb)                                                 |
| 3       | What variables can predict the extent of the impact of Euro 2020 matches                                                                               | [correlation_cleaner](./notebooks/supplementary/correlation_plots_cleaner.ipynb)                     |
| S3      | Overview of the sum of primary and subsequent cases accountable to the Euro 2020                                                                       | [primary_and_subsequent_fraction](./notebooks/supplementary/primary_and_subsequent_fraction.ipynb)   |
| S4      | Overview of cases in all considered countries apart from the Netherlands                                                                               | [primary_and_subsequent_overview](./notebooks/supplementary/primary_and_subsequent_overview.ipynb)   |
| S5      | We found no significant correlation between cases arising from the Euro 2020 and human mobility                                                        | [correlation_cleaner](./notebooks/supplementary/correlation_plots_cleaner.ipynb)                     |
| S6      | We found no significant correlation between cases arising from the Euro 2020 and the stringency of governmental interventions (NPIs)                   | [correlation_cleaner](./notebooks/supplementary/correlation_plots_cleaner.ipynb)                     |
| S7      | We found slight trends in the correlations between the impact of Euro 2020 and the base reproduction number and country popularity                     | [correlation_cleaner](./notebooks/supplementary/correlation_plots_cleaner.ipynb)                     |
| S8      | Prediction of the impact of Euro 2020 matches without the two most significant countries in the main model (England and Scotland)                      | [correlation_cleaner](./notebooks/supplementary/correlation_plots_cleaner.ipynb)                     |
| S9      | Effect of single Euro 2020 matches on the spread of COVID-19 across competing countries. Whiskers denote 68% and 95% CI.                               | [matches_forest](./notebooks/supplementary/matches_forest.ipynb)                                     |
| S10     | Including in our model the potential local transmission around the stadium where the matches occur does not significantly increase the overall effect. | [beta](./notebooks/supplementary/beta.ipynb)                                                         |
| S11     | A temporal offset of 14 days leads to no inferred effect.                                                                                              | [offset_14_days](./notebooks/supplementary/robustness/offset_14_days.ipynb)                          |
| S12     | Changing the days of the match by a large offset results in a non-significant Effect                                                                   | [delay_and_large_offsets](./notebooks/supplementary/robustness/delays_and_large_offsets.ipynb)       |
| S13     | Robustness test for the effect of the temporal association between matches<br>and cases by varying the effective delay                                 | [delay_and_offsets](./notebooks/supplementary/robustness/delays_and_offsets.ipynb)                   |
| S14     | Robustness test for the effect of the width of the delay kernel.                                                                                       | [delay_width](./notebooks/supplementary/robustness/delay_width.ipynb)                                |
| S15     | Robustness test for the effect of the allowed base reproduction number variability.                                                                    | [change_point_intervals](./notebooks/supplementary/robustness/change_point_interval.ipynb)           |
| S16     | Robustness test for the effect of the fraction of female participation in football related gatherings                                                  | [omega_gender](./notebooks/supplementary/robustness/omega_gender.ipynb)                              |
| S17     | Robustness test for the effect the generation interval.                                                                                                | MISSING                                                                                              |
| S18     | Robustness test for the remaining priors not studied in the previous figures                                                                           | MISSING                                                                                              |
| S19     | The combination of the case numbers of England and Scotland leads to similar results                                                                   | [overview_figure_Eng_Sct_combined](./notebooks/supplementary/overview_figure_Eng_Sct_combined.ipynb) |
| S20     | Our model is able to identify the delay between infection and reporting of it                                                                          | [delay_and_offsets](./notebooks/supplementary/robustness/delays_and_offsets.ipynb)                   |
| S21     | Relative popularity of the search term “football” in England and Scotland                                                                              | [google_search](./notebooks/supplementary/google_search.ipynb)                                       |
| S22     | Male-female imbalance over time shows the largest deviations during championship.                                                                      | [imbalance_analysis](./notebooks/supplementary/imbalance_analysis.ipynb)                             |
| S23     | The inferred noise terms don’t depend strongly on the length of the analyzed Time-period                                                               | [compare_long_vs_short](./notebooks/supplementary/compare_long_vs_short.ipynb)                       |
| S24     | The inferred effect size (percentage of football-related primary infections) don’t depend strongly on the length of the analyzed time-period           | [compare_long_vs_short](./notebooks/supplementary/compare_long_vs_short.ipynb)                       |
| S25-S36 | Overview of the posterior for selected countries                                                                                                       | [extended_overview](./notebooks/supplementary/extended_overview.ipynb)                               |
| S37-S47 | Chain mixing of selected parameters for selected countries                                                                                             | [chain_mixing](./notebooks/supplementary/chain_mixing.ipynb)                                         |

## Notes:

You need `python>=3.9`.

Before you can use the code and rerun the analyses you have to:

- init the submodules:

  ```bash
  #Init
  git submodule init
  # Update package manual (inside covid19_inference folder)
  cd covid19_inference
  git pull origin master
  ```

- You may want to download or update the raw data
  [(see here)](./data/README.md)
