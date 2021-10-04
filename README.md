# Soccer related COVID-19 spreading

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Supplementary code for our **Using gender disparities to measure the EURO~2020 match-induced effect on COVID-19 cases in selected European countries** publication. **This is an ongoing project!** Read the latest draft of the publication [here](./DRAFT_04_10_21.pdf) (updated 4.10.21).

## Usage
Clone with 
```bash
git clone --recurse-submodules git@github.com:Priesemann-Group/covid19_soccer.git
```

The easiest way to start with our projects codebase is by looking into our model generator script. This can be found in `covid19_soccer\models.py`.

## Dev notes
You need `python>3.8`.

Submodules:
```bash
#Init
git submodule init
# Update package manual (inside covid19_inference folder)
git pull origin master
```
