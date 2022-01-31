# Using gender disparities to identify EURO2020 contributions to the COVID-19 spread 

[![License: GPL v3](https://img.shields.io/badge/License-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

Supplementary code for our **Using gender disparities to identify EURO2020 contributions to the COVID-19 spread** publication. **This is an ongoing project!** Read the latest draft of the publication [here](./Draft_covid19_soccer_211014.pdf) (updated on 14th October 2021).

## Usage

The easiest way to start with our projects is to have a look at the [getting started](./notebooks/getting_started.ipynb) notebook. Please also read our publication!

Clone with 
```bash
git clone --recurse-submodules git@github.com:Priesemann-Group/covid19_soccer.git
```
## Notes:
You need `python>3.8`.

Before you can use the code and rerun the analyses you have to:

- init the submodules:
	```bash
	#Init
	git submodule init
	# Update package manual (inside covid19_inference folder)
	cd covid19_inference
	git pull origin master
	```

- install the requirements
	```bash
	pip install -r requirments.txt
	```

- You may want to download or update the raw data
	[(see here)](./data/README.md)
