# Automating Value-Oriented Forecast Model Selection by Meta-learning: Application on a Dispatchable Feeder

This repository contains code and data to replicate the results in "Automating Value-Oriented Forecast Model Selection by Meta-learning: Application on a Dispatchable Feeder".

Werling, D., Beichter, M., Heidrich, B., Phipps, K., Mikut, R., Hagenmeyer, V. (2024). 
Automating Value-Oriented Forecast Model Selection by Meta-learning: Application on a Dispatchable Feeder. 
In: Jørgensen, B.N., da Silva, L.C.P., Ma, Z. (eds) Energy Informatics. EI.A 2023. Lecture Notes in Computer Science, vol 14467. Springer, Cham. https://doi.org/10.1007/978-3-031-48649-4_6

## Acknowledgements and Funding

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, the Helmholtz Association under the Program “Energy System Design”, 
and the German Research Foundation (DFG) as part of the Research Training Group 2153 “Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation”.


## Environment

Use the given requirements.txt to create a python environment with the python version 3.9.7.

## Quick Start:

To start the experiment for a specific building (bldg_id) and a specific dataset with potentially manipulated loads (load_factor), you need to call

```python
python pipeline.py --id bldg_id --factor load_factor
```

Note that you have to create the dataset with the corresponding load_factor in advance.

## Data
 The [solar home electricity dataset](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) was used for the paper. This consists of the years 2011 - 2013, which are divided into individual files. To run the forecasting pipeline on it, the data must be downloaded and prepared so that all 3 years are concatenated and within the columns the prosumption data of a single house are located. Additionally the factors have to be applied.
Further the data has to be placed at "data/solar_home_all_data_2010-2013{args.factor}.csv" , where args.factor is for example "_ldiv2".


## License
