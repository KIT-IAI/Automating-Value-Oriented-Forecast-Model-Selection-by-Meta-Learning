# Automating Value-Oriented Forecast Model Selection by Meta-learning: Application on a Dispatchable Feeder

This repository contains code and data to replicate the results in "Automating Value-Oriented Forecast Model Selection by Meta-learning: Application on a Dispatchable Feeder". 

Werling, D., Beichter, M., Heidrich, B., Phipps, K., Mikut, R., Hagenmeyer, V. (2024). 
Automating Value-Oriented Forecast Model Selection by Meta-learning: Application on a Dispatchable Feeder. 
In: Jørgensen, B.N., da Silva, L.C.P., Ma, Z. (eds) Energy Informatics. EI.A 2023. Lecture Notes in Computer Science, vol 14467. Springer, Cham. https://doi.org/10.1007/978-3-031-48649-4_6

## Acknowledgements and Funding

This project is funded by the Helmholtz Association’s Initiative and Networking Fund through Helmholtz AI, the Helmholtz Association under the Program “Energy System Design”, 
and the German Research Foundation (DFG) as part of the Research Training Group 2153 “Energy Status Data: Informatics Methods for its Collection, Analysis and Exploitation”.


## Environment

Use the given requirements_classification.txt to create a python environment with the python version 3.9.7.


## Data
The [solar home electricity dataset](https://www.ausgrid.com.au/Industry/Our-Research/Data-to-share/Solar-home-electricity-data) was used for the paper. This dataset consists of 3 data files for the years 2010 - 2013. To extend the dataset, additional scenarios were generated using load scaling factors
$\beta_{\text{load}}$ and PV power generation scaling factors $\beta_{\text{PV}}$. The considered parameter combinations $(\beta_{\text{load}}, \beta_{\text{PV}})$ were:
(2.5, 0.5), (1, 0.5), (0.25, 0.5), (0.1, 0.5), (0.5, 0.5), (0.5, 2.5), and (0.5, 5).
This data is fed into the forecasting and optimisation pipeline in https://github.com/KIT-IAI/Impact-of-Forecast-Characteristics-on-Forecast-Value-for-Dispatchable-Feeder, which generates different forecasts and calculates the results of the optimisation problem. 
Based on the results of the optimisation problem, the target variables for the classification can be calculated. The results have to be placed at "data/data_analysis/target_imb2.csv".
To generate the input features for the classification, the Ausgrid dataset must be prepared so that all 3 years are concatenated and the columns contain the data of a single house. Further, the data has to be placed at "data/data/solar_gg_all_2010-2013.csv", "data/data/solar_gc_all_2010-2013.csv", and "data/data/solar_cl_all_2010-2013.csv". 


## License
This code is licensed under the [MIT License](LICENSE).
