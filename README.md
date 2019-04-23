# TWIM_CCS_Calibration

Algorithm development: Sugyan Dixit (UofM)
Theory development: Keith Richardson (Waters), David Langridge (Waters), Keith Richardson (Waters), Sugyan Dixit (UofM), and Brandon T. Ruotolo (UofM)
The current code is still in developmental mode. Please be cautious in using this. All methods except for power_law are still in progress.
Provided is a set of python scripts for TWIM CCS calibration using various methods.
calibration_functions.py contains the equations used for calibration methods. The functions in this script is called by all the other scripts.

#########################################

Following is a description, and the order, for scripts to use for calibration purpose:

1. twim_cc_gen_assemble.py - Assembles the input for calibration and creates a .assemble file.

Input file example is located in example_data\input_wv_300.0_wh_40.0.csv. Input file contains fields marked with # on the first column that are required for calculation.
Please use this as a template to make input file of your own. You can change variables such as wave height, wave velocity, pressure, temperature, EDC, twim_length, pot_factor, and wave_lambda.
twim_length, pot_factor, and wave_lambda are currently set for G2 TWIM system.

The script uses a database file to match entries in input file. Database files are located in folder CCDatabase. It contains two database files for postivie and negative CCS.
Input files entries need to match the entries in database file for id, oligomer, and charge ! Also make sure input entries have a corresponding CCS.
You can edit the database file as you need.

--help

-i INPUTF, --inputf ----> csv input file that contains the calibration input. See example_data\input_wv_300.0_wh_40.0.csv file for more info
-c CCS_DB, --ccs_db ----> ccs database file path. See CCSDatabase\ccsdatabse_positive.csv for more info
-g GAS_TYPE, --gas_type ----> gas type for calibration. Enter either n2 or he (for nitrogen or helium CCS values)


2. twim_ccs_gen_calibration.py - Uses the .assemble file to create calibration using different methods.

Six different calibration methods currently exists. The method that is used in the TWIM community is power_law.
The rest of the methods are currently under development. Please be cautious of using those.
After the calibration is done, it creates three output files
a) example_data\input_wv_300.0_wh_40.0_power_law_True.cal ----> Calibration file. This will be used for predictions
b) example_data\input_wv_300.0_wh_40.0_power_law_True_cal_output.csv ----> It contains the experimental parameters used for calibration. Also includes the xdata and ydata used to construct calibration. Includes the optimization paramters used in calibration. Finally, it reports the statistics on how good the calibration is with linear regression output and rmse in velocity and ccs.
c) example_data\input_wv_300.0_wh_40.0_power_law_True_calibrants_output.csv ----> Contains the input calibrants used and calibration output. Shows the predicted ccs and mobility and also shows the deviation on those values based on database values.


--help

-i INPUTF, --inputf ----> assemble file input type .assemble. See example_data\input_wv_300.0_wh_40.0.assemble file for example
-c CAL_TYPE, --cal_type ----> calibration method. Options are: power_law, power_law_exp, relax_true_6, relax_true_6_exp, blended, and blended_exp


3. twim_ccs_gen_prediction.py - Uses the calibration .cal file to predict ccs for unknowns.

Unknown file is a .csv file with id, oligomer, mass, charge, and dt fields. See example_data\unk_wv_300.0_wh_40.0.csv

It outputs a prediction file (see example_data\unk_wv_300.0_wh_40.0_power_law_True_pred.csv) that has all the experimental paramters used, CCS RMSE, and the predicted ccs and mobility

-- help

-i INPUTF, --inputf ----> calibration .cal file
-u UNKF, --unkf ----> unknown file type .csv


######################################


There are examples of manual data input scheme (without commandline arguments) in each code for users who want to automate calculations on large datasets.

