import os
import optparse
import pickle
import pandas as pd
from math import pi
import numpy as np


def save_object_to_pickle(obj, file_path):
    with open(file_path, 'wb') as out_obj:
        pickle.dump(obj, out_obj)
        out_obj.close()

def ccs_nm2_to_m2(ccs):
    return ccs * 1e-18


def mass_da_to_kg(mass):
    """
    convert mass in daltons to kg
    :param mass: mass in da
    :return: mass in kg
    """
    return mass*1.66054e-27

def reduced_mass(mass1, mass2):
    """
    compute reduce mass of two bodies
    :param mass1:
    :param mass2:
    :return: reduce mass
    """
    y = (mass1*mass2)/(mass1 + mass2)
    return y

def correct_drift_time(time, mass_to_charge, edc_constant):
    """
    convert drift time to seconds, subtract the mass dependent flight time
    :param time: drift time in ms
    :param mass_to_charge: mass / charge_state
    :param edc_constant: instrument edc constant
    :return: corrected drift time accounting for time spent outside IM cell
    """
    y = np.subtract(time, np.divide(np.multiply(edc_constant, np.sqrt(mass_to_charge)), 1000))
    # apply a constant of -0.61 ms to correct for transfer time
    y_ = np.subtract(y, 0.61)
    return y_


def calculate_average_velocity(time, length):
    """
    calculate average velocity of the ion
    :param time: corrected drift time
    :param length: length of IM cell
    :return: average velocity in m/s
    """
    y = np.divide(length, np.multiply(time, 1e-3))
    # y = length/(time*1e-3)
    return y


def correct_ccs(ccs, charge, reduce_mass):
    """
    calculate the corrected ccs based on charge and reduce mass
    :param ccs: ccs
    :param charge: charge state
    :param reduce_mass: reduce mass
    :return: corrected ccs
    """
    y = ccs/(charge * np.sqrt(1/reduce_mass))
    return y


def pressure_bar_to_pascals(pressure):
    return pressure * 1e5


def calculate_number_density(pressure_pascal, temperature_kelvin):
    gas_constant = 8.314
    num_density = (pressure_pascal * 6.022e23) / (gas_constant * temperature_kelvin)
    return num_density

def calculate_mobility(mass_da, charge_state, ccs_nm2, temp_k, pressure_bar, mass_gas_da):
    pressure_pasc = pressure_bar_to_pascals(pressure_bar)
    num_den = calculate_number_density(pressure_pasc, temp_k)
    mass_analyte_kg = mass_da_to_kg(mass_da)
    mass_gas_kg = mass_da_to_kg(mass_gas_da)
    ccs_m2 = ccs_nm2_to_m2(ccs_nm2)
    y = (3 / 16) * np.sqrt(( 2 * pi * (mass_analyte_kg + mass_gas_kg)) /
                           (mass_analyte_kg * mass_gas_kg * 1.38e-23 * temp_k)) * ((charge_state * 1.6e-19)
                                                                                   / (num_den * ccs_m2))
    return y



def calculate_ccs(mass_da, charge_state, mobility, temp_k, pressure_bar, mass_gas_da):
    pressure_pasc = pressure_bar_to_pascals(pressure_bar)
    num_den = calculate_number_density(pressure_pasc, temp_k)
    mass_analyte_kg = mass_da_to_kg(mass_da)
    mass_gas_kg = mass_da_to_kg(mass_gas_da)
    y = (3 / 16) * np.sqrt((2 * pi * (mass_analyte_kg + mass_gas_kg)) /
                           (mass_analyte_kg * mass_gas_kg * 1.38e-23 * temp_k)) * ((charge_state * 1.6e-19)
                                                                                   / (num_den * mobility))
    # convert m2 to nm2
    y = np.multiply(y, 1e18)
    return y



def read_cal_input_file(fpath):
    list_file = []
    with open(fpath, 'r') as cal_input:
        cal_input_read = cal_input.read().splitlines()
        for line in cal_input_read:
            if not line.startswith('#'):
                list_file.append(line)
    return list_file


def read_ccs_database(ccsdatabase_file):
    """
    read ccs_database_file and stores information in a class
    :param ccsdatabase_file:
    :return: ccs_database_object
    """
    ccs_db_df = pd.read_csv(ccsdatabase_file)
    return ccs_db_df


class AssembleCalibrants(object):
    """
    store the input needed for calibration
    """

    def __init__(self, cal_input_fpath, ccs_db_df, gas_type):
        self.cal_input_csv = cal_input_fpath
        self.ccs_db_df = ccs_db_df
        self.gas_type = gas_type
        self.temp = None
        self.twim_length = None
        self.assemble_cal_object_path = None
        self.gas_mass = None
        self.wave_height = None
        self.wave_velocity = None
        self.wave_volt_pot_fac = None
        self.wave_lambda = None
        self.pressure = None
        self.species = None
        self.oligomer = None
        self.mass = None
        self.charge = None
        self.superclass = None
        self.subclass = None
        self.mass_over_charge = None
        self.reduce_mass = None
        self.edc_constant = None
        self.drift_time = None
        self.corrected_drift_time = None
        self.exp_avg_velocity = None
        self.ccs = None
        self.corrected_ccs = None
        self.mobility = None


    def gen_cal_input(self):
        wave_height = []
        wave_velocity = []
        pressure = []
        temperature = []
        tw_length = []
        edc = []
        pot_fact = []
        wave_lambd = []
        species_id = []
        oligomer = []
        mass = []
        charge = []
        drift_time = []
        ccs_pub = []
        with open(self.cal_input_csv, 'r') as calf:
            calf_read = calf.read().splitlines()
            for line in calf_read:
                if line.startswith('#'):
                    if line.find('Waveht') == 1:
                        wave_height.append(float(line.split(',')[1]))
                    if line.find('Wavevel') == 1:
                        wave_velocity.append(float(line.split(',')[1]))
                    if line.find('Pressure') == 1:
                        pressure.append(float(line.split(',')[1]))
                    if line.find('EDC') == 1:
                        edc.append(float(line.split(',')[1]))
                    if line.find('Pot_factor') == 1:
                        pot_fact.append(float(line.split(',')[1]))
                    if line.find('Wave_lambda') == 1:
                        wave_lambd.append(float(line.split(',')[1]))
                    if line.find('Temperature') == 1:
                        temperature.append(float(line.split(',')[1]))
                    if line.find('TWIM_length') == 1:
                        tw_length.append(float(line.split(',')[1]))

        # store information from input file
        data_frame = pd.read_csv(self.cal_input_csv, header=8)
        self.species = data_frame['#id'].values
        self.oligomer = data_frame['oligomer'].values
        self.mass = data_frame['mass'].values
        self.charge = data_frame['charge'].values
        self.drift_time = data_frame['dt'].values

        if 'superclass' in data_frame:
            self.superclass = data_frame['superclass'].values
        if 'subclass' in data_frame:
            self.subclass = data_frame['subclass'].values

        #match ccs value and entry based on the gas type, species, oligomer, and charge
        gas_mass = 0
        ccs_db_label = ''
        if self.gas_type == 'n2':
            gas_mass += 28
            ccs_db_label = 'ccs_n2'
        if self.gas_type == 'he':
            gas_mass += 4
            ccs_db_label = 'ccs_he'
        if self.gas_type == None:
            print('Error: gas_type = None | specify gas type (he or n2)')

        for index, (species_, oligo_, charge_) in enumerate(
                zip(data_frame['#id'].values, data_frame['oligomer'].values, data_frame['charge'].values)):
            db_df_ccs = self.ccs_db_df[
                (self.ccs_db_df['id'] == species_) & (self.ccs_db_df['n_oligomers'] == oligo_) & (
                            self.ccs_db_df['z'] == charge_)][ccs_db_label].values[0]
            ccs_pub.append(db_df_ccs)

        self.wave_height = wave_height[0]
        self.wave_velocity = wave_velocity[0]
        self.wave_volt_pot_fac = pot_fact[0]
        self.wave_lambda = wave_lambd[0]
        self.pressure = pressure[0]
        self.temp = temperature[0]
        self.twim_length = tw_length[0]
        self.edc_constant = edc[0]
        self.ccs = np.array(ccs_pub)
        self.gas_mass = gas_mass

        self.mass_over_charge = self.mass / self.charge
        self.reduce_mass = reduced_mass(self.mass, self.gas_mass)

        self.corrected_drift_time = correct_drift_time(self.drift_time, self.mass_over_charge, self.edc_constant)
        self.exp_avg_velocity = calculate_average_velocity(self.corrected_drift_time, self.twim_length)

        self.corrected_ccs = correct_ccs(self.ccs, self.charge, self.reduce_mass)

        self.mobility = calculate_mobility(self.mass, self.charge, self.ccs, self.temp, self.pressure/1000, self.gas_mass)

        dirpath, cal_input_csv_fname = os.path.split(self.cal_input_csv)
        assemble_obj_fname = str(cal_input_csv_fname).split('.csv')[0] + '.assemble'
        self.assemble_cal_object_path = os.path.join(dirpath, assemble_obj_fname)

        return self


def gen_assemble_cal_object(cal_input_file, ccs_db_file, gas_type='he'):
    ccs_db_df = read_ccs_database(ccs_db_file)
    assemble_cal_obj = AssembleCalibrants(cal_input_file, ccs_db_df, gas_type).gen_cal_input()
    save_object_to_pickle(assemble_cal_obj, assemble_cal_obj.assemble_cal_object_path)
    return assemble_cal_obj


def check_path_return_absolute_path(path):
    return_path = ''
    if os.path.isabs(path):
        return_path = path
    else:
        return_path = os.path.join(os.getcwd(), path)
    return return_path



def gen_assemble_cal_object_from_parser(parser):
    (options, args) = parser.parse_args()
    inputf = options.inputf
    ccs_db = options.ccs_db
    gas_type = options.gas_type
    cal_input_fpath = check_path_return_absolute_path(inputf)
    ccs_db_fpath = check_path_return_absolute_path(ccs_db)
    assemble_cal_obj = gen_assemble_cal_object(cal_input_fpath, ccs_db_fpath, gas_type=gas_type)



def parser_commands():
    parser = optparse.OptionParser(description='Creates an assemble file ready to generate calibration scheme')
    parser.add_option('-i' ,'--inputf', dest='inputf', default='example_data\input_wv_300.0_wh_40.0.csv',
                      help='csv input file that contains the calibration input. See example_data\input_wv_300.0_wh_40.0.csv file for more info')
    parser.add_option('-c', '--ccs_db', dest='ccs_db', default='CCSDatabase\ccsdatabse_positive.csv',
                      help='csv ccs database file. See ccsdatabase_positive.csv and ccsdatabse_negative.csv for examples')
    parser.add_option('-g', '--gas_type', dest='gas_type', default='n2', help='gas type for calibration. Either n2 or he')
    return parser


if __name__ == '__main__':
    # parser = parser_commands()
    # gen_assemble_cal_object_from_parser(parser)


    ## examples of manual data input below
    ###

    # using a single input file

    ccs_db_file = r"CCSDatabase\ccsdatabse_positive.csv"
    # cal_input_file_ = r"T:\Sugyan\test_varun\cal_poly3only\cal_input_wv_500.0_wh_35.0.csv"
    # gas_type = 'n2'
    # gen_assemble_cal_object(cal_input_file_, ccs_db_file, gas_type=gas_type)


    # using a list of input files listed in a csv file. Input files are in each row

    cal_input_file = r"C:\Users\sugyan\Documents\Processed data\051819_CalProcessing\All\cal_input_files.csv"
    list_file = read_cal_input_file(cal_input_file)
    for file in list_file:
        print(file)
        gen_assemble_cal_object(file, ccs_db_file, gas_type='n2')
