# generate calibration with calibration assemble object input
# includes powerfit, powerfit exp, relax (4, 6), relax (4, 6) exp, no_relax (4, 6), no_relax (4, 6) exp -
# blend_fit, blend_fit exp
# gnerate calibration scheme as an object

import os
import sys, logging
import optparse
import pickle
import twim_ccs_gen_assemble
from twim_ccs_gen_assemble import AssembleCalibrants, CCSDatabase, save_object_to_pickle, read_cal_input_file, gen_assemble_cal_object, pressure_bar_to_pascals, calculate_mobility, calculate_number_density, mass_da_to_kg, correct_ccs, calculate_ccs, check_path_return_absolute_path
import calibration_functions
import numpy as np
from scipy.optimize import curve_fit, root, least_squares
from sklearn.linear_model import LinearRegression
from scipy.stats import linregress
import matplotlib.pyplot as plt
import time


def load_pickle_object(pickle_file_path):
    """
    opens the pickled object!
    Caution: You have to import the class(es) into current python script in order to load the object
    First import the python script: import python_script
    then import the class: from python_script import class(es)
    :param pickle_file_path: pickle object path
    :return: object
    """
    with open(pickle_file_path, 'rb') as pk_file:
        obj = pickle.load(pk_file)
    return obj


def generate_cal_mode_dict(power_law=None, relax=None, terms=None, blended=None, exp=None):
    """
    create cal mode dictionary
    :param mode: cal_mode
    :param exp: bool
    :return: dict
    """
    cal_mode_dict = dict()
    cal_mode_dict['power_law'] = power_law
    cal_mode_dict['relax'] = relax
    cal_mode_dict['terms'] = terms
    cal_mode_dict['blended'] = blended
    cal_mode_dict['exp'] = exp
    return cal_mode_dict


def gen_exp_cond_from_assemble_obj(assemble_obj):
    """
    store the exp condition from assemble object to cal scheme object
    :param assemble_obj: assemble object
    :return: dict
    """
    exp_cond_dict = dict()
    exp_cond_dict['wave_height'] = assemble_obj.wave_height
    exp_cond_dict['wave_velocity'] = assemble_obj.wave_velocity
    exp_cond_dict['gas_type'] = assemble_obj.gas_type
    exp_cond_dict['gas_mass'] = assemble_obj.gas_mass
    exp_cond_dict['pressure'] = assemble_obj.pressure
    exp_cond_dict['temperature'] = assemble_obj.temp
    exp_cond_dict['twim_length'] = assemble_obj.twim_length
    exp_cond_dict['edc_constant'] = assemble_obj.edc_constant
    exp_cond_dict['pot_factor'] = assemble_obj.wave_volt_pot_fac
    exp_cond_dict['wave_lambda'] = assemble_obj.wave_lambda
    return exp_cond_dict


def gen_calibrant_list_from_assemble_obj(assemble_obj):
    """
    store the moelcule id, oligomer number, mass, and charge information from assemble object into a dictionary
    :param assemble_obj: assemble object
    :return: dictionary
    """
    cal_list_dict = dict()
    cal_list_dict['species'] = assemble_obj.species
    cal_list_dict['oligomer'] = assemble_obj.oligomer
    cal_list_dict['mass'] = assemble_obj.mass
    cal_list_dict['charge'] = assemble_obj.charge
    return cal_list_dict



def create_cal_fit_dict(xdata, ydata, cal_fit_obj, cal_fit_obj_score, cal_fit_params, blended_fixa, cal_fit_pcov, linreg_true_pred,
                        pub_ccs, pub_corr_ccs, mobility, pred_vel, pred_ccs, pred_corr_ccs, pred_mobility, vel_deviation,
                        ccs_deviation, vel_rmse, ccs_rmse):
    """
    create cal fit dictionary
    :param xdata: xdata used for fitting
    :param ydata: ydata used for fitting
    :param cal_fit_obj: cal fitting object if available
    :param cal_fit_obj_score: cal fit object score if available
    :param cal_fit_params: cal fitting params if available
    :param cal_fit_pcov: cal fit pcov is available
    :param linreg_true_pred: linregression between true and predicted values
    :param pred_vel: predicted velocity
    :param pred_ccs: predicted ccs
    :param pred_corr_ccs: predicted corrected ccs
    :param vel_deviation: velocity deviation
    :param ccs_deviation: ccs deviation
    :param vel_rmse: rmse velocity
    :param ccs_rmse: rmse ccs

    :return: cal fit dict
    """
    cal_fit_dict = dict()
    cal_fit_dict['xdata'] = xdata
    cal_fit_dict['ydata'] = ydata
    cal_fit_dict['cal_fit_obj'] = cal_fit_obj
    cal_fit_dict['cal_fit_obj_score'] = cal_fit_obj_score
    cal_fit_dict['cal_fit_params'] = cal_fit_params
    cal_fit_dict['blended_fixa'] = blended_fixa
    cal_fit_dict['cal_fit_pcov'] = cal_fit_pcov
    cal_fit_dict['linreg_true_pred'] = linreg_true_pred
    cal_fit_dict['pub_ccs'] = pub_ccs
    cal_fit_dict['pub_corr_ccs'] = pub_corr_ccs
    cal_fit_dict['mobility'] = mobility
    cal_fit_dict['pred_vel'] = pred_vel
    cal_fit_dict['pred_ccs'] = pred_ccs
    cal_fit_dict['pred_corr_ccs'] = pred_corr_ccs
    cal_fit_dict['pred_mobility'] = pred_mobility
    cal_fit_dict['vel_deviation'] = vel_deviation
    cal_fit_dict['ccs_deviation'] = ccs_deviation
    cal_fit_dict['vel_rmse'] = vel_rmse
    cal_fit_dict['ccs_rmse'] = ccs_rmse


    return cal_fit_dict



def corr_ccs_to_ccs(corr_ccs, charge, reduced_mass):
    """
    Calculate CCS from corrected ccs
    :param corr_ccs: corrected ccs
    :param charge: charge
    :param reduced_mass: reduced mass
    :return: CCS (un corrected)
    """
    y = corr_ccs * (charge * np.sqrt(1/reduced_mass))
    return y


def percent_deviation_predict_true(predict_values, true_values):
    """
    calculate the % deviation between prediction and true values
    :param predict_values: predictions
    :param true_values: true values
    :return: % deviation
    """
    y = (predict_values - true_values)*100/true_values
    return y


def rms_error(predict_values, true_values):
    """
    Calculate the percent error and computes the rms error in %
    :param predict_values: predict values
    :param true_values: true values
    :return: % rmse
    """
    diff = np.subtract(predict_values, true_values)
    percent_error = np.multiply(np.divide(diff, true_values), 100)
    percent_error_sq = np.square(percent_error)
    rmse = np.sqrt(np.divide(np.sum(percent_error_sq), len(percent_error_sq)))
    return rmse


class CalibrationScheme(object):
    """
    calibration scheme object
    includes the mode of calibration, calibration fit object, calibration output
    """

    def __init__(self):
        """
        store the assemble object, mode of calibration, fit objects, cal outputs
        """
        self.cal_calibrants = None
        self.cal_exp_cond = None
        self.cal_object_fpath = None
        self.cal_mode = None # dictionary: 'power_fit, relax_2_4, relax_2_4_6, no_relax_2_4, no_relax_2_4_6, blend_fit (all with or without exp)
        self.cal_output = None # dictionary


def input_data_relax_no_relax_without_exp(ccs, massovercharge, relaxation=True, terms=6):
    """
    This generates the input data for calibration with or without relaxation, with terms specified.
    This does not deal with fitting with the exponent form of calibration
    :param ccs: corrected ccs
    :param massovercharge: mass over charge
    :param relaxation: bool. True or False. If True, includes the m/z terms
    :param terms: int. Specify how many terms to include for calibration
    :return: xdata input for linear regression model
    """
    ccs_rec = np.reciprocal(ccs)
    ccs_rec_power_data = []
    mass_over_charge_power_data = []
    if relaxation == True:
        ccs_power_range = np.arange(2, 9, 2) # 2, 4, 6, 8
        for ccs_power in ccs_power_range:
            ccs_rec_power = np.power(ccs_rec, ccs_power)
            ccs_rec_power_data.append(ccs_rec_power)
        mz_power_range = np.arange(2, 7, 2) # 2, 4, 6
        for mz_power in mz_power_range:
            mass_over_charge_power = np.power(massovercharge, mz_power)
            mass_over_charge_power_data.append(mass_over_charge_power)
        if terms == 4:
            xdata = np.array((ccs_rec_power_data[0], ccs_rec_power_data[1], np.multiply(ccs_rec_power_data[1],
                                                                                        mass_over_charge_power_data[0]),
                              ccs_rec_power_data[2])).T
        if terms == 6:
            xdata = np.array((ccs_rec_power_data[0], ccs_rec_power_data[1],
                              np.multiply(ccs_rec_power_data[1], mass_over_charge_power_data[0]),
                              ccs_rec_power_data[2],
                              np.multiply(ccs_rec_power_data[2], mass_over_charge_power_data[0]),
                              np.multiply(ccs_rec_power_data[2], mass_over_charge_power_data[1]))).T
        if terms == 10:
            xdata = np.array((ccs_rec_power_data[0], ccs_rec_power_data[1],
                              np.multiply(ccs_rec_power_data[1], mass_over_charge_power_data[0]),
                              ccs_rec_power_data[2],
                              np.multiply(ccs_rec_power_data[2], mass_over_charge_power_data[0]),
                              np.multiply(ccs_rec_power_data[2], mass_over_charge_power_data[1]),
                             ccs_rec_power_data[3], np.multiply(ccs_rec_power_data[3], mass_over_charge_power_data[0]),
                             np.multiply(ccs_rec_power_data[3], mass_over_charge_power_data[1]),
                             np.multiply(ccs_rec_power_data[3], mass_over_charge_power_data[2]))).T
    if relaxation == False:
        ccs_power_range = np.arange(2, 21, 2) # 2, 4, 6, 8, 10, 12, 14, 16, 18, 20
        for ccs_power in ccs_power_range:
            ccs_rec_power = np.power(ccs_rec, ccs_power)
            ccs_rec_power_data.append(ccs_rec_power)
        if terms == 4:
            xdata = np.array((ccs_rec_power_data[0], ccs_rec_power_data[1], ccs_rec_power_data[2],
                              ccs_rec_power_data[3])).T
        if terms == 6:
            xdata = np.array((ccs_rec_power_data[0], ccs_rec_power_data[1], ccs_rec_power_data[2],
                              ccs_rec_power_data[3], ccs_rec_power_data[4], ccs_rec_power_data[5])).T
        if terms == 10:
            xdata = np.array((ccs_rec_power_data[0], ccs_rec_power_data[1], ccs_rec_power_data[2], ccs_rec_power_data[3],
                              ccs_rec_power_data[4], ccs_rec_power_data[5], ccs_rec_power_data[6], ccs_rec_power_data[7],
                              ccs_rec_power_data[8], ccs_rec_power_data[9])).T

    return xdata


def pred_corr_ccs_with_root_finding_relax_no_relax_without_exp(linreg, corr_ccs, mass_over_charge, charge, reduce_mass,
                                                               exp_avg_vel, relax, terms):
    """
    determine predicted corr ccs using root finding solution
    :param linreg: lin reg object from LinearRegression Model (scikit learn)
    :param corr_ccs: corrected ccs
    :param mass_over_charge: mass over charge
    :param charge: charge
    :param reduce_mass: reduce mass
    :param exp_avg_vel: exp velocity
    :return: predicted corr ccs
    """
    pred_corr_ccs = []

    for ind, (ccs, m_z, vel, z, red_mass) in enumerate(zip(corr_ccs, mass_over_charge, exp_avg_vel, charge, reduce_mass)):
        init_guess = ccs
        if relax == True:
            input_args = tuple([m_z, vel] + [x for x in linreg.coef_])
            if terms == 4:
                sol = root(calibration_functions.relax_4_terms_root_finding_function, x0=init_guess, args=input_args)
                pred_corr_ccs.append(sol.x[0])
            if terms == 6:
                sol = root(calibration_functions.relax_6_terms_root_finding_function, x0=init_guess, args=input_args)
                pred_corr_ccs.append(sol.x[0])
            if terms == 10:
                sol = root(calibration_functions.relax_10_terms_root_finding_function, x0=init_guess, args=input_args)
                pred_corr_ccs.append(sol.x[0])
        if relax == False:
            input_args = tuple([vel] + [x for x in linreg.coef_])
            if terms == 4:
                sol = root(calibration_functions.mob_4_terms_root_finding_function, x0=init_guess, args=input_args)
                pred_corr_ccs.append(sol.x[0])
            if terms == 6:
                sol = root(calibration_functions.mob_6_terms_root_finding_function, x0=init_guess, args=input_args)
                pred_corr_ccs.append(sol.x[0])
            if terms == 10:
                sol = root(calibration_functions.mob_10_terms_root_finding_function, x0=init_guess, args=input_args)
                pred_corr_ccs.append(sol.x[0])

    pred_corr_ccs = np.array(pred_corr_ccs)
    return pred_corr_ccs



def pred_corr_ccs_with_root_finding_relax_with_exp_with_mass(popt, corr_ccs, mass_over_charge, charge, mass, exp_avg_vel, terms=6):
    """
    determine predicted corr ccs using root finding solution for relax with exp
    :param popt: popt from curve fitting
    :param corr_ccs: corrected ccs
    :param mass_over_charge: mass to charge
    :param charge: charge
    :param mass: mass
    :param exp_avg_vel: exp average velocity
    :param terms: number of terms
    :return: pred corr ccs
    """
    pred_corr_ccs = []


    for ind, (ccs, m_z, vel, z, mass_) in enumerate(zip(corr_ccs, mass_over_charge, exp_avg_vel, charge, mass)):
        init_guess = ccs
        input_args = tuple([m_z, z, mass_, vel] + [x for x in popt])
        if terms == 4:
            sol = root(calibration_functions.relax_4_terms_exp_mass_root_finding_function, x0=init_guess,
                       args=input_args)
            pred_corr_ccs.append(sol.x[0])
        if terms == 6:
            sol = root(calibration_functions.relax_6_terms_exp_mass_root_finding_function, x0=init_guess,
                       args=input_args)
            pred_corr_ccs.append(sol.x[0])
        if terms == 10:
            sol = root(calibration_functions.relax_10_terms_exp_mass_root_finding_function, x0=init_guess,
                       args=input_args)
            pred_corr_ccs.append(sol.x[0])

    pred_corr_ccs = np.array(pred_corr_ccs)
    return pred_corr_ccs


def pred_corr_ccs_with_root_finding_relax_with_exp(popt, corr_ccs, mass_over_charge, charge, exp_avg_vel, terms=6):
    """
    determine predicted corr ccs using root finding solution for relax with exp
    :param popt: popt from curve fitting
    :param corr_ccs: corrected ccs
    :param mass_over_charge: mass to charge
    :param charge: charge
    :param mass: mass
    :param exp_avg_vel: exp average velocity
    :param terms: number of terms
    :return: pred corr ccs
    """
    pred_corr_ccs = []


    for ind, (ccs, m_z, vel, z) in enumerate(zip(corr_ccs, mass_over_charge, exp_avg_vel, charge)):
        init_guess = ccs
        input_args = tuple([m_z, z, vel] + [x for x in popt])
        if terms == 4:
            sol = root(calibration_functions.relax_4_terms_exp_root_finding_function, x0=init_guess,
                       args=input_args)
            pred_corr_ccs.append(sol.x[0])
        if terms == 6:
            sol = root(calibration_functions.relax_6_terms_exp_root_finding_function, x0=init_guess,
                       args=input_args)
            pred_corr_ccs.append(sol.x[0])
        if terms == 10:
            sol = root(calibration_functions.relax_10_terms_exp_root_finding_function, x0=init_guess,
                       args=input_args)
            pred_corr_ccs.append(sol.x[0])

    pred_corr_ccs = np.array(pred_corr_ccs)
    return pred_corr_ccs



def pred_mob_with_root_finding_v_blend_func(mobility_arr, exp_vel_arr, mass_arr, charge_arr, par1, fit_params,
                                            exp=False, fixa=True):
    """
    use the root finding to find solution to determine mobility
    :param mobility: mobility array from experiment. This will be used for initial guess in root finding function
    :param pred_vel_arr: predicted velocity from least squares
    :param mass_arr: mass (da) array from assemble obj
    :param charge_arr: charge (z) array from assemble obj
    :param par1: wave_ht, wave_vel, wave_vel, wave_v_pot_fac, wave_lambda
    :param fit_params: fit params from least sq (least_squares.x)
    :return: pred mobility array
    """


    pred_mob = []
    for ind, (mobility, mass, charge, exp_vel) in enumerate(zip(mobility_arr, mass_arr, charge_arr, exp_vel_arr)):
        init_guess = mobility
        # input_args = (exp_vel, mass, charge, par1, fit_params, fixa)
        if not exp:
            if fixa:
                fit_params_ = fit_params[1]
            else:
                fit_params_ = fit_params
            input_args = (exp_vel, mass, charge, par1, fit_params_, fixa)
            sol = root(calibration_functions.v_blend_root_finding_func, x0=init_guess, args=input_args)
            pred_mob.append(sol.x[0])

        else:
            if fixa:
                fit_params_ = fit_params[1:]
            else:
                fit_params_ = fit_params
            input_args = (exp_vel, mass, charge, par1, fit_params_, fixa)
            sol = root(calibration_functions.v_blend_exp_root_finding_func, x0=init_guess, args=input_args)
            pred_mob.append(sol.x[0])
    pred_mob = np.array(pred_mob)
    return pred_mob


def generate_w_wo_relax_without_exp(assemble_object, relax=True, terms=6):
    """
    generate calibration with relax or no relax without exp
    :param assemble_object: assemble object from twim_ccs_calibration_assemble py file
    :param relax: bool
    :param terms: int
    :return: calibartion scheme
    """

    cal_mode_dict = generate_cal_mode_dict(relax=relax, terms=terms)
    cal_mode_string = generate_cal_mode_string(cal_mode_dict)
    print('CAL_MODE:', cal_mode_string)

    x_ = input_data_relax_no_relax_without_exp(assemble_object.corrected_ccs, assemble_object.mass_over_charge,
                                               relaxation=relax, terms=terms)
    y_ = assemble_object.exp_avg_velocity
    lin_reg_fit_obj = LinearRegression(fit_intercept=False, n_jobs=-1)
    lin_reg_fit_obj.fit(x_, y_)
    lin_reg_fit_score = lin_reg_fit_obj.score(x_, y_)
    pred_velocity = lin_reg_fit_obj.predict(x_)

    pred_corr_ccs = pred_corr_ccs_with_root_finding_relax_no_relax_without_exp(lin_reg_fit_obj, assemble_object.corrected_ccs,
                                                    assemble_object.mass_over_charge, assemble_object.charge,
                                                    assemble_object.reduce_mass, y_, relax=relax, terms=terms)

    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, assemble_object.charge, assemble_object.reduce_mass)

    # (mass_da, charge_state, ccs_nm2, temp_k, pressure_bar, mass_gas_da)
    pred_mobility = calculate_mobility(assemble_object.mass, assemble_object.charge, pred_ccs, assemble_object.temp,
                                       assemble_object.pressure / 1000, assemble_object.gas_mass)

    lin_reg_exp_pred_vel = linregress(y_, pred_velocity)
    vel_deviation = percent_deviation_predict_true(pred_velocity, y_)
    ccs_deviation = percent_deviation_predict_true(pred_ccs, assemble_object.ccs)
    ccs_rmse = rms_error(pred_ccs, assemble_object.ccs)
    vel_rmse = rms_error(pred_velocity, y_)

    cal_fit_dict = create_cal_fit_dict(xdata=x_,
                                       ydata=y_,
                                       cal_fit_obj=lin_reg_fit_obj,
                                       cal_fit_obj_score=lin_reg_fit_score,
                                       cal_fit_params=None,
                                       blended_fixa=None,
                                       cal_fit_pcov=None,
                                       linreg_true_pred=lin_reg_exp_pred_vel,
                                       pub_ccs=assemble_object.ccs,
                                       pub_corr_ccs=assemble_object.corrected_ccs,
                                       mobility=assemble_object.mobility,
                                       pred_vel=pred_velocity,
                                       pred_ccs=pred_ccs,
                                       pred_corr_ccs=pred_corr_ccs,
                                       pred_mobility=pred_mobility,
                                       vel_deviation=vel_deviation,
                                       ccs_deviation=ccs_deviation,
                                       vel_rmse=vel_rmse,
                                       ccs_rmse=ccs_rmse)

    cal_scheme = CalibrationScheme()
    cal_scheme.cal_mode = cal_mode_dict
    cal_scheme.cal_output = cal_fit_dict

    dirpath, cal_assemble_fname = os.path.split(assemble_object.assemble_cal_object_path)

    cal_scheme_fname = str(cal_assemble_fname).split('.assemble')[0] + '_' + cal_mode_string + '.cal'
    cal_scheme.cal_object_fpath = os.path.join(dirpath, cal_scheme_fname)

    exp_cond_dict = gen_exp_cond_from_assemble_obj(assemble_object)
    cal_scheme.cal_exp_cond = exp_cond_dict

    cal_list_dict = gen_calibrant_list_from_assemble_obj(assemble_object)
    cal_scheme.cal_calibrants = cal_list_dict


    return cal_scheme



def generate_relax_with_exp_with_mass(assemble_object, terms=6):
    """
    generate calibration with relax with exp terms
    :param assemble_object: assemble object from twim_ccs_calibration_assemble py file
    :param terms: int
    :return: cal_scheme
    """
    cal_mode_dict = generate_cal_mode_dict(relax=True, terms=terms, exp=True)
    cal_mode_string = generate_cal_mode_string(cal_mode_dict)
    print('CAL_MODE:', cal_mode_string)

    x_ = (assemble_object.corrected_ccs, assemble_object.mass_over_charge, assemble_object.charge, assemble_object.mass)
    y_ = assemble_object.exp_avg_velocity

    popt, pcov, pred_velocity = pred_velocity_relax_with_exp_with_mass(x_, y_, terms=terms)

    pred_corr_ccs = pred_corr_ccs_with_root_finding_relax_with_exp_with_mass(popt, assemble_object.corrected_ccs,
                                                                   assemble_object.mass_over_charge,
                                                                   assemble_object.charge, assemble_object.mass, y_,
                                                                   terms=terms)
    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, assemble_object.charge, assemble_object.reduce_mass)
    pred_mobility = calculate_mobility(assemble_object.mass, assemble_object.charge, pred_ccs, assemble_object.temp,
                                       assemble_object.pressure / 1000, assemble_object.gas_mass)

    lin_reg_exp_pred_vel = linregress(y_, pred_velocity)
    vel_deviation = percent_deviation_predict_true(pred_velocity, y_)
    ccs_deviation = percent_deviation_predict_true(pred_ccs, assemble_object.ccs)
    ccs_rmse = rms_error(pred_ccs, assemble_object.ccs)
    vel_rmse = rms_error(pred_velocity, y_)

    cal_fit_dict = create_cal_fit_dict(xdata=x_,
                                       ydata=y_,
                                       cal_fit_obj=None,
                                       cal_fit_obj_score=None,
                                       cal_fit_params=popt,
                                       blended_fixa=None,
                                       cal_fit_pcov=pcov,
                                       linreg_true_pred=lin_reg_exp_pred_vel,
                                       pub_ccs=assemble_object.ccs,
                                       pred_vel=pred_velocity,
                                       pred_ccs=pred_ccs,
                                       pub_corr_ccs=assemble_object.corrected_ccs,
                                       mobility=assemble_object.mobility,
                                       pred_corr_ccs=pred_corr_ccs,
                                       pred_mobility=pred_mobility,
                                       vel_deviation=vel_deviation,
                                       ccs_deviation=ccs_deviation,
                                       vel_rmse=vel_rmse,
                                       ccs_rmse=ccs_rmse)

    cal_scheme = CalibrationScheme()
    cal_scheme.cal_mode = cal_mode_dict
    cal_scheme.cal_output = cal_fit_dict

    dirpath, cal_assemble_fname = os.path.split(assemble_object.assemble_cal_object_path)

    cal_scheme_fname = str(cal_assemble_fname).split('.assemble')[0] + '_' + cal_mode_string + '.cal'
    cal_scheme.cal_object_fpath = os.path.join(dirpath, cal_scheme_fname)

    exp_cond_dict = gen_exp_cond_from_assemble_obj(assemble_object)
    cal_scheme.cal_exp_cond = exp_cond_dict

    cal_list_dict = gen_calibrant_list_from_assemble_obj(assemble_object)
    cal_scheme.cal_calibrants = cal_list_dict

    return cal_scheme


def generate_relax_with_exp(assemble_object, terms=6):
    """
    generate calibration with relax with exp terms
    :param assemble_object: assemble object from twim_ccs_calibration_assemble py file
    :param terms: int
    :return: cal_scheme
    """
    cal_mode_dict = generate_cal_mode_dict(relax=True, terms=terms, exp=True)
    cal_mode_string = generate_cal_mode_string(cal_mode_dict)
    print('CAL_MODE:', cal_mode_string)

    x_ = (assemble_object.corrected_ccs, assemble_object.mass_over_charge, assemble_object.charge)
    y_ = assemble_object.exp_avg_velocity

    popt, pcov, pred_velocity = pred_velocity_relax_with_exp(x_, y_, terms=terms)

    pred_corr_ccs = pred_corr_ccs_with_root_finding_relax_with_exp(popt, assemble_object.corrected_ccs,
                                                                   assemble_object.mass_over_charge,
                                                                   assemble_object.charge, y_,
                                                                   terms=terms)
    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, assemble_object.charge, assemble_object.reduce_mass)
    pred_mobility = calculate_mobility(assemble_object.mass, assemble_object.charge, pred_ccs, assemble_object.temp,
                                       assemble_object.pressure / 1000, assemble_object.gas_mass)

    lin_reg_exp_pred_vel = linregress(y_, pred_velocity)
    vel_deviation = percent_deviation_predict_true(pred_velocity, y_)
    ccs_deviation = percent_deviation_predict_true(pred_ccs, assemble_object.ccs)
    ccs_rmse = rms_error(pred_ccs, assemble_object.ccs)
    vel_rmse = rms_error(pred_velocity, y_)

    cal_fit_dict = create_cal_fit_dict(xdata=x_,
                                       ydata=y_,
                                       cal_fit_obj=None,
                                       cal_fit_obj_score=None,
                                       cal_fit_params=popt,
                                       blended_fixa=None,
                                       cal_fit_pcov=pcov,
                                       linreg_true_pred=lin_reg_exp_pred_vel,
                                       pub_ccs=assemble_object.ccs,
                                       pred_vel=pred_velocity,
                                       pred_ccs=pred_ccs,
                                       pub_corr_ccs=assemble_object.corrected_ccs,
                                       mobility=assemble_object.mobility,
                                       pred_corr_ccs=pred_corr_ccs,
                                       pred_mobility=pred_mobility,
                                       vel_deviation=vel_deviation,
                                       ccs_deviation=ccs_deviation,
                                       vel_rmse=vel_rmse,
                                       ccs_rmse=ccs_rmse)

    cal_scheme = CalibrationScheme()
    cal_scheme.cal_mode = cal_mode_dict
    cal_scheme.cal_output = cal_fit_dict

    dirpath, cal_assemble_fname = os.path.split(assemble_object.assemble_cal_object_path)

    cal_scheme_fname = str(cal_assemble_fname).split('.assemble')[0] + '_' + cal_mode_string + '.cal'
    cal_scheme.cal_object_fpath = os.path.join(dirpath, cal_scheme_fname)

    exp_cond_dict = gen_exp_cond_from_assemble_obj(assemble_object)
    cal_scheme.cal_exp_cond = exp_cond_dict

    cal_list_dict = gen_calibrant_list_from_assemble_obj(assemble_object)
    cal_scheme.cal_calibrants = cal_list_dict

    return cal_scheme



def pred_velocity_relax_with_exp_with_mass(x_, y_, terms):
    popt, pcov, pred_vel = None, None, None
    if terms == 4:
        popt, pcov = curve_fit(calibration_functions.relax_4_terms_exp_mass, x_, y_,
                               maxfev=100000, gtol=1e-20, method='lm')
        pred_vel = calibration_functions.relax_4_terms_exp_mass(x_, *popt)
    if terms == 6:
        popt, pcov = curve_fit(calibration_functions.relax_6_terms_exp_mass, x_, y_, maxfev=100000, gtol=1e-20,
                               method='lm')
        pred_vel = calibration_functions.relax_6_terms_exp_mass(x_, *popt)
    if terms == 10:
        popt, pcov = curve_fit(calibration_functions.relax_10_terms_exp_mass, x_, y_, maxfev=100000, gtol=1e-20,
                               method='lm')
        pred_vel = calibration_functions.relax_10_terms_exp_mass(x_, *popt)
    return popt, pcov, pred_vel


def pred_velocity_relax_with_exp(x_, y_, terms):
    popt, pcov, pred_vel = None, None, None
    if terms == 4:
        popt, pcov = curve_fit(calibration_functions.relax_4_terms_exp, x_, y_,
                               maxfev=100000, gtol=1e-20, method='lm')
        pred_vel = calibration_functions.relax_4_terms_exp(x_, *popt)
    if terms == 6:
        popt, pcov = curve_fit(calibration_functions.relax_6_terms_exp, x_, y_, maxfev=100000, gtol=1e-20,
                               method='lm')
        pred_vel = calibration_functions.relax_6_terms_exp(x_, *popt)
    if terms == 10:
        popt, pcov = curve_fit(calibration_functions.relax_10_terms_exp, x_, y_, maxfev=100000, gtol=1e-20,
                               method='lm')
        pred_vel = calibration_functions.relax_10_terms_exp(x_, *popt)
    return popt, pcov, pred_vel


def generate_power_fit_cal_exp_mass(assemble_object, exp=False):
    """
    generate power fit calibration with or without exp
    :param assemble_object: assemble object from twim_ccs_calibration_assemble py file
    :param exp: bool
    :return: calibration scheme
    """

    cal_mode_dict = generate_cal_mode_dict(power_law=True, exp=exp)
    cal_mode_string = generate_cal_mode_string(cal_mode_dict)
    print('CAL_MODE:', cal_mode_string)

    x_ = (assemble_object.corrected_ccs, assemble_object.charge, assemble_object.mass)
    y_ = assemble_object.exp_avg_velocity
    if not exp:
        popt, pcov = curve_fit(calibration_functions.power_law_fit, x_, y_, maxfev=10000, gtol=1e-20)
        pred_velocity = calibration_functions.power_law_fit(x_, *popt)
        pred_corr_ccs = calibration_functions.power_law_ccs_function(y_, *popt)
    else:
        popt, pcov = curve_fit(calibration_functions.power_law_exp_mass_fit, x_, y_, maxfev=10000, gtol=1e-20)
        pred_velocity = calibration_functions.power_law_exp_mass_fit(x_, *popt)
        pred_corr_ccs = calibration_functions.power_law_exp_mass_ccs_function(y_, assemble_object.charge,
                                                                              assemble_object.mass, *popt)

    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, assemble_object.charge, assemble_object.reduce_mass)
    pred_mobility = calculate_mobility(assemble_object.mass, assemble_object.charge, pred_ccs, assemble_object.temp,
                                       assemble_object.pressure / 1000, assemble_object.gas_mass)

    lin_reg_exp_pred_vel = linregress(y_, pred_velocity)
    vel_deviation = percent_deviation_predict_true(pred_velocity, y_)
    ccs_deviation = percent_deviation_predict_true(pred_ccs, assemble_object.ccs)
    ccs_rmse = rms_error(pred_ccs, assemble_object.ccs)
    vel_rmse = rms_error(pred_velocity, y_)


    cal_fit_dict = create_cal_fit_dict(xdata=x_,
                                       ydata=y_,
                                       cal_fit_obj=None,
                                       cal_fit_obj_score=None,
                                       cal_fit_params=popt,
                                       blended_fixa=None,
                                       cal_fit_pcov=pcov,
                                       linreg_true_pred=lin_reg_exp_pred_vel,
                                       pub_ccs=assemble_object.ccs,
                                       pub_corr_ccs=assemble_object.corrected_ccs,
                                       mobility=assemble_object.mobility,
                                       pred_vel=pred_velocity,
                                       pred_ccs=pred_ccs,
                                       pred_corr_ccs=pred_corr_ccs,
                                       pred_mobility=pred_mobility,
                                       vel_deviation=vel_deviation,
                                       ccs_deviation=ccs_deviation,
                                       vel_rmse=vel_rmse,
                                       ccs_rmse=ccs_rmse)

    cal_scheme = CalibrationScheme()
    cal_scheme.cal_mode = cal_mode_dict
    cal_scheme.cal_output = cal_fit_dict

    dirpath, cal_assemble_fname = os.path.split(assemble_object.assemble_cal_object_path)

    cal_scheme_fname = str(cal_assemble_fname).split('.assemble')[0] + '_' + cal_mode_string + '.cal'
    cal_scheme.cal_object_fpath = os.path.join(dirpath, cal_scheme_fname)

    exp_cond_dict = gen_exp_cond_from_assemble_obj(assemble_object)
    cal_scheme.cal_exp_cond = exp_cond_dict

    cal_list_dict = gen_calibrant_list_from_assemble_obj(assemble_object)
    cal_scheme.cal_calibrants = cal_list_dict


    return cal_scheme



def generate_power_fit_cal(assemble_object, exp=False):
    """
    generate power fit calibration with or without exp
    :param assemble_object: assemble object from twim_ccs_calibration_assemble py file
    :param exp: bool
    :return: calibration scheme
    """

    cal_mode_dict = generate_cal_mode_dict(power_law=True, exp=exp)
    cal_mode_string = generate_cal_mode_string(cal_mode_dict)
    print('CAL_MODE:', cal_mode_string)

    x_ = (assemble_object.corrected_ccs, assemble_object.charge, assemble_object.mass)
    y_ = assemble_object.exp_avg_velocity
    if not exp:
        popt, pcov = curve_fit(calibration_functions.power_law_fit, x_, y_, maxfev=10000, gtol=1e-20)
        pred_velocity = calibration_functions.power_law_fit(x_, *popt)
        pred_corr_ccs = calibration_functions.power_law_ccs_function(y_, *popt)
    else:
        popt, pcov = curve_fit(calibration_functions.power_law_exp_fit, x_, y_, maxfev=10000, gtol=1e-20)
        pred_velocity = calibration_functions.power_law_exp_fit(x_, *popt)
        pred_corr_ccs = calibration_functions.power_law_exp_ccs_function(y_, assemble_object.charge, *popt)

    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, assemble_object.charge, assemble_object.reduce_mass)
    pred_mobility = calculate_mobility(assemble_object.mass, assemble_object.charge, pred_ccs, assemble_object.temp,
                                       assemble_object.pressure / 1000, assemble_object.gas_mass)

    lin_reg_exp_pred_vel = linregress(y_, pred_velocity)
    vel_deviation = percent_deviation_predict_true(pred_velocity, y_)
    ccs_deviation = percent_deviation_predict_true(pred_ccs, assemble_object.ccs)
    ccs_rmse = rms_error(pred_ccs, assemble_object.ccs)
    vel_rmse = rms_error(pred_velocity, y_)


    cal_fit_dict = create_cal_fit_dict(xdata=x_,
                                       ydata=y_,
                                       cal_fit_obj=None,
                                       cal_fit_obj_score=None,
                                       cal_fit_params=popt,
                                       blended_fixa=None,
                                       cal_fit_pcov=pcov,
                                       linreg_true_pred=lin_reg_exp_pred_vel,
                                       pub_ccs=assemble_object.ccs,
                                       pub_corr_ccs=assemble_object.corrected_ccs,
                                       mobility=assemble_object.mobility,
                                       pred_vel=pred_velocity,
                                       pred_ccs=pred_ccs,
                                       pred_corr_ccs=pred_corr_ccs,
                                       pred_mobility=pred_mobility,
                                       vel_deviation=vel_deviation,
                                       ccs_deviation=ccs_deviation,
                                       vel_rmse=vel_rmse,
                                       ccs_rmse=ccs_rmse)

    cal_scheme = CalibrationScheme()
    cal_scheme.cal_mode = cal_mode_dict
    cal_scheme.cal_output = cal_fit_dict

    dirpath, cal_assemble_fname = os.path.split(assemble_object.assemble_cal_object_path)

    cal_scheme_fname = str(cal_assemble_fname).split('.assemble')[0] + '_' + cal_mode_string + '.cal'
    cal_scheme.cal_object_fpath = os.path.join(dirpath, cal_scheme_fname)

    exp_cond_dict = gen_exp_cond_from_assemble_obj(assemble_object)
    cal_scheme.cal_exp_cond = exp_cond_dict

    cal_list_dict = gen_calibrant_list_from_assemble_obj(assemble_object)
    cal_scheme.cal_calibrants = cal_list_dict


    return cal_scheme


def generate_blended_cal(assemble_object, exp=False, fixa=True):
    """
    generate blended calibration
    :param assemble_object: assemble object from twim_ccs_calibration_assemble_2 py
    :param exp: bool
    :return: cal_scheme
    """
    cal_mode_dict = generate_cal_mode_dict(blended=True, exp=exp)
    cal_mode_string = generate_cal_mode_string(cal_mode_dict)
    print('CAL_MODE:', cal_mode_string)

    # mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = X

    x_ = (assemble_object.mobility, assemble_object.mass, assemble_object.charge, assemble_object.wave_height,
          assemble_object.wave_velocity, assemble_object.wave_volt_pot_fac, assemble_object.wave_lambda)
    y_ = assemble_object.exp_avg_velocity


    if not exp:
        if fixa:
            x0 = [0.5]
        else:
            x0 = [0.5, 0.5]

        v_blend_least_sq = least_squares(calibration_functions.minimize_v_gamma_alpha_blended, x0=x0,
                                       args=(x_, y_, calibration_functions.v_blend_cal_func, fixa), method='lm')
        if fixa:
            fit_params_ = [1, v_blend_least_sq.x[0]]
            pred_mob = pred_mob_with_root_finding_v_blend_func(assemble_object.mobility, y_, assemble_object.mass,
                                                               assemble_object.charge, x_[3:], fit_params_,
                                                               exp=exp, fixa=fixa)
            pred_velocity = calibration_functions.v_blend_cal_func(x_, fit_params_, fixa=fixa)
        else:
            fit_params_ = v_blend_least_sq.x
            pred_mob = pred_mob_with_root_finding_v_blend_func(assemble_object.mobility, y_, assemble_object.mass,
                                                               assemble_object.charge, x_[3:], fit_params_,
                                                               exp=exp, fixa=fixa)
            pred_velocity = calibration_functions.v_blend_cal_func(x_, fit_params_, fixa=fixa)


    else:
        if fixa:
            x0 = [0.5,0.5]
        else:
            x0 = [0.5, 0.5, 0.5]

        v_blend_least_sq = least_squares(calibration_functions.minimize_v_gamma_alpha_blended, x0=x0,
                                         args=(x_, y_, calibration_functions.v_blend_exp_cal_func, fixa), method='lm')



        if fixa:
            fit_params_ = [1, *v_blend_least_sq.x]
            pred_velocity = calibration_functions.v_blend_exp_cal_func(x_, fit_params_, fixa=fixa)
            pred_mob = pred_mob_with_root_finding_v_blend_func(assemble_object.mobility, y_, assemble_object.mass,
                                                               assemble_object.charge, x_[3:], fit_params_,
                                                               exp=exp, fixa=fixa)
        else:
            fit_params_ = v_blend_least_sq.x
            pred_velocity = calibration_functions.v_blend_exp_cal_func(x_, fit_params_, fixa=fixa)
            pred_mob = pred_mob_with_root_finding_v_blend_func(assemble_object.mobility, y_, assemble_object.mass,
                                                               assemble_object.charge, x_[3:], fit_params_,
                                                               exp=exp, fixa=fixa)


    pred_ccs = calculate_ccs(assemble_object.mass, assemble_object.charge, pred_mob, assemble_object.temp,
                             assemble_object.pressure / 1000, assemble_object.gas_mass)
    pred_corr_ccs = correct_ccs(pred_ccs, assemble_object.charge, assemble_object.reduce_mass)

    lin_reg_exp_pred_vel = linregress(y_, pred_velocity)

    vel_deviation = percent_deviation_predict_true(pred_velocity, y_)
    ccs_deviation = percent_deviation_predict_true(pred_ccs, assemble_object.ccs)

    ccs_rmse = rms_error(pred_ccs, assemble_object.ccs)
    vel_rmse = rms_error(pred_velocity, y_)


    cal_fit_dict = create_cal_fit_dict(xdata=x_,
                                       ydata=y_,
                                       cal_fit_obj=v_blend_least_sq,
                                       cal_fit_obj_score=None,
                                       cal_fit_params=fit_params_,
                                       blended_fixa=fixa,
                                       cal_fit_pcov=None,
                                       linreg_true_pred=lin_reg_exp_pred_vel,
                                       pub_ccs=assemble_object.ccs,
                                       pub_corr_ccs=assemble_object.corrected_ccs,
                                       mobility=assemble_object.mobility,
                                       pred_vel=pred_velocity,
                                       pred_ccs=pred_ccs,
                                       pred_corr_ccs=pred_corr_ccs,
                                       pred_mobility=pred_mob,
                                       vel_deviation=vel_deviation,
                                       ccs_deviation=ccs_deviation,
                                       vel_rmse=vel_rmse,
                                       ccs_rmse=ccs_rmse)

    cal_scheme = CalibrationScheme()

    cal_scheme.cal_mode = cal_mode_dict
    cal_scheme.cal_output = cal_fit_dict

    dirpath, cal_assemble_fname = os.path.split(assemble_object.assemble_cal_object_path)

    cal_scheme_fname = str(cal_assemble_fname).split('.assemble')[0] + '_' + cal_mode_string + '.cal'
    cal_scheme.cal_object_fpath = os.path.join(dirpath, cal_scheme_fname)

    exp_cond_dict = gen_exp_cond_from_assemble_obj(assemble_object)
    cal_scheme.cal_exp_cond = exp_cond_dict

    cal_list_dict = gen_calibrant_list_from_assemble_obj(assemble_object)
    cal_scheme.cal_calibrants = cal_list_dict


    return cal_scheme



def generate_cal_mode_string(cal_mode_dict):
    """
    uses the cal mode dictionary to generate cal mode string
    :param cal_mode_dict: cal mode dict
    :return: cal mode string
    """
    cal_mode_key = []
    for name, value in cal_mode_dict.items():
        if value:
            cal_mode_key.append([name, value])

    cal_mode_string = ''
    for ind, item in enumerate(cal_mode_key):
        item_str = '_'.join([str(x) for x in item])
        cal_mode_string += item_str + '_'

    if cal_mode_string.find('relax') == -1:
        if cal_mode_string.find('power_law') == -1:
            if cal_mode_string.find('blended') == -1:
                cal_mode_string_chars = cal_mode_string.split('_')
                cal_mode_list = ['relax', 'False'] + cal_mode_string_chars
                cal_mode_string = '_'.join([x for x in cal_mode_list])
    else:
        cal_mode_string = cal_mode_string

    return cal_mode_string


def write_calibrants_output_to_csv(cal_scheme_obj):
    """
    use the cal scheme object to write outputs for calibrants
    :param cal_scheme_obj:
    :return: cal output csv file
    """

    output_string = ''

    cal_mode_string = generate_cal_mode_string(cal_scheme_obj.cal_mode)

    header1 = '#' + cal_mode_string + '\n'
    header_wh = '#Waveht,'+str(cal_scheme_obj.cal_exp_cond['wave_height'])+'\n'
    header_wv = '#Wavevel,'+str(cal_scheme_obj.cal_exp_cond['wave_velocity'])+'\n'
    header_gas_type = '#Gas_type,'+str(cal_scheme_obj.cal_exp_cond['gas_type'])+'\n'
    header_gas_mass = '#Gas_mass,' + str(cal_scheme_obj.cal_exp_cond['gas_mass']) + '\n'
    header_press = '#Pressure,'+str(cal_scheme_obj.cal_exp_cond['pressure'])+'\n'
    header_temp = '#Temperature,'+str(cal_scheme_obj.cal_exp_cond['temperature'])+'\n'
    header_twlength = '#TWIM_length,'+str(cal_scheme_obj.cal_exp_cond['twim_length'])+'\n'
    header_edc = '#EDC,'+str(cal_scheme_obj.cal_exp_cond['edc_constant'])+'\n'
    header_pot_factor = '#Pot_Factor,'+str(cal_scheme_obj.cal_exp_cond['pot_factor'])+'\n'
    header_wave_lambda = '#Wave_lambda,'+str(cal_scheme_obj.cal_exp_cond['wave_lambda'])+'\n'

    header_ = '#id,oligomer,mass,charge,exp_avg_vel,pub_ccs,pub_corr_ccs,mobility,pred_vel,pred_ccs,pred_corr_ccs,pred_mobility,vel_deviation,' \
              'ccs_deviation\n'

    output_string += header1 + header_wh + header_wv + header_gas_mass + header_gas_type + header_press + header_temp + header_twlength + header_edc + header_pot_factor + header_wave_lambda + header_


    for num in range(len(cal_scheme_obj.cal_calibrants['species'])):
        line = '{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n'.format(cal_scheme_obj.cal_calibrants['species'][num],
                                                              cal_scheme_obj.cal_calibrants['oligomer'][num],
                                                              cal_scheme_obj.cal_calibrants['mass'][num],
                                                              cal_scheme_obj.cal_calibrants['charge'][num],
                                                              cal_scheme_obj.cal_output['ydata'][num],
                                                              cal_scheme_obj.cal_output['pub_ccs'][num],
                                                              cal_scheme_obj.cal_output['pub_corr_ccs'][num],
                                                              cal_scheme_obj.cal_output['mobility'][num],
                                                              cal_scheme_obj.cal_output['pred_vel'][num],
                                                              cal_scheme_obj.cal_output['pred_ccs'][num],
                                                              cal_scheme_obj.cal_output['pred_corr_ccs'][num],
                                                              cal_scheme_obj.cal_output['pred_mobility'][num],
                                                              cal_scheme_obj.cal_output['vel_deviation'][num],
                                                              cal_scheme_obj.cal_output['ccs_deviation'][num])
        output_string += line



    dirpath, cal_obj_fname = os.path.split(cal_scheme_obj.cal_object_fpath)
    calibrant_output_fname = str(cal_obj_fname).split('.cal')[0] + 'calibrants_output.csv'

    print ('Saving...', calibrant_output_fname)

    with open(os.path.join(dirpath, calibrant_output_fname), 'w') as outfile:
        outfile.write(output_string)
        outfile.close()



def write_calfit_output(cal_scheme_obj):
    """
    write the calibration fit output to csv [includes x data, ydata, calfit_obj, calfit_obj_score, cal_fit_params, cal_fit_pcov, linreg_true_pred, vel_rmse, ccs_rmse]
    :param cal_scheme_obj:
    :return:
    """
    output_string = ''

    cal_mode_string = generate_cal_mode_string(cal_scheme_obj.cal_mode)

    header1 = '#' + cal_mode_string + '\n'

    header_wh = '#Waveht,' + str(cal_scheme_obj.cal_exp_cond['wave_height']) + '\n'
    header_wv = '#Wavevel,' + str(cal_scheme_obj.cal_exp_cond['wave_velocity']) + '\n'
    header_press = '#Pressure,' + str(cal_scheme_obj.cal_exp_cond['pressure']) + '\n'
    header_edc = '#EDC,' + str(cal_scheme_obj.cal_exp_cond['edc_constant']) + '\n'

    output_string += header1 + header_wh + header_wv + header_press + header_edc


    ##xdata
    if cal_mode_string.find('power_law') >= 0 or (cal_mode_string.find('relax') >= 0 and cal_mode_string.find('exp') >=0):
        xdata_reshape = np.array(cal_scheme_obj.cal_output['xdata']).T
        if cal_mode_string.find('power_law') >= 0:
            header2 = '#xdata,corr_ccs,charge,mass\n'
        else:
            header2 = '#xdata,corr_ccs,mass_over_charge,charge,mass\n'
        # if cal_mode_string.find('exp') >= 0:
        #     header2 = '#xdata,corr_ccs,mass_over_charge,charge,mass\n'
        # else:
        #     header2 = '#xdata,corr_ccs,charge,mass\n'
        output_string += header2
        for arr in xdata_reshape:
            join_arr = ','.join([str(x) for x in arr])
            line='{},{}\n'.format('#xdata',join_arr)
            output_string += line
    if cal_mode_string.find('relax') >= 0 and cal_mode_string.find('exp') == -1 :
        xdata = cal_scheme_obj.cal_output['xdata']
        num_x_vars = np.arange(1, len(xdata[0])+1)
        num_x_vars_str = ','.join(['x'+str(x) for x in num_x_vars])
        header2 = '{},{}\n'.format('#xdata', num_x_vars_str)
        output_string += header2
        for arr in xdata:
            join_arr = ','.join([str(x) for x in arr])
            line = '{},{}\n'.format('#xdata', join_arr)
            output_string += line
    if cal_mode_string.find('blended') >=0:
        xdata = cal_scheme_obj.cal_output['xdata']
        xdata_ = np.array(xdata[:3]).T
        header2 = '#xdata,mobility,mass,charge\n'
        output_string += header2
        for arr in xdata_:
            join_arr = ','.join([str(x) for x in arr])
            line = '{},{}\n'.format('#xdata', join_arr)
            output_string += line
        header2_1 = '#xdata,wave_ht\n'
        output_string += header2_1 + '{},{}\n'.format('#xdata', xdata[3])
        header2_2 = '#xdata,wave_vel\n'
        output_string += header2_2 + '{},{}\n'.format('#xdata', xdata[4])
        header2_3 = '#xdata,wave_volt_pot_fac\n'
        output_string += header2_3 + '{},{}\n'.format('#xdata', xdata[5])
        header2_4 = '#xdata,wave_lambda\n'
        output_string += header2_4 + '{},{}\n'.format('#xdata', xdata[6])




    #ydata
    header3 = '#ydata,exp_avg_vel\n'
    output_string += header3
    for exp_vel in cal_scheme_obj.cal_output['ydata']:
        line = '{},{}\n'.format('#ydata',str(exp_vel))
        output_string += line



    #cal fit obj/params
    if cal_mode_string.find('power_law') >= 0 or (cal_mode_string.find('relax') >= 0 and cal_mode_string.find('exp') >=0):
        #popt
        header4 = '#cal_fit_obj_coefs,None\n'
        header5 = '#cal_fit_obj_score,None\n'
        output_string += header4 + header5

        num_pars = np.arange(1, len(cal_scheme_obj.cal_output['cal_fit_params']) + 1)
        num_pars_str = ','.join(['param'+str(x) for x in num_pars])
        header6 = '{},{}\n'.format('#popt',num_pars_str)

        popt_join_arr = ','.join([str(x) for x in cal_scheme_obj.cal_output['cal_fit_params']])

        line = '{},{}\n'.format('#popt', popt_join_arr)
        output_string += header6 + line

        #pcov
        for arr in cal_scheme_obj.cal_output['cal_fit_pcov']:
            pcov_join_arr = ','.join([str(x) for x in arr])
            line = '{},{}\n'.format('#pcov', pcov_join_arr)
            output_string += line


    if cal_mode_string.find('relax') >= 0 and cal_mode_string.find('exp') == -1:
        #cal_fit_obj_coefs
        cal_fit_coefs = cal_scheme_obj.cal_output['cal_fit_obj'].coef_
        num_coefs = np.arange(1, len(cal_fit_coefs)+1)
        num_coefs_str = ','.join(['coefs_'+str(x) for x in num_coefs])
        header4 = '{},{}\n'.format('#cal_fit_obj_coefs', num_coefs_str)
        coeff_join_str = ','.join([str(x) for x in cal_fit_coefs])
        line = '{},{}\n'.format('#cal_fit_obj_coefs', coeff_join_str)
        output_string += header4 + line

        #cal_fit_obj_score
        header5 = '#cal_fit_obj_score,r2\n'
        line = '{},{}\n'.format('#cal_fit_obj_score', cal_scheme_obj.cal_output['cal_fit_obj_score'])
        output_string += header5 + line

        header6 = '#popt,None\n'
        header_pcov = '#pcov,None\n'
        output_string += header6 + header_pcov

    if cal_mode_string.find('blended') >= 0:
        header4 = '#cal_fit_obj_coefs,None\n'
        header5 = '#cal_fit_obj_score,None\n'
        output_string += header4 + header5

        num_pars = np.arange(1, len(cal_scheme_obj.cal_output['cal_fit_params']) + 1)
        num_pars_str = ','.join(['param' + str(x) for x in num_pars])
        header6 = '{},{}\n'.format('#popt', num_pars_str)

        popt_join_arr = ','.join([str(x) for x in cal_scheme_obj.cal_output['cal_fit_params']])

        line = '{},{}\n'.format('#popt', popt_join_arr)
        output_string += header6 + line

        header_pcov = '#pcov,None\n'
        output_string += header_pcov




    #linreg_true_pred
    header7 = '#linreg_true_pred,slope,intercept,rvalue,pvalue,stderr\n'
    linreg_true_pred_str = ','.join([str(x) for x in cal_scheme_obj.cal_output['linreg_true_pred']])
    line_linreg = '{},{}\n'.format('#linreg_true_pred',linreg_true_pred_str)
    output_string += header7 + line_linreg

    #rmse
    header8 = '#rmse,vel_percent,ccs_percent\n'
    line_rmse = '{},{},{}\n'.format('#rmse',cal_scheme_obj.cal_output['vel_rmse'],cal_scheme_obj.cal_output['ccs_rmse'])
    output_string += header8 + line_rmse

    dirpath, cal_obj_fname = os.path.split(cal_scheme_obj.cal_object_fpath)
    cal_output_fname = str(cal_obj_fname).split('.cal')[0] + 'cal_output.csv'


    print('Saving...', cal_output_fname)

    with open(os.path.join(dirpath, cal_output_fname), 'w') as outfile:
        outfile.write(output_string)
        outfile.close()



def gen_calibration_scheme(assemble_obj_file, cal_mode, fixa_blended=True):
    """
    generate cal scheme object and save it. generate cal ouputs and save them.
    :param assemble_obj_file: assemble file
    :param cal_mode: calibration mode key ['power_law', 'power_law_exp', 'relax_true_4', 'relax_true_6',
    'relax_false_4', 'relax_false_6', 'relax_true_4_exp', 'relax_true_6_exp', 'blended', 'blended_exp']
    :return: cal scheme object
    """
    start_time = time.perf_counter()
    assemble_obj = load_pickle_object(assemble_obj_file)
    if cal_mode == 'power_law':
        cal_scheme = generate_power_fit_cal(assemble_obj, exp=False)
    if cal_mode == 'power_law_exp':
        cal_scheme = generate_power_fit_cal(assemble_obj, exp=True)
    if cal_mode == 'relax_true_4':
        cal_scheme = generate_w_wo_relax_without_exp(assemble_obj, relax=True, terms=4)
    if cal_mode == 'relax_true_6':
        cal_scheme = generate_w_wo_relax_without_exp(assemble_obj, relax=True, terms=6)
    if cal_mode == 'relax_false_4':
        cal_scheme = generate_w_wo_relax_without_exp(assemble_obj, relax=False, terms=4)
    if cal_mode == 'relax_false_6':
        cal_scheme = generate_w_wo_relax_without_exp(assemble_obj, relax=False, terms=6)
    if cal_mode == 'relax_true_4_exp':
        cal_scheme = generate_relax_with_exp(assemble_obj, terms=4)
    if cal_mode == 'relax_true_6_exp':
        cal_scheme = generate_relax_with_exp(assemble_obj, terms=6)
    if cal_mode == 'blended':
        cal_scheme = generate_blended_cal(assemble_obj, exp=False, fixa=fixa_blended)
    if cal_mode == 'blended_exp':
        cal_scheme = generate_blended_cal(assemble_obj, exp=True, fixa=fixa_blended)

    write_calibrants_output_to_csv(cal_scheme)
    write_calfit_output(cal_scheme)

    print('Saving....', os.path.split(cal_scheme.cal_object_fpath)[1])
    save_object_to_pickle(cal_scheme, cal_scheme.cal_object_fpath)

    end_time = time.perf_counter() - start_time
    print('Took:', end_time, ' seconds')

    return cal_scheme


def gen_calibration_from_parser(parser):
    (options, args) = parser.parse_args()
    inputf = options.inputf
    cal_mode = options.cal_type
    assemble_fpath = check_path_return_absolute_path(inputf)
    cal_scheme = gen_calibration_scheme(assemble_fpath, cal_mode)


def parser_commands():
    parser = optparse.OptionParser(description='Creates calibration file from the assemble file')
    parser.add_option('-i', '--inputf', dest='inputf', default='assemblefile.assemble', help='assemble file input type .assemble')
    parser.add_option('-c', '--cal_type', dest='cal_type', default='blended_exp', help='cal_type options: power_law, '
                                                                                 'power_law_exp,'
                                                                                 'relax_true_6,'
                                                                                 'relax_true_6_exp,'
                                                                                 'blended,'
                                                                                 'blended_exp')
    return parser


if __name__ == '__main__':

    parser = parser_commands()
    gen_calibration_from_parser(parser)

    ## examples of manual data input below
    ###

    # using a single input assemble file and single cal type

    # assemble_file = r"C:\Users\sugyan\Documents\Processed data\021519_CalProcessing\cal_input_wv_300.0_wh_20.0.assemble"
    # cal_scheme = gen_calibration_scheme(assemble_file, 'blended')


    # using a list of cal mode list to all assemble files in a directory

    # dirpath = r"C:\Users\sugyan\Documents\Processed data\021519_CalProcessing\MixClass\CalNatProtBSAAvidinCytc\test"
    #
    # sys.stdout = open(os.path.join(dirpath, 'gen_cal_log.txt'), 'w')
    #
    # file_list = os.listdir(dirpath)
    # assemble_file_list = [x for x in file_list if x.endswith('.assemble')]
    # cal_mode_list = ['power_law', 'power_law_exp', 'relax_true_6','relax_true_6_exp', 'blended', 'blended_exp']
    # start_time_tot = time.perf_counter()
    # for assemble_file in assemble_file_list:
    #     for ind, cal_mode in enumerate(cal_mode_list):
    #         print(assemble_file, ' ---> ', cal_mode)
    #         scheme = gen_calibration_scheme(os.path.join(dirpath, assemble_file), cal_mode=cal_mode, fixa_blended=True)
    # end_time_tot = time.perf_counter() - start_time_tot
    # print('Total time took:', end_time_tot, ' seconds')