#prediction of ccs

import os
import optparse
import calibration_functions
from scipy.optimize import curve_fit, root
import matplotlib.pyplot as plt
import pandas as pd
from twim_ccs_gen_calibration import CalibrationScheme, load_pickle_object, pred_mob_with_root_finding_v_blend_func, corr_ccs_to_ccs, pred_corr_ccs_with_root_finding_relax_no_relax_without_exp, pred_corr_ccs_with_root_finding_relax_with_exp, generate_cal_mode_string
from twim_ccs_gen_assemble import correct_drift_time, calculate_average_velocity, calculate_ccs, reduced_mass, calculate_mobility, check_path_return_absolute_path


def exp_func(x, a, b):
    y = a*b**x
    return y

def lin_func(x, m, b):
    y = m*x + b
    return y

def power_func(x, a, b):
    y = a*x**b
    return y

def gen_assemble_uncal_dict(unk_file, cal_scheme_obj):

    cal_mode_string = generate_cal_mode_string(cal_scheme_obj.cal_mode)
    dirpath, unkfname = os.path.split(unk_file)
    unk_pred_fname = str(unkfname).split('.csv')[0]+'_'+cal_mode_string+'pred.csv'
    unk_pred_fpath = os.path.join(dirpath, unk_pred_fname)

    unk_df = pd.read_csv(unk_file)
    assemble_uncal_dict = dict()
    uncal_id = unk_df['#id'].values
    uncal_mass = unk_df['mass'].values
    uncal_charge = unk_df['charge'].values
    uncal_dt = unk_df['dt'].values
    uncal_oligomer = unk_df['oligomer'].values
    uncal_mass_over_charge = uncal_mass/uncal_charge
    corr_dt = correct_drift_time(uncal_dt, uncal_mass_over_charge, cal_scheme_obj.cal_exp_cond['edc_constant'])
    exp_avg_vel = calculate_average_velocity(corr_dt, cal_scheme_obj.cal_exp_cond['twim_length'])
    reduce_mass = reduced_mass(uncal_mass, cal_scheme_obj.cal_exp_cond['gas_mass'])

    assemble_uncal_dict['id'] = uncal_id
    assemble_uncal_dict['oligomer'] = uncal_oligomer
    assemble_uncal_dict['mass'] = uncal_mass
    assemble_uncal_dict['charge'] = uncal_charge
    assemble_uncal_dict['dt'] = uncal_dt
    assemble_uncal_dict['mass_over_charge'] = uncal_mass_over_charge
    assemble_uncal_dict['corr_dt'] = corr_dt
    assemble_uncal_dict['exp_avg_vel'] = exp_avg_vel
    assemble_uncal_dict['reduce_mass'] = reduce_mass

    assemble_uncal_dict['fpath'] = unk_pred_fpath

    return assemble_uncal_dict


def pred_ccs_blended_cal(unk_file, cal_scheme_obj):
    uncal_dict = gen_assemble_uncal_dict(unk_file, cal_scheme_obj)

    popt, pcov = curve_fit(power_func, cal_scheme_obj.cal_output['ydata'], cal_scheme_obj.cal_output['mobility'])

    mob_init_guess_arr = power_func(uncal_dict['exp_avg_vel'], *popt)

    par1 = (cal_scheme_obj.cal_exp_cond['wave_height'], cal_scheme_obj.cal_exp_cond['wave_velocity'],
            cal_scheme_obj.cal_exp_cond['pot_factor'], cal_scheme_obj.cal_exp_cond['wave_lambda'])

    pred_mob = pred_mob_with_root_finding_v_blend_func(mob_init_guess_arr, uncal_dict['exp_avg_vel'],
                                                   uncal_dict['mass'], uncal_dict['charge'], par1,
                                                   cal_scheme_obj.cal_output['cal_fit_params'], exp=cal_scheme_obj.cal_mode['exp'],
                                                       fixa=cal_scheme_obj.cal_output['blended_fixa'])

    pred_ccs = calculate_ccs(uncal_dict['mass'], uncal_dict['charge'], pred_mob, cal_scheme_obj.cal_exp_cond['temperature'],
                                  cal_scheme_obj.cal_exp_cond['pressure'] / 1000, cal_scheme_obj.cal_exp_cond['gas_mass'])

    uncal_dict['pred_mob'] = pred_mob
    uncal_dict['pred_ccs'] = pred_ccs

    return uncal_dict


def pred_ccs_power_law(unk_file, cal_scheme_obj):
    uncal_dict = gen_assemble_uncal_dict(unk_file, cal_scheme_obj)
    if not cal_scheme_obj.cal_mode['exp']:
        pred_corr_ccs = calibration_functions.power_law_ccs_function(uncal_dict['exp_avg_vel'],
                                                                     *cal_scheme_obj.cal_output['cal_fit_params'])
    else:
        pred_corr_ccs = calibration_functions.power_law_exp_ccs_function(uncal_dict['exp_avg_vel'],
                                                                         uncal_dict['charge'],
                                                                         *cal_scheme_obj.cal_output['cal_fit_params'])
    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, uncal_dict['charge'], uncal_dict['reduce_mass'])
    pred_mob = calculate_mobility(uncal_dict['mass'], uncal_dict['charge'], pred_ccs, cal_scheme_obj.cal_exp_cond['temperature'],
                                  cal_scheme_obj.cal_exp_cond['pressure']/1000, cal_scheme_obj.cal_exp_cond['gas_mass'])

    uncal_dict['pred_mob'] = pred_mob
    uncal_dict['pred_ccs'] = pred_ccs

    print('heho')
    return uncal_dict


def pred_ccs_relax_no_relax_without_exp(unk_file, cal_scheme_obj):
    uncal_dict = gen_assemble_uncal_dict(unk_file, cal_scheme_obj)

    popt, pcov = curve_fit(power_func, cal_scheme_obj.cal_output['ydata'], cal_scheme_obj.cal_output['pub_corr_ccs'])
    corr_ccs_guess_arr = power_func(uncal_dict['exp_avg_vel'], *popt)

    pred_corr_ccs = pred_corr_ccs_with_root_finding_relax_no_relax_without_exp(cal_scheme_obj.cal_output['cal_fit_obj'],
                                                                               corr_ccs_guess_arr,
                                                                               uncal_dict['mass_over_charge'],
                                                                               uncal_dict['charge'],
                                                                               uncal_dict['reduce_mass'],
                                                                               uncal_dict['exp_avg_vel'],
                                                                               relax=cal_scheme_obj.cal_mode['relax'],
                                                                               terms=cal_scheme_obj.cal_mode['terms'])

    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, uncal_dict['charge'], uncal_dict['reduce_mass'])
    pred_mob = calculate_mobility(uncal_dict['mass'], uncal_dict['charge'], pred_ccs,
                                  cal_scheme_obj.cal_exp_cond['temperature'],
                                  cal_scheme_obj.cal_exp_cond['pressure'] / 1000,
                                  cal_scheme_obj.cal_exp_cond['gas_mass'])

    uncal_dict['pred_mob'] = pred_mob
    uncal_dict['pred_ccs'] = pred_ccs

    print('heho')
    return uncal_dict


def pred_ccs_relax_with_exp(unk_file, cal_scheme_obj):
    uncal_dict = gen_assemble_uncal_dict(unk_file, cal_scheme_obj)

    popt, pcov = curve_fit(power_func, cal_scheme_obj.cal_output['ydata'], cal_scheme_obj.cal_output['pub_corr_ccs'])
    corr_ccs_guess_arr = power_func(uncal_dict['exp_avg_vel'], *popt)

    pred_corr_ccs = pred_corr_ccs_with_root_finding_relax_with_exp(cal_scheme_obj.cal_output['cal_fit_params'],
                                                                   corr_ccs_guess_arr,
                                                                   uncal_dict['mass_over_charge'], uncal_dict['charge'],
                                                                   uncal_dict['exp_avg_vel'],
                                                                   terms=cal_scheme_obj.cal_mode['terms'])

    pred_ccs = corr_ccs_to_ccs(pred_corr_ccs, uncal_dict['charge'], uncal_dict['reduce_mass'])
    pred_mob = calculate_mobility(uncal_dict['mass'], uncal_dict['charge'], pred_ccs,
                                  cal_scheme_obj.cal_exp_cond['temperature'],
                                  cal_scheme_obj.cal_exp_cond['pressure'] / 1000,
                                  cal_scheme_obj.cal_exp_cond['gas_mass'])

    uncal_dict['pred_mob'] = pred_mob
    uncal_dict['pred_ccs'] = pred_ccs

    print('heho')
    return uncal_dict



def write_pred_output_to_csv(uncal_dict, cal_scheme_obj):
    """
    use the uncal dict after predictions and cal scheme object to write ouptut to csv file
    :param uncal_dict: uncal dict after predictions saved
    :param cal_scheme_obj: cal scheme object
    :param dirpath: dir lcoation
    :return:
    """
    output_string = ''

    cal_mode_string = generate_cal_mode_string(cal_scheme_obj.cal_mode)

    header1 = '#' + cal_mode_string + '\n'
    header_wh = '#Waveht,' + str(cal_scheme_obj.cal_exp_cond['wave_height']) + '\n'
    header_wv = '#Wavevel,' + str(cal_scheme_obj.cal_exp_cond['wave_velocity']) + '\n'
    header_gas_type = '#Gas_type,' + str(cal_scheme_obj.cal_exp_cond['gas_type']) + '\n'
    header_gas_mass = '#Gas_mass,' + str(cal_scheme_obj.cal_exp_cond['gas_mass']) + '\n'
    header_press = '#Pressure,' + str(cal_scheme_obj.cal_exp_cond['pressure']) + '\n'
    header_temp = '#Temperature,' + str(cal_scheme_obj.cal_exp_cond['temperature']) + '\n'
    header_twlength = '#TWIM_length,' + str(cal_scheme_obj.cal_exp_cond['twim_length']) + '\n'
    header_edc = '#EDC,' + str(cal_scheme_obj.cal_exp_cond['edc_constant']) + '\n'
    header_pot_factor = '#Pot_Factor,' + str(cal_scheme_obj.cal_exp_cond['pot_factor']) + '\n'
    header_wave_lambda = '#Wave_lambda,' + str(cal_scheme_obj.cal_exp_cond['wave_lambda']) + '\n'
    header_ccs_rmse = '#CCS_RMSE(%),'+str(cal_scheme_obj.cal_output['ccs_rmse']) + '\n'
    header_ = '#id,oligomer,mass,charge,dt,exp_avg_vel,pred_mob,pred_ccs\n'

    output_string += header1 + header_wh + header_wv + header_gas_mass + header_gas_type + header_press + header_temp + header_twlength + header_edc + header_pot_factor + header_wave_lambda + header_ccs_rmse + header_

    for num in range(len(uncal_dict['id'])):
        line = '{},{},{},{},{},{},{},{}\n'.format(uncal_dict['id'][num],
                                               uncal_dict['oligomer'][num],
                                               uncal_dict['mass'][num],
                                               uncal_dict['charge'][num],
                                               uncal_dict['dt'][num],
                                               uncal_dict['exp_avg_vel'][num],
                                               uncal_dict['pred_mob'][num],
                                               uncal_dict['pred_ccs'][num])
        output_string += line

    with open(uncal_dict['fpath'], 'w') as outfile:
        outfile.write(output_string)
        outfile.close()



def pred_ccs_(unkfile, cal_object_file):
    """
    determine the type of cal mode and use respective pred function
    :param unkfile: unknown file
    :param cal_object_file: cal object file
    :return: output csv
    """
    cal_scheme_obj = load_pickle_object(cal_object_file)
    cal_mode_string = generate_cal_mode_string(cal_scheme_obj.cal_mode)
    if cal_mode_string.find('power_law') >=0:
        uncal_dict = pred_ccs_power_law(unkfile, cal_scheme_obj)
    if cal_mode_string.find('blended') >=0:
        uncal_dict = pred_ccs_blended_cal(unkfile, cal_scheme_obj)
    if cal_mode_string.find('relax') >=0 and cal_mode_string.find('exp') >= 0:
        uncal_dict = pred_ccs_relax_with_exp(unkfile, cal_scheme_obj)
    if cal_mode_string.find('relax') >=0 and cal_mode_string.find('exp') ==-1:
        uncal_dict = pred_ccs_relax_no_relax_without_exp(unkfile, cal_scheme_obj)

    write_pred_output_to_csv(uncal_dict, cal_scheme_obj)

    return uncal_dict


def prediction_from_parser(parser):
    (options, args) = parser.parse_args()
    inputf = options.inputf
    unkf = options.unkf
    cal_fpath = check_path_return_absolute_path(inputf)
    unk_fpath = check_path_return_absolute_path(unkf)
    cal_scheme = pred_ccs_(unk_fpath, cal_fpath)


def parser_commands():
    parser = optparse.OptionParser(description='Creates calibration file from the assemble file')
    parser.add_option('-i', '--inputf', dest='inputf', default='calfile.cal', help='cal file input type .cal')
    parser.add_option('-u', '--unkf', dest='unkf', default='unk_file.csv', help='unknown file type .csv')
    return parser




if __name__ == '__main__':
    parser = parser_commands()
    prediction_from_parser(parser)

    ## examples of manual data input below
    ###

    # using a single unknown and cal object file


    # unkfile = r"C:\Users\sugyan\Documents\Processed data\021519_CalProcessing\MixClass\CalNatProtBSAAvidinCytc\test\unk_input_wv_300.0_wh_20.0.csv"
    # cal_object_file = r"C:\Users\sugyan\Documents\Processed data\021519_CalProcessing\MixClass\CalNatProtBSAAvidinCytc\test\nofixa_cal_input_wv_300.0_wh_20.0_blended_True_exp_True_.cal"
    # uncal_out = pred_ccs_(unkfile, cal_object_file)
    #

    # using a list of pair of unk file and cal object file stored in csv

    # cal_predict_file = r"C:\Users\sugyan\Documents\Processed data\021519_CalProcessing\MixClass\CalNatProtBSAAvidinCytc\cal_predict_files.csv"
    # cal_predict_file_df = pd.read_csv(cal_predict_file)
    # for ind, (unkfile, cal_object_file) in enumerate(zip(cal_predict_file_df['unk_csv_fpath'].values, cal_predict_file_df['cal_obj_fpath'].values)):
    #     print(unkfile, ' -----> ', cal_object_file)
    #     uncal_out = pred_ccs_(unkfile, cal_object_file)
    #
    # print('heho')
