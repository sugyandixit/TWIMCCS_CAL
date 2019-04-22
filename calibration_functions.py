import numpy as np
from math import pi, sqrt
# import matplotlib.pyplot as plt


def mass_da_to_kg(mass):
    """
    convert mass in daltons to kg
    :param mass: mass in da
    :return: mass in kg
    """
    return mass*1.66054e-27

# list calibration related functions for twim ccs calibration and prediction

def relax_4_terms(X, c2, c4, c42, c6):
    corr_ccs, m_z = X
    ccs_rec = np.reciprocal(corr_ccs)
    ccs_rec_2 = np.power(ccs_rec, 2)
    ccs_rec_4 = np.power(ccs_rec, 4)
    ccs_rec_6 = np.power(ccs_rec, 6)
    mz_2 = np.power(m_z, 2)
    ccs_rec_4_mz_2 = np.multiply(ccs_rec_4, mz_2)
    y = c2 * ccs_rec_2 + c4 * ccs_rec_4 + c42 * ccs_rec_4_mz_2 + c6 * ccs_rec_6
    return y

def relax_4_terms_exp(X, c2, c4, c42, c6, cexp):
    corr_ccs, m_z, z = X
    ccs_rec = np.reciprocal(corr_ccs)
    ccs_rec_2 = np.power(ccs_rec, 2)
    ccs_rec_4 = np.power(ccs_rec, 4)
    ccs_rec_6 = np.power(ccs_rec, 6)
    mz_2 = np.power(m_z, 2)
    ccs_rec_4_mz_2 = np.multiply(ccs_rec_4, mz_2)
    y = (c2 * ccs_rec_2 + c4 * ccs_rec_4 + c42 * ccs_rec_4_mz_2 + c6 * ccs_rec_6) * np.exp(cexp/(z**0.5))
    return y

def relax_4_terms_exp_mass(X, c2, c4, c42, c6, cexp):
    corr_ccs, m_z, z, mass = X
    ccs_rec = np.reciprocal(corr_ccs)
    ccs_rec_2 = np.power(ccs_rec, 2)
    ccs_rec_4 = np.power(ccs_rec, 4)
    ccs_rec_6 = np.power(ccs_rec, 6)
    mz_2 = np.power(m_z, 2)
    ccs_rec_4_mz_2 = np.multiply(ccs_rec_4, mz_2)
    y = (c2 * ccs_rec_2 + c4 * ccs_rec_4 + c42 * ccs_rec_4_mz_2 + c6 * ccs_rec_6) * np.exp((cexp * mass ** 0.5)/z)
    return y

def relax_6_terms_exp(X, c2, c4, c42, c6, c62, c64, cexp):
    corr_ccs, m_z, z = X
    ccs_rec = np.reciprocal(corr_ccs)
    ccs_rec_2 = np.power(ccs_rec, 2)
    ccs_rec_4 = np.power(ccs_rec, 4)
    ccs_rec_6 = np.power(ccs_rec, 6)
    mz_2 = np.power(m_z, 2)
    mz_4 = np.power(m_z, 4)
    ccs_rec_4_mz_2 = np.multiply(ccs_rec_4, mz_2)
    ccs_rec_6_mz_2 = np.multiply(ccs_rec_6, mz_2)
    ccs_rec_6_mz_4 = np.multiply(ccs_rec_6, mz_4)
    y = (c2 * ccs_rec_2 + c4 * ccs_rec_4 + c42 * ccs_rec_4_mz_2 + c6 * ccs_rec_6 + c62 * ccs_rec_6_mz_2 + c64 * ccs_rec_6_mz_4) * np.exp(cexp/(z**0.5))
    return y

def relax_6_terms_exp_mass(X, c2, c4, c42, c6, c62, c64, cexp):
    corr_ccs, m_z, z, mass = X
    ccs_rec = np.reciprocal(corr_ccs)
    ccs_rec_2 = np.power(ccs_rec, 2)
    ccs_rec_4 = np.power(ccs_rec, 4)
    ccs_rec_6 = np.power(ccs_rec, 6)
    mz_2 = np.power(m_z, 2)
    mz_4 = np.power(m_z, 4)
    ccs_rec_4_mz_2 = np.multiply(ccs_rec_4, mz_2)
    ccs_rec_6_mz_2 = np.multiply(ccs_rec_6, mz_2)
    ccs_rec_6_mz_4 = np.multiply(ccs_rec_6, mz_4)
    y = (c2 * ccs_rec_2 + c4 * ccs_rec_4 + c42 * ccs_rec_4_mz_2 + c6 * ccs_rec_6 + c62 * ccs_rec_6_mz_2 +
         c64 * ccs_rec_6_mz_4) * np.exp((cexp * mass ** 0.5)/z)
    return y


def relax_10_terms_exp(X, c2, c4, c42, c6, c62, c64, c8, c82, c84, c86, cexp):
    corr_ccs, m_z, z = X
    ccs_rec = np.reciprocal(corr_ccs)
    ccs_rec_2 = np.power(ccs_rec, 2)
    ccs_rec_4 = np.power(ccs_rec, 4)
    ccs_rec_6 = np.power(ccs_rec, 6)
    ccs_rec_8 = np.power(ccs_rec, 8)
    mz_2 = np.power(m_z, 2)
    mz_4 = np.power(m_z, 4)
    mz_6 = np.power(m_z, 6)
    ccs_rec_4_mz_2 = np.multiply(ccs_rec_4, mz_2)
    ccs_rec_6_mz_2 = np.multiply(ccs_rec_6, mz_2)
    ccs_rec_6_mz_4 = np.multiply(ccs_rec_6, mz_4)
    ccs_rec_8_mz_2 = np.multiply(ccs_rec_8, mz_2)
    ccs_rec_8_mz_4 = np.multiply(ccs_rec_8, mz_4)
    ccs_rec_8_mz_6 = np.multiply(ccs_rec_8, mz_6)
    y = (c2 * ccs_rec_2 + c4 * ccs_rec_4 + c42 * ccs_rec_4_mz_2 + c6 * ccs_rec_6 + c62 * ccs_rec_6_mz_2 +
         c64 * ccs_rec_6_mz_4 + c8 * ccs_rec_8 + c82 * ccs_rec_8_mz_2 + c84 * ccs_rec_8_mz_4 + c86 * ccs_rec_8_mz_6) * np.exp(cexp/(z**0.5))
    return y


def relax_10_terms_exp_mass(X, c2, c4, c42, c6, c62, c64, c8, c82, c84, c86, cexp):
    corr_ccs, m_z, z, mass = X
    ccs_rec = np.reciprocal(corr_ccs)
    ccs_rec_2 = np.power(ccs_rec, 2)
    ccs_rec_4 = np.power(ccs_rec, 4)
    ccs_rec_6 = np.power(ccs_rec, 6)
    ccs_rec_8 = np.power(ccs_rec, 8)
    mz_2 = np.power(m_z, 2)
    mz_4 = np.power(m_z, 4)
    mz_6 = np.power(m_z, 6)
    ccs_rec_4_mz_2 = np.multiply(ccs_rec_4, mz_2)
    ccs_rec_6_mz_2 = np.multiply(ccs_rec_6, mz_2)
    ccs_rec_6_mz_4 = np.multiply(ccs_rec_6, mz_4)
    ccs_rec_8_mz_2 = np.multiply(ccs_rec_8, mz_2)
    ccs_rec_8_mz_4 = np.multiply(ccs_rec_8, mz_4)
    ccs_rec_8_mz_6 = np.multiply(ccs_rec_8, mz_6)
    y = (c2 * ccs_rec_2 + c4 * ccs_rec_4 + c42 * ccs_rec_4_mz_2 + c6 * ccs_rec_6 + c62 * ccs_rec_6_mz_2 +
         c64 * ccs_rec_6_mz_4 + c8 * ccs_rec_8 + c82 * ccs_rec_8_mz_2 + c84 * ccs_rec_8_mz_4 + c86 * ccs_rec_8_mz_6) * np.exp((cexp * mass ** 0.5)/z)
    return y


def power_law_fit(X, a, b):
    corr_ccs, z, mass = X
    ccs_rec = np.reciprocal(corr_ccs)
    y = a * (ccs_rec ** b)
    # y = np.multiply(a, np.power(np.reciprocal(corr_ccs), b))
    return y

def power_law_exp_fit(X, a, b, cexp):
    corr_ccs, z, mass = X
    ccs_rec = np.reciprocal(corr_ccs)
    y = (a * (ccs_rec ** b)) * np.exp(cexp/(z ** 0.5))
    return y


def power_law_exp_mass_fit(X, a, b, cexp):
    corr_ccs, z, mass = X
    ccs_rec = np.reciprocal(corr_ccs)
    y = (a * (ccs_rec ** b)) * np.exp((cexp * mass ** 0.5)/z)
    return y


def power_law_ccs_function(vel, a, b):
    y = 1/((vel/a)**(1/b))
    return y

def power_law_exp_ccs_function(vel, z, a, b, cexp):
    y = 1/(np.exp((1/b) * (np.log(vel) - (cexp/z**0.5)- np.log(a))))
    return y

def power_law_exp_mass_ccs_function(vel, z, mass, a, b, cexp):
    y = 1/(np.exp((1/b) * (np.log(vel) - ((cexp* mass ** 0.5)/z)- np.log(a))))
    return y


def power_law_root_finding_function(x, vel, a, b):
    y = (a * ((1/x) ** b)) - vel
    return y

def power_law_exp_root_finding_function(x, z, vel, a, b, cexp):
    y = ((a * ((1/x) ** b)) * np.exp(cexp/(z ** 0.5))) - vel
    return y

def power_law_exp_mass_root_finding_function(x, z, mass, vel, a, b, cexp):
    y = ((a * ((1/x) ** b)) * np.exp((cexp * mass**0.5)/z)) - vel
    return y

def relax_4_terms_root_finding_function(x, m_z, vel, c2, c4, c42, c6):
    y = c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c6 * ((1 / x) ** 6) - vel
    return y

def relax_6_terms_root_finding_function(x, m_z, vel, c2, c4, c42, c6, c62, c64):
    y = c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c62 * (
                (1 / x) ** 6) * (m_z ** 2) + c64 * ((1 / x) ** 6) * (m_z ** 4) - vel
    return y

def relax_10_terms_root_finding_function(x, m_z, vel, c2, c4, c42, c6, c62, c64, c8, c82, c84, c86):
    y = c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c62 * (
                (1 / x) ** 6) * (m_z ** 2) + c64 * ((1 / x) ** 6) * (m_z ** 4) + c8 * ((1 / x) ** 8) + \
        c82 * ((1 / x) ** 8) * (m_z ** 2) + c84 * ((1 / x) ** 8) * (m_z ** 4) + c86 * ((1 / x) ** 8) * (m_z ** 6) - vel
    return y

def relax_4_terms_exp_root_finding_function(x, m_z, z, vel, c2, c4, c42, c6, cexp):
    y = ((c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c6 * ((1 / x) ** 6)) *
         np.exp(cexp/(z ** 0.5))) - vel
    return y

def relax_4_terms_exp_mass_root_finding_function(x, m_z, z, mass, vel, c2, c4, c42, c6, cexp):
    y = ((c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c6 * ((1 / x) ** 6)) *
         np.exp((cexp * mass ** 0.5)/z)) - vel
    return y

def relax_6_terms_exp_root_finding_function(x, m_z, z, vel, c2, c4, c42, c6, c62, c64, cexp):
    y = ((c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c62 * (
            (1 / x) ** 6) * (m_z ** 2) + c64 * ((1 / x) ** 6) * (m_z ** 4)) * np.exp(cexp/(z ** 0.5))) - vel
    return y

def relax_6_terms_exp_mass_root_finding_function(x, m_z, z, mass, vel, c2, c4, c42, c6, c62, c64, cexp):
    y = ((c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c62 * (
            (1 / x) ** 6) * (m_z ** 2) + c64 * ((1 / x) ** 6) * (m_z ** 4)) * np.exp((cexp * mass ** 0.5)/z)) - vel
    return y

def relax_10_terms_exp_root_finding_function(x, m_z, z, vel, c2, c4, c42, c6, c62, c64, c8, c82, c84, c86, cexp):
    y = ((c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c62 * (
                (1 / x) ** 6) * (m_z ** 2) + c64 * ((1 / x) ** 6) * (m_z ** 4) + c8 * ((1 / x) ** 8) +
          c82 * ((1 / x) ** 8) * (m_z ** 2) + c84 * ((1 / x) ** 8) * (m_z ** 4) + c86 * ((1 / x) ** 8) * (m_z ** 6)) *
         np.exp(cexp/(z ** 0.5))) - vel
    return y

def relax_10_terms_exp_mass_root_finding_function(x, m_z, z, mass, vel, c2, c4, c42, c6, c62, c64, c8, c82, c84, c86, cexp):
    y = ((c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c42 * ((1 / x) ** 4) * (m_z ** 2) + c62 * (
                (1 / x) ** 6) * (m_z ** 2) + c64 * ((1 / x) ** 6) * (m_z ** 4) + c8 * ((1 / x) ** 8) +
          c82 * ((1 / x) ** 8) * (m_z ** 2) + c84 * ((1 / x) ** 8) * (m_z ** 4) + c86 * ((1 / x) ** 8) * (m_z ** 6)) *
         np.exp((cexp * mass ** 0.5)/z)) - vel
    return y

def mob_4_terms_root_finding_function(x, vel, c2, c4, c6, c8):
    y = c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c8 * ((1 / x) ** 8) - vel
    return y

def mob_6_terms_root_finding_function(x, vel, c2, c4, c6, c8, c10, c12):
    y = c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c8 * ((1 / x) ** 8) + c10 * (
                (1 / x) ** 10) + c12 * ((1 / x) ** 12) - vel
    return y

def mob_10_terms_root_finding_function(x, vel, c2, c4, c6, c8, c10, c12, c14, c16, c18, c20):
    y = c2 * ((1 / x) ** 2) + c4 * ((1 / x) ** 4) + c6 * ((1 / x) ** 6) + c8 * ((1 / x) ** 8) + c10 * (
                (1 / x) ** 10) + c12 * ((1 / x) ** 12) + c14 * ((1 / x) ** 14) + c16 * ((1 / x) ** 16) + \
        c18 * ((1 / x) ** 18) + c20 * ((1 / x) ** 20) - vel
    return y



#########################################
########################################
## blended calibration functions ######
#########################################


def gamma(wave_amp_potential, wave_vel, wave_lambda, mobility):
    y = (2*pi*wave_amp_potential*mobility)/(wave_vel*wave_lambda)
    return y

def alpha(wave_vel, wave_lambda, mobility, mass, charge):
    y = (2*pi*wave_vel*mobility*mass)/(wave_lambda*charge)
    return y


def v_const_expansion_6(x, y):
    """
    expansion function with x constant and y expansion with terms up to 6
    :param x: constant
    :param y: expansion
    :return:
    """
    plusx = 1 + x ** 2
    plusx4 = 1 + 4 * x ** 2
    plusx9 = 1 + 9 * x ** 2

    fun1 = 1 / (1 + x ** 2)
    fun2 = (y ** 2) / 2
    fun3 = ((y ** 4) / 8) * ((1 + 10 * x ** 2 + 15 * x ** 4) / ((plusx ** 2) * plusx4))

    fun4 = ((y ** 6) / 16) * ((1 + 23 * x ** 2 + 234 * x ** 4 + 1171 * x ** 6 + 2291 * x ** 8 + 1620 * x ** 10) / (
                (plusx ** 4) * (plusx4 ** 2) * plusx9))

    fun_ = fun2 + fun3 + fun4
    func = fun1 * fun_

    return func


# def v_alpha_6(alpha, gamma, wave_vel):
#     """
#     equation for v alpha 6
#     :param gamma:
#     :param alpha:
#     :return:
#     """
#     omega = np.sqrt(1 - gamma**2)
#     func1 = 1 - omega
#     func2 = (alpha**2) * (omega**2)
#     func3 = (alpha**4) * (omega**2) * (-2 -3*omega + 6*omega**2)
#     func4 = (1/8) * (alpha**6) * (omega**2) * (49 + 81 * omega - 205 * omega ** 2 - 341 * omega ** 3 + 424 * omega ** 4)
#     func = wave_vel * func1 * (1 - func2 + func3 - func4)
#     return func

def v_alpha_6_exp(X, pars, fixa=True):
    """
    equation for v gamma 6
    :param alpha:
    :param gamma:
    :return:
    """
    if fixa:
        g, cexp = pars
        a = 1.
    else:
        a, g, cexp = pars

    mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = X
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht * wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal

    omega = np.sqrt(1 - red_gamma ** 2)

    func1 = 1 - omega
    func2 = (red_alpha ** 2) * (omega ** 2)
    func3 = (red_alpha ** 4) * (omega ** 2) * (-2 - 3 * omega + 6 * omega ** 2)
    func4 = (1 / 8) * (red_alpha ** 6) * (omega ** 2) * (
                49 + 81 * omega - 205 * omega ** 2 - 341 * omega ** 3 + 424 * omega ** 4)
    func = wave_vel * func1 * (1 - func2 + func3 - func4)

    return func




def v_alpha_6(X, pars, fixa=True):
    """
    equation for v gamma 6
    :param alpha:
    :param gamma:
    :return:
    """

    if fixa:
        g = pars[0]
        a = 1.
    else:
        a, g = pars

    mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = X
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht * wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal

    omega = np.sqrt(1 - red_gamma ** 2)

    func1 = 1 - omega
    func2 = (red_alpha ** 2) * (omega ** 2)
    func3 = (red_alpha ** 4) * (omega ** 2) * (-2 - 3 * omega + 6 * omega ** 2)
    func4 = (1 / 8) * (red_alpha ** 6) * (omega ** 2) * (
                49 + 81 * omega - 205 * omega ** 2 - 341 * omega ** 3 + 424 * omega ** 4)
    func = wave_vel * func1 * (1 - func2 + func3 - func4)

    return func


def v_gamma_6_exp(X, pars, fixa=True):
    """
    equation for v gamma 6
    :param alpha:
    :param gamma:
    :return:
    """
    if fixa:
        g, cexp = pars
        a = 1.
    else:
        a, g, cexp = pars

    mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = X
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht * wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal


    plusalpha = 1 + red_alpha ** 2
    plus4alpha = 1 + 4 * red_alpha ** 2
    plus9alpha = 1 + 9 * red_alpha ** 2

    fun1 = 1 / (1 + red_alpha ** 2)
    fun2 = (red_gamma ** 2) / 2
    fun3 = ((red_gamma ** 4) / 8) * ((1 + 10 * red_alpha ** 2 + 15 * red_alpha ** 4) / ((plusalpha ** 2) * plus4alpha))

    fun4 = ((red_gamma ** 6) / 16) * ((1 + 23 * red_alpha ** 2 + 234 * red_alpha ** 4 + 1171 * red_alpha ** 6 + 2291 * red_alpha ** 8 + 1620 * red_alpha ** 10) / (
            (plusalpha ** 4) * (plus4alpha ** 2) * plus9alpha))

    fun_ = fun2 + fun3 + fun4
    func = wave_vel * fun1 * fun_
    return func



def v_gamma_6(X, pars, fixa=True):
    """
    equation for v gamma 6
    :param alpha:
    :param gamma:
    :return:
    """

    if fixa:
        g = pars[0]
        a = 1.
    else:
        a, g = pars

    mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = X
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht * wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal


    plusalpha = 1 + red_alpha ** 2
    plus4alpha = 1 + 4 * red_alpha ** 2
    plus9alpha = 1 + 9 * red_alpha ** 2

    fun1 = 1 / (1 + red_alpha ** 2)
    fun2 = (red_gamma ** 2) / 2
    fun3 = ((red_gamma ** 4) / 8) * ((1 + 10 * red_alpha ** 2 + 15 * red_alpha ** 4) / ((plusalpha ** 2) * plus4alpha))

    fun4 = ((red_gamma ** 6) / 16) * ((1 + 23 * red_alpha ** 2 + 234 * red_alpha ** 4 + 1171 * red_alpha ** 6 + 2291 * red_alpha ** 8 + 1620 * red_alpha ** 10) / (
            (plusalpha ** 4) * (plus4alpha ** 2) * plus9alpha))

    fun_ = fun2 + fun3 + fun4
    func = wave_vel * fun1 * fun_
    return func



def v_blend_cal_func(X, pars, fixa=True):
    """
    blended calibration function. a and g are fitting parameters
    :param a: fitting param
    :param g: fitting param
    :param X: mobility, mass, and charge of the ion
    :return:
    """

    if fixa:
        g = pars[0]
        a = 1.
    else:
        a, g = pars
    mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = X
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht*wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal

    alpha_func = v_alpha_6(X, pars, fixa=fixa)
    gamm_func = v_gamma_6(X, pars, fixa=fixa)

    weight_v_alpha_6 = (red_gamma ** 12) / (red_alpha ** 8 + red_gamma ** 12)
    weight_v_gamma_6 = (red_alpha ** 8) / (red_alpha ** 8 + red_gamma ** 12)

    y = (weight_v_alpha_6 * alpha_func) + (weight_v_gamma_6 * gamm_func)

    return y


def v_blend_exp_cal_func(X, pars, fixa=True):
    """
        blended calibration function with exp term. a and g are fitting parameters
        :param a: fitting param
        :param g: fitting param
        :param c: fitting param
        :param X: mobility, mass, and charge of the ion
        :return:
        """
    if fixa:
        g, cexp = pars
        a = 1.
    else:
        a, g, cexp = pars
    mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = X
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht * wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal

    alpha_func = v_alpha_6_exp(X, pars, fixa=fixa)
    gamm_func = v_gamma_6_exp(X, pars, fixa=fixa)

    weight_v_alpha_6 = (red_gamma ** 12) / (red_alpha ** 8 + red_gamma ** 12)
    weight_v_gamma_6 = (red_alpha ** 8) / (red_alpha ** 8 + red_gamma ** 12)

    y = ((weight_v_alpha_6 * alpha_func) + (weight_v_gamma_6 * gamm_func)) * np.exp(cexp/charge**0.5)

    return y


def v_blend_exp_root_finding_func(mobility, vel, mass, charge, par1, fit_params, fixa=True):
    """
    root finding function to obtain mobility of the ion
    :param mob: mob
    :param X: others
    :return: solution
    """
    wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = par1
    X = (mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda)
    if fixa:
        g, cexp = fit_params
        a = 1.
    else:
        a, g, cexp = fit_params
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht * wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal

    alpha_func = v_alpha_6_exp(X, fit_params, fixa=fixa)
    gamm_func = v_gamma_6_exp(X, fit_params, fixa=fixa)

    weight_v_alpha_6 = (red_gamma ** 12) / (red_alpha ** 8 + red_gamma ** 12)
    weight_v_gamma_6 = (red_alpha ** 8) / (red_alpha ** 8 + red_gamma ** 12)

    y = ((weight_v_alpha_6 * alpha_func) + (weight_v_gamma_6 * gamm_func))*np.exp(cexp/charge**0.5) - vel

    return y




def v_blend_root_finding_func(mobility, vel, mass, charge, par1, fit_params, fixa=True):
    """
    root finding function to obtain mobility of the ion
    :param mob: mob
    :param X: others
    :return: solution
    """
    wave_ht, wave_vel, wave_v_pot_fac, wave_lambda = par1
    X = (mobility, mass, charge, wave_ht, wave_vel, wave_v_pot_fac, wave_lambda)
    if fixa:
        g = fit_params[0]
        a = 1.
    else:
        a, g = fit_params
    mass_kg = mass_da_to_kg(mass)
    charge_q = charge * 1.6e-19
    wave_potential = wave_ht * wave_v_pot_fac
    alpha_nominal = alpha(wave_vel, wave_lambda, mobility, mass_kg, charge_q)
    gamma_nominal = gamma(wave_potential, wave_vel, wave_lambda, mobility)
    red_alpha = a * alpha_nominal
    red_gamma = g * gamma_nominal

    alpha_func = v_alpha_6(X, fit_params, fixa=fixa)
    gamm_func = v_gamma_6(X, fit_params, fixa=fixa)

    weight_v_alpha_6 = (red_gamma ** 12) / (red_alpha ** 8 + red_gamma ** 12)
    weight_v_gamma_6 = (red_alpha ** 8) / (red_alpha ** 8 + red_gamma ** 12)

    y = ((weight_v_alpha_6 * alpha_func) + (weight_v_gamma_6 * gamm_func)) - vel

    return y



def minimize_v_gamma_alpha_blended(pars, X, y, cal_fun, fixa=True):
    y_fit = cal_fun(X, pars, fixa=fixa)
    err = (y-y_fit)/y_fit
    return err**2


########################################
#####
########################################