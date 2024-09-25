#
##
###
####
#####

##########################################################################################################################
import os
import math
import numpy as np
from scipy import integrate
from matplotlib import pyplot as plt
from scipy.optimize import curve_fit
from joblib import Parallel, delayed
from flask import Flask, request, render_template, send_file

##########################################################################################################################

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['STATIC_FOLDER'] = 'static'

##########################################################################################################################

OM_default = 0.286
OA_default = 1 - OM_default
c_default = 3 * (10**8)
H0_default = 69600
Flim_default = 0.4
Ndata_default = 447
kstart_default = 5
kend_default = 6
year_default = 6.25

##########################################################################################################################

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)

##########################################################################################################################

def func(x, OM, OA):
    return 1 / (OM * (1 + x)**3 + OA)**0.5

def s(func, m, n, OM, OA):
    return integrate.quad(func, m, n, args=(OM, OA))[0]

def DLF(x, OM, OA, H0, c):
    return 3.08568 * 10**24 * c * (1 + x) * s(func, 0, x, OM, OA) / H0

def Relation(x, OM, OA, H0, Flim, c):
    return 10**39 * 4 * math.pi * (DLF(x, OM, OA, H0, c) / (10**28))**2 * Flim * (600 / 1000) / (1 + x)

def EvoFc(x, l):
    return (1 + x)**l

def set_custom_axes_style(ax):
    ax.minorticks_on()

    ax.tick_params(
        axis='x',
        direction='in',
        length=4,
        width=1,
        colors='k',
        labelsize=8,
        bottom=True,
        top=True
    )

    ax.tick_params(
        axis='x',
        direction='in',
        length=2,
        width=0.5,
        colors='k',
        which='minor'
    )

    ax.tick_params(
        axis='y',
        direction='in',
        length=4,
        width=1,
        colors='k',
        labelsize=8,
        left=True,
        right=True
    )

    ax.tick_params(
        axis='y',
        direction='in',
        length=2,
        width=0.5,
        colors='k',
        which='minor'
    )

    ax.tick_params(
        top=True,
        right=True,
        which='both'
    )

    
    ax.spines['bottom'].set_linewidth(1)
    ax.spines['top'].set_linewidth(1)
    ax.spines['left'].set_linewidth(1)
    ax.spines['right'].set_linewidth(1)
##########################################################################################################################

# main code
@app.route('/')
def index():
    return render_template('index.html')

###############################################################

@app.route('/calculate', methods=['POST'])
def calculate():
    OM = float(request.form.get('OM', OM_default))
    OA = float(request.form.get('OA', OA_default))
    H0 = float(request.form.get('H0', H0_default))
    c = float(request.form.get('c', c_default))
    Flim = float(request.form.get('Flim', Flim_default))
    Ndata = int(request.form.get('Ndata', Ndata_default))
    kstart = float(request.form.get('kstart', kstart_default))
    kend = float(request.form.get('kend', kend_default))
    year = float(request.form.get('year', year_default))
    file = request.files['data_file']

###############################################################
    
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(file_path)
    data = np.loadtxt(file_path)
    z = data[:, 0]
    Eiso = data[:, 1]

    
    fig, ax = plt.subplots(figsize=(6.25, 5), dpi=300)
    axp = plt.gca()
    set_custom_axes_style(axp)
    ax.set_ylabel("$\mathregular{E_{iso}}$(erg)", fontsize=8)
    ax.set_xlabel("z", fontsize=8)
    ax.set_yscale('log')
    ax.set_xlim((0, 3))
    ax.set_ylim((1e36, 1e43))
    ax.errorbar(z, Eiso, color='blue', marker='o', markersize=3, capsize=0.5, linestyle='None', alpha=0.8)

    
    listElim = []
    listz = []
    for i in np.arange(0.01, 10, 0.01):
        LlimE = Relation(i, OM, OA, H0, Flim, c)
        listElim.append(LlimE)
        listz.append(i)
    ax.plot(listz, listElim, 'g-')

    output_image_path1 = os.path.join(app.config['STATIC_FOLDER'], 'Eiso_z_plot.png')
    plt.savefig(output_image_path1)
    plt.close()

 ###############################################################   



    K1 = []
    TAU1 = []

    def process_k(K):
        sumup = 0
        sumdn = 0
        E0com = Eiso / EvoFc(z, K)
        merged_array = np.vstack((E0com, z)).T
    
        for i in range(Ndata):
            E0 = Eiso[i] / EvoFc(z[i], K)
            E0lim = Relation(z[i], OM, OA, H0, Flim, c) / EvoFc(z[i], K)
            zcriteria = z[i]

            for q in np.arange(0, 10.01, 0.01):
                Re = Relation(q, OM, OA, H0, Flim, c) / EvoFc(q, K)
                gap = Re / E0
                if abs(gap) > 1:
                    break

            zmax = q
            Ni = 1
            Mi = 1
            Ri = 1

            condition1 = lambda x: x[0] > E0
            condition2 = lambda x: x[1] < zmax
            condition3 = lambda x: x[0] > E0lim
            condition4 = lambda x: x[1] < zcriteria

            for element in merged_array:
                if condition1(element) and condition2(element):
                    Ni += 1
            for element in merged_array:
                if condition3(element) and condition4(element):
                    Mi += 1 
            for element in merged_array:
                if condition1(element) and condition4(element):
                    Ri += 1

            Ei = (1 + Ni) / 2
            Vi = (Ni - 1) ** 2 / 12
            temp1 = Ri - Ei
            sumup += temp1
            sumdn += Vi

        tao = sumup / (sumdn ** 0.5)
        return K, tao

    results = Parallel(n_jobs=-1)(delayed(process_k)(K) for K in np.linspace(kstart, kend, 30))
    K1, TAU1 = zip(*results)

    fig, ax1 = plt.subplots(figsize=(6.25, 5), dpi=300)
    axp = plt.gca()
    set_custom_axes_style(axp)
    ax1.set_ylabel("\u03C4", fontsize=8)
    ax1.set_xlabel("k", fontsize=8)
    ax1.plot(K1, [-tau for tau in TAU1], label='P', color='pink')
    ax1.axhline(y=0, color='gray', linestyle='--')


    for i in range(0,len(K1)+1):
        K_fix = (K1[i+1]+K1[i])/2
        T_fix = (TAU1[i+1]+TAU1[i])/2
        if TAU1[i+1]*TAU1[i]<0:
            break
    print(K_fix,T_fix)

    output_image_path2 = os.path.join(app.config['STATIC_FOLDER'], 'K_TAU_plot.png')
    plt.savefig(output_image_path2)
    plt.close()
###############################################################

    NI = []
    MI = []

    E0com_fix  = Eiso / EvoFc(z, K_fix)
    array3 = E0com_fix
    array4 = z
    merged_array1 = np.vstack((array3, array4)).T

    def compute_Ni_Mi(i, Eiso, z, K_fix, merged_array1, OM, OA, H0, Flim, c):
        E0_fix = Eiso[i] / EvoFc(z[i], K_fix)
        E0lim_fix = Relation(z[i], OM, OA, H0, Flim, c) / EvoFc(z[i], K_fix)
        zcriteria_fix = z[i]

        for p in np.arange(0, 10.01, 0.01):
            Re = Relation(p, OM, OA, H0, Flim, c) / EvoFc(p, K_fix)
            gap = Re / E0_fix
            if abs(gap) > 1:
                break
        zmax_fix = p

        num_Ni = 1
        num_Mi = 1

        condition1 = lambda x: x[0] > E0_fix
        condition2 = lambda x: x[1] < zmax_fix
        condition3 = lambda x: x[0] > E0lim_fix
        condition4 = lambda x: x[1] < zcriteria_fix

        for element in merged_array1:
            if condition1(element) and condition2(element):
                num_Ni += 1
            if condition3(element) and condition4(element):
                num_Mi += 1

        return num_Ni, num_Mi   
     
    results = Parallel(n_jobs=-1)(delayed(compute_Ni_Mi)(i, Eiso, z, K_fix, merged_array1, OM, OA, H0, Flim, c) for i in range(Ndata))
    NI, MI = zip(*results)
    NI = np.array(NI)
    MI = np.array(MI) 

###############################################################

    E    = []
    PsiE = []
    Z    = []
    Phiz = []

    psiE = 1
    phiz = 1

    for i in range(0, Ndata):
        psiE = 1
        phiz = 1
        for j in range(0, Ndata):
            if E0com_fix[j] > E0com_fix[i] and NI[j] != 0:
                psiE = psiE * (1 + 1 / NI[j])
            if z[j] < z[i] and MI[j] != 0:
                phiz = phiz * (1 + 1 / MI[j])
        
        E.append(E0com_fix[i])
        PsiE.append(psiE)
        Z.append(z[i])
        Phiz.append(phiz)

    E = np.array(E)
    PsiE = np.array(PsiE)
    Z = np.array(Z)
    Phiz = np.array(Phiz)
    sorted_indices1 = np.argsort(E)
    sorted_E   = E[sorted_indices1]
    sorted_PsiE = PsiE[sorted_indices1]

    plt.figure(figsize=(6.25, 5), dpi=300)
    axp = plt.gca()
    set_custom_axes_style(axp)
    plt.ylabel("Cumulative E", fontsize=8)
    plt.xlabel("E ($\mathregular{erg}$)", fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.plot(sorted_E, sorted_PsiE, '-')

    output_image_path3 = os.path.join(app.config['STATIC_FOLDER'], 'CumulativeE_plot.png')
    plt.savefig(output_image_path3)
    plt.close()

###############################################################

    sorted_indices2 = np.argsort(Z)
    sorted_Z   = Z[sorted_indices2]
    sorted_Phiz = Phiz[sorted_indices2]

    plt.figure(figsize=(6.25, 5), dpi=300)
    axp = plt.gca()
    set_custom_axes_style(axp)
    plt.ylabel("Cumulative redshift", fontsize=8)
    plt.xlabel("z", fontsize=8)
    plt.plot(sorted_Z, sorted_Phiz, '-')

    output_image_path4 = os.path.join(app.config['STATIC_FOLDER'], 'CumulativeZ_plot.png')
    plt.savefig(output_image_path4)
    plt.close()

###############################################################

    Meanz = []
    Rou = []
    Rouerr = []

    
    select = np.array([0, math.ceil(Ndata / 15), math.ceil(Ndata / 12), math.ceil(Ndata / 5.96), math.ceil(Ndata / 4.42), math.ceil(Ndata / 2.85), math.ceil(Ndata / 2.18), math.ceil(Ndata / 1.75), math.ceil(Ndata / 1.48), math.ceil(Ndata / 1.28), math.ceil(Ndata / 1.10), math.ceil(Ndata / 1.02), Ndata])
    select2 = select - 1

    for i in range(1, len(select)):
        dphiz = sorted_Phiz[select2[i]] - sorted_Phiz[select2[i - 1]]
        dz = sorted_Z[select2[i]] - sorted_Z[select2[i - 1]]    
        slope = dphiz / dz
        meanz = (sorted_Z[select2[i]] + sorted_Z[select2[i - 1]]) / 2
        dVdz = (1.32243428571429E+28 * 4 * math.pi * DLF(meanz, OM, OA, H0, c)**2 / ((1 + meanz)**2) * func(meanz, OM, OA))  # cm^3
        errorbar = (1 + meanz) * (slope)**0.5
        rou = ((3.08568e+24)**3) * slope * (1 + meanz) * (dVdz**(-1)) / year  # yr⁻¹ Mpc⁻³
        rouerr = ((3.08568e+24)**3) * errorbar * (dVdz**(-1)) / year  # yr⁻¹ Mpc⁻³

        Meanz.append(meanz)
        Rou.append(rou)
        Rouerr.append(rouerr)

    Meanz = np.array(Meanz)
    Rou = np.array(Rou)
    Rouerr = np.array(Rouerr)

    sorted_indices3 = np.argsort(Meanz)
    sorted_Meanz = Meanz[sorted_indices3]
    sorted_Rou   = Rou[sorted_indices3]
    sorted_Rouerr= Rouerr[sorted_indices3]


    plt.figure(figsize=(6.25, 5), dpi=300)
    axp = plt.gca()
    set_custom_axes_style(axp)
    plt.ylabel("Rate", fontsize=8)
    plt.xlabel("1+z", fontsize=8)
    plt.xscale('log')
    plt.yscale('log')
    plt.step(1+sorted_Meanz, sorted_Rou, where='mid', linewidth=2)
    plt.errorbar(1+sorted_Meanz, sorted_Rou, yerr=sorted_Rouerr, linestyle='None',  marker='o' , markersize=1, color='b')

    output_image_path5 = os.path.join(app.config['STATIC_FOLDER'], 'Rate_plot.png')
    plt.savefig(output_image_path5)
    plt.close()

###############################################################

    def rho(x,p,q):
        return p + q*(x)
    
    logrou = np.log10(sorted_Rou/[sorted_Rou[0]])
    logz   = np.log10(sorted_Meanz+1)

    
    popt, pcov = curve_fit(rho, logz, logrou)
    perr = np.sqrt(np.diag(pcov))  
    PL_FIT_INDEX = popt[1]
    PL_FIT_INDEX_ERROR = perr[1]
 
    logrou_fit = rho(logz, *popt)

    
    plt.figure(figsize=(6.25, 5), dpi=300)
    axp = plt.gca()
    set_custom_axes_style(axp)
    plt.scatter(logz, logrou, label='Data', color='blue', s=10)
    plt.plot(logz, logrou_fit, label='Fitted curve', color='red', linestyle='--', linewidth=2)
    plt.xlabel('log(1+z)', fontsize=8)
    plt.ylabel('log rate', fontsize=8)
    plt.legend()
    plt.show()

    output_image_path6 = os.path.join(app.config['STATIC_FOLDER'], 'Rate_Fit_plot.png')
    plt.savefig(output_image_path6)
    plt.close()

###############################################################

    return render_template('result.html', image_path1=output_image_path1, image_path2=output_image_path2, image_path3=output_image_path3, image_path4=output_image_path4, image_path5=output_image_path5, image_path6=output_image_path6, K_fix=K_fix, T_fix=T_fix, PL_FIT_INDEX=PL_FIT_INDEX, PL_FIT_INDEX_ERROR=PL_FIT_INDEX_ERROR)

##########################################################################################################################

@app.route('/download/<filename>')
def download_file(filename):
    return send_file(os.path.join(app.config['STATIC_FOLDER'], filename), as_attachment=True)

##########################################################################################################################

if __name__ == '__main__':
    app.run(debug=True)
