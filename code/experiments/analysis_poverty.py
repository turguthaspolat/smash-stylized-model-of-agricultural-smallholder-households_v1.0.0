'''
multi-replication scenario analysis
'''
import os
import model.model as mod
import model.base_inputs as inp
import plot.poverty as plt_pov
import plot.single_run as plt
import calibration.POM as POM
import imp
import copy
import code
import tqdm
import numpy as np
import pickle
import copy
from joblib import Parallel, delayed
from tqdm import tqdm
import multiprocessing

def main():
    nreps = 3000
    exp_name_POM = 'es_r2' # for reading POM outputs
    exp_name_base = 'es_r2' # for writing outputs
    ncores = 1

    # load default params
    inp_base = inp.compile()
    #### OR ####
    # load from POM experiment
    pom_nvars = 100000
    pom_nreps = 10
    soln_number = 0 # from the POM procedure
    exp_name = '{}/model_{}'.format(exp_name_base, soln_number)
    f = '../outputs/{}/POM/{}_{}reps/input_params_{}.pkl'.format(exp_name_POM, pom_nvars, pom_nreps, soln_number)
    inp_base = pickle.load(open(f, 'rb'))
    # manually specify some variables (common to all scenarios)
    T = 50
    inp_base['model']['T'] = T + inp_base['adaptation']['burnin_period']
    inp_base['model']['n_agents'] = 300
    inp_base['model']['exp_name'] = exp_name
    inp_base['agents']['adap_type'] = 'always'

    #### adaptation scenarios
    scenarios = {
        'baseline' : {'model' : {'adaptation_option' : 'none'}},
        'insurance' : {'model' : {'adaptation_option' : 'insurance'}},
        'cover_crop' : {'model' : {'adaptation_option' : 'cover_crop'}},
    }
    shock_years = []

    #### shock scenarios
    # scenarios = {
    #     'baseline' : {'model' : {'adaptation_option' : 'none', 'shock' : False}},
    #     'shock' : {'model' : {'adaptation_option' : 'none', 'shock' : True}, 'climate' : {'shock_years' : [15], 'shock_rain' : 0.1}},
    # }
    # shock_years = [15]

    #### RUN THE MODELS ####
    mods = multi_mod_run(nreps, inp_base, scenarios, ncores)

    #### PLOT ####
    plt_pov.main(mods, nreps, inp_base, scenarios, exp_name, T, shock_years)

def multi_mod_run(nreps, inp_base, scenarios, ncores):
    mods = {}
    for name, vals in scenarios.items():
        # change the params
        params = copy.copy(inp_base)
        for k, v in vals.items():
            for k2, v2 in v.items():
                params[k][k2] = v2

        rep_chunks = POM.chunkIt(np.arange(nreps), ncores)
        tmp = Parallel(n_jobs=ncores)(delayed(run_chunk_reps)(rep_chunks[i], params) for i in range(len(rep_chunks)))

        # format the outputs
        mods[name] = {
            'wealth' : np.array([oi for tmp_i in tmp for oi in tmp_i['wealth']]).astype(int),
            'organic' : np.array([oi.astype(int) for tmp_i in tmp for oi in tmp_i['organic']]), # each rep is different length
            'land_area' : np.array([oi for tmp_i in tmp for oi in tmp_i['land_area']]),
            'coping' : np.array([oi for tmp_i in tmp for oi in tmp_i['coping']]),
            'owners' : np.array([oi for tmp_i in tmp for oi in tmp_i['owners']]),
            'income' : np.array([oi for tmp_i in tmp for oi in tmp_i['income']]).astype(int),
            'yields' : np.array([oi for tmp_i in tmp for oi in tmp_i['yields']]).astype(int),
            'climate' : np.array([oi for tmp_i in tmp for oi in tmp_i['climate']]), # cant be int
        }

    return mods

def run_chunk_reps(reps, params):
    '''
    run a chunk of replications
    '''
    params = copy.copy(params)
    ms = {'wealth' : [], 'organic' : [], 'coping' : [], 'land_area' : [], 'owners' : [],
        'income' : [], 'yields' : [], 'climate' : []}
    # with tqdm(reps, disable = not True) as pbar:
    for r in reps:
        params['model']['seed'] = r # set the seed
        
        # initialize and run model
        m = mod.Model(params)
        for t in range(m.T):
            m.step()
        # append to list
        ms['wealth'].append(m.agents.wealth)
        ms['organic'].append(m.land.organic)
        ms['land_area'].append(m.agents.land_area)
        ms['coping'].append(m.agents.coping_rqd)
        ms['owners'].append(m.land.owner)
        ms['income'].append(m.agents.income)
        ms['yields'].append(m.land.yields)
        ms['climate'].append(m.climate.rain)
        # pbar.update()

    return ms

if __name__ == '__main__':
    main()