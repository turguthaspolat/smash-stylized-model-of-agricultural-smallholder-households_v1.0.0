'''
model class
'''
import numpy as np
from .agents import Agents
from .land import Land
from .climate import Climate
import code

class Model():
    def __init__(self, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['model']
        for key, val in self.inputs.items():
            setattr(self, key, val)

        np.random.seed(self.seed)

        # initialize the sub-objects
        self.climate = Climate(inputs)
        self.agents = Agents(inputs)
        self.land = Land(self.agents, inputs)

        # set the time
        # save as list so it is mutable (stays same over all objects)
        self.t = self.agents.t = self.land.t = self.climate.t = [0] 
        
        # initialize adaptation options
        self.init_adaptation_option()
        np.random.seed(self.seed*2) # set seed again -- init adap option uses random numbers only if adap option selected

    def step(self):
        '''
        advance the simulation by one year
        '''
        self.land.update_soil(self.agents, self.adap_properties, self.climate)
        self.land.crop_yields(self.agents, self.climate)
        self.agents.calculate_income(self.land, self.climate, self.adap_properties)
        self.agents.coping_measures(self.land)
        self.agents.adaptation(self.land, self.adap_properties)
        # increment the year
        self.t[0] += 1

    def init_adaptation_option(self):
        '''
        calculate any required parameters for the selected adaptation option
        '''
        if self.adaptation_option == 'insurance':
            ## calculate the insurance cost and payout
            ## make it fair in expectation
            [cost, payout, magnitude] = self.calc_insurance_cost()
            # save these
            self.adap_properties = {
                'type' : 'insurance',
                'cost' : cost * self.all_inputs['adaptation']['insurance']['cost_factor'],
                'payout' : payout,
                'magnitude' : magnitude,
                'adap' : True,
            }
        elif self.adaptation_option == 'cover_crop':
            # assume the cost is the same as insurance
            cost = self.calc_insurance_cost()[0]
            props = self.all_inputs['adaptation']['cover_crop']
            self.adap_properties = {
                'type' : 'cover_crop',
                'cost' : cost * props['cost_factor'],
                'N_fixation_min' : props['N_fixation_min'],
                'N_fixation_max' : props['N_fixation_min'], # NOTE: THIS OPTION IS DISABLED FOR NOW
                'climate_dependence' : props['climate_dependence'],
                'adap' : True,
            }
        elif self.adaptation_option == 'both':
            [cost, payout, magnitude] = self.calc_insurance_cost()
            # save these
            self.adap_properties = {
                # general
                'type' : 'both',
                'adap' : True,
                'cost' : cost * self.all_inputs['adaptation']['insurance']['cost_factor'] + \
                            cost * self.all_inputs['adaptation']['cover_crop']['cost_factor'],
                # insurance
                'payout' : payout,
                'magnitude' : magnitude,
                # cover cropping
                'N_fixation_min' : self.all_inputs['adaptation']['cover_crop']['N_fixation_min'],
                'N_fixation_max' : self.all_inputs['adaptation']['cover_crop']['N_fixation_min'], # NOTE: THIS OPTION IS DISABLED FOR NOW
                'climate_dependence' : self.all_inputs['adaptation']['cover_crop']['climate_dependence'],
            }

        else:
            self.adap_properties = {
                'adap' : False,
                'type' : 'none'
            }

        # set burn-in period
        self.adap_properties['burnin_period'] = self.all_inputs['adaptation']['burnin_period']

    def calc_insurance_cost(self):
        '''
        calculate the annual cost for insurance, assuming fair payouts
        and the given coverage (related to crop yields / income)
        '''
        props = self.all_inputs['adaptation']['insurance']
        rains = np.random.normal(self.climate.rain_mu, self.climate.rain_sd, 1000)
        rain_facs = np.full(rains.shape, np.nan)
        for r, rain in enumerate(rains):
            rain_facs[r] = self.land.calculate_rainfall_factor(rain, virtual=True)
        exp_yield = np.mean(rain_facs) * self.land.max_yield * 0.5 # 0.5 assumed mean nutrient factor
        exp_crop_income = exp_yield * self.agents.crop_sell_price # birr/ha
        payout = exp_crop_income * props['payout_magnitude'] # birr/ha
        cost = payout * props['climate_percentile'] # birr/ha
        magnitude = np.percentile(rains, props['climate_percentile']*100)

        return cost, payout, magnitude