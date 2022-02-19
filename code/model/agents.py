import numpy as np
import scipy.stats as stat
import code
import sys

class Agents():
    def __init__(self, inputs):
        # attribute the parameters to the object
        self.all_inputs = inputs
        self.inputs = inputs['agents']
        for key, val in self.inputs.items():
            setattr(self, key, val)

        self.N = self.all_inputs['model']['n_agents']
        self.T = self.all_inputs['model']['T']
        self.id = np.arange(self.N)

        # generate land ownership
        self.land_area = self.init_farm_size()      
        self.crop_production = np.full([self.T, self.N], -9999)

        # wealth (livestock holdings)
        # this represents the START of the year
        self.wealth = np.full([self.T+1, self.N], -9999)
        self.wealth[0] = np.random.normal(self.wealth_init_mean, self.wealth_init_sd, self.N)
        self.wealth[0][self.wealth[0]<0] = 0 # fix any -ve values
        # money
        self.income = np.full([self.T, self.N], -9999)
        self.cash_req = np.random.normal(self.cash_req_mean, self.cash_req_sd, self.N)
        # coping measures
        self.coping_rqd = np.full([self.T, self.N], False)
        self.cant_cope = np.full([self.T, self.N], False)
        # adaptation option decisions
        self.adapt = np.full([self.T+1, self.N], False)

    def init_farm_size(self):
        '''
        initialize agent-level farm size (ha)
        '''
        mult = self.land_area_multiplier
        if self.N == len(self.land_area_init):
            return np.array(self.land_area_init) * mult
        elif self.N % len(self.land_area_init) == 0:
            # equal number of each
            return np.repeat(self.land_area_init, self.N / len(self.land_area_init)) * mult
        else:
            return np.random.choice(self.land_area_init, size=self.N) * mult
        
    def calculate_income(self, land, climate, adap_properties):
        '''
        calculate end-of-year income
        '''
        t = self.t[0]
        [payouts, adap_costs] = self.calc_adap_costs(adap_properties, climate)

        # income = crop_sales - cash_req - adap_costs - fertilizer costs
        self.income[t] = self.crop_sell_price*self.crop_production[t] - self.cash_req - adap_costs
        
        if self.insurance_payout_year:
            # assume that agents first use their payout to neutralize their income
            # and any left over, they use to buy fodder
            # which will increase their maximum wealth capacity
            self.remaining_payout = np.minimum(np.maximum(payouts+self.income[t], 0), payouts) # outer "minimum" is in case their income is +ve --> they can only use the payout for fodder
            self.income[t] += payouts.astype(int)

    def calc_adap_costs(self, adap_properties, climate):
        # costs and payouts for adaptation option
        t = self.t[0]
        adap_costs = np.full(self.N, 0.)
        payouts = np.full(self.N, 0.)
        self.insurance_payout_year = False
        if adap_properties['type'] in ['insurance','both']:
            # costs
            adap_costs[self.adapt[t]] = adap_properties['cost'] * self.land_area[self.adapt[t]]
            # payouts
            if climate.rain[t] < adap_properties['magnitude']:
                payouts[self.adapt[t]] = adap_properties['payout'] * self.land_area[self.adapt[t]]
                self.insurance_payout_year = True
        if adap_properties['type'] in ['cover_crop','both']:
            # note: with "both", the cost parameter includes the cost of both options. it's input twice and written over here, so we are not double counting
            adap_costs[self.adapt[t]] = adap_properties['cost'] * self.land_area[self.adapt[t]]

        return payouts, adap_costs

    def coping_measures(self, land):
        '''
        calculate end-of-year income balance
        and simulate coping measures
        '''
        t = self.t[0]
        # assume those with -ve income are required to engage in coping measure
        self.coping_rqd[t, self.income[t] < 0] = True
        # add (or subtract) agent income to their wealth
        # this proxies the effect of buying (+ve income) or selling (-ve income) livestock
        self.wealth[t+1] = self.wealth[t] + self.income[t]
        # record agents with -ve wealth (not able to cope)
        self.cant_cope[t, self.wealth[t+1] < 0] = True
        
        # wealth (/livestock) constraints: can't carry more than your crop residues allows
        # if 80% of livestock must be grazed on fodder, then the maximum wealth you can carry
        # is 20% of your current livestock herds + whatever you can sustain from your crop residues
        # i.e. it's assumed that some fraction of your livestock are fully independent of crop residue
        # rather than all livestock requiring this fraction of feed from fodder
        # if wealth_constraint==False, HHs can always buy fodder so there is no upper limit
        if self.fodder_constraint:
            max_ls_fodder = self.crop_production[t] * land.residue_multiplier * land.residue_loss_factor / \
                    (land.livestock_residue_factor) # TLU = kgCrop * kgDM/kgCrop / kgDM/TLU
            max_ls_wealth_tot = max_ls_fodder*self.livestock_cost + (1-land.livestock_frac_crops) * self.wealth[t]

            if self.insurance_payout_year:
                # assume that any leftover income from the insurance payout is converted to livestock/wealth
                max_ls_wealth_tot += self.remaining_payout
        
            too_much = self.wealth[t+1] > max_ls_wealth_tot
            self.wealth[t+1, too_much] = max_ls_wealth_tot[too_much]
        
        self.wealth[t+1, self.wealth[t+1] < self.max_neg_wealth] = self.max_neg_wealth

    def adaptation(self, land, adap_properties):
        '''
        simulate adaption decision-making
        assume there is a burn-in period before any adaptation option comes into effect
        this is because there's some dependence on the initial condition / starting wealth value
        '''
        t = self.t[0]
        if adap_properties['adap'] and (t >= adap_properties['burnin_period']):
            if self.adap_type == 'coping':
                # agents engage in the adaptation option next period
                # if they had to cope this period
                self.adapt[t+1, self.coping_rqd[t]] = True
            elif self.adap_type == 'switching':
                # agents SWITCH adaptation types if they had to cope in this period
                self.adapt[t+1, ~self.coping_rqd[t]] = self.adapt[t, ~self.coping_rqd[t]]
                self.adapt[t+1, self.coping_rqd[t]] = ~self.adapt[t, self.coping_rqd[t]]
            elif self.adap_type == 'affording':
                # agents adapt if they can afford it
                afford = self.wealth[t+1] >= (adap_properties['cost'] * self.land_area)
                self.adapt[t+1, afford] = True
            elif self.adap_type == 'always':
                # all agents adapt
                self.adapt[t+1] = True
            else:
                print('ERROR: unrecognized adaptation type')
                sys.exit()