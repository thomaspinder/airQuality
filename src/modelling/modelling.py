import sys

import fiona
import matplotlib as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm

sns.set(style='ticks')


class Shapefile:
    def __init__(self):
        self.bounds = None

    def load_shapefile(self, loc, verbose=False):
        shape = fiona.open(loc)
        self.bounds = shape.bounds
        if verbose:
            print(shape.schema)
        shape.close()

class Dataset:
    def __init__(self):
        self.data = None
        self.val_col = None

    def data_load(self, loc, value_col, verbose=False, units=''):
        data_in = pd.read_csv(loc)
        data_in = data_in.dropna()
        if verbose:
            print('County Count: {}\nAverage Value: £{} {}\n'.format(data_in.shape[0], np.round(np.mean(data_in[value_col]), 3), units))
        self.data = data_in
        self.val_col = value_col

    def plot_density(self, view=False, filename=None):
        splot = sns.distplot(self.data[self.val_col], rug=True)
        sns.despine()
        fig = splot.get_figure()
        if view:
            fig.show()
        if filename:
            fig.savefig('results/plots/{}'.format(filename))

    def bin_obs(self, n_bins):
        splitter = [(100.0/n_bins)*decomp for decomp in np.arange(n_bins)]
        percs = list(reversed(np.percentile(self.data[self.val_col], splitter)))
        groups = []
        for val in tqdm(self.data[self.val_col], desc='Assigning Groups'):
            for idx, perc in enumerate(percs):
                if val > perc-0.1:
                    groups.append(idx+1)
                    break
                else:
                    pass
        self.data['group'] = groups
        print('Observations grouped into {} bins.\n'.format(n_bins))

    def rename_col(self, old_name, new_name, value=False):
        self.data.rename(columns={old_name: new_name}, inplace=True)
        if value:
            print('Updating known value variable...')
            self.val_col = new_name

    def write_data(self, output_file):
        self.data.to_csv('src/data/cleaned/{}'.format(output_file), index=False, encoding='utf8')


if __name__ == '__main__':
    counties = Dataset()
    counties.data_load(loc='src/data/county_gva.csv', value_col='value_billions', verbose=True, units='billion')
    counties.bin_obs(8)
    counties.write_data('counties.csv')

    msoa = Dataset()
    msoa.data_load(loc='src/data/msoa_household.csv', value_col='Total annual income (£)', verbose=False, units='')
    msoa.rename_col('Total annual income (£)', 'toal_income', value=True)
    msoa.bin_obs(8)
    msoa.write_data('msoa.csv')

    msoas = Shapefile()
    msoas.load_shapefile(loc='src/data/shapes/Middle_Layer_Super_Output_Areas_December_2011_Super_Generalised_Clipped_Boundaries_in_England_and_Wales.shp', verbose=True)
