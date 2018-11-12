import sys

import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import pandas as pd
import seaborn as sns
from tqdm import tqdm
from copy import deepcopy
from shapely.geometry import Point

sns.set(style='ticks')
plt.style.use('seaborn-ticks')
mpl.rc("font", family="Verdana")


class Shapefile:
    """
    Shapefile object that allows for centroids to be calculated and visualised.
    """
    def __init__(self, meta_name=None):
        self.poly = None
        self.points = None
        self.meta_name = meta_name

    def load_csv(self, loc, county_id='iso3', verbose=False, points=True):
        # Load dataset
        shape_csv = pd.read_csv(loc)
        shape_csv.columns = [x.lower() for x in shape_csv.columns]
        try:
            shape = shape_csv[(shape_csv[county_id] == 'GBR') & (shape_csv['year'] == 2014)]
        except KeyError:
            print('Country identifier missing or incorrectly specified. Data for {} contains all countries.'.format(self.meta_name))

        # Build coordinates column
        shape = self._find_coords(shape)

        # Convert to GeoDataFrame
        shape_geo = gpd.GeoDataFrame(shape, geometry='coordinates')

        if verbose:
            print(shape_geo.head())
        if points:
            self.points = shape_geo
        else:
            self.poly = shape_geo
        print('CSV successfully coerced in a GeoDataFrame!')

    @staticmethod
    def _find_coords(df):
        if 'latitude' in df.columns:
            try:
                df['coordinates'] = list(zip(df.longitude, df.latitude))
            except AttributeError:
                raise AttributeError('Supplied .csv does not contain columns for latitude and longitude. Please reformat.')
        elif 'lat' in df.columns:
            try:
                df['coordinates'] = list(zip(df.lon, df.lat))
            except AttributeError:
                raise AttributeError('Supplied .csv does not contain columns for lat and lon. Please reformat.')
        else:
            raise AttributeError('Coordinates not found in desired format. Please ensure two columns exist, one title longitude, the second titled latitude.')
        df['coordinates'] = df['coordinates'].apply(Point)
        return df

    def load_shapefile(self, loc, verbose=False, shp_type='polygon'):
        """
        Read in the shape file
        """
        shape = gpd.read_file(loc)
        print('Points successfully loaded')
        
        # Decide where to place file
        if shp_type == 'polygon':
            self.poly = shape
        elif shp_type == 'point':
            if not self.point:
                self.points = shape
            elif self.point.shape[0] > 1:
                self.point_set = [self.points, shape]        
        if verbose:
            print(shape.head())

    def plot_shape(self, outname, markers=10):
        """
        Plot the shapefile using geopandas' pyplot wrapper. If a points object exists, this will be overlayed on the shapefile
        """
        fig, ax = plt.subplots(figsize=(50,33))
        ax.set_aspect('equal')
        self.poly.plot(ax=ax)
        self.points.plot(ax=ax, marker='o', color='red', markersize=markers)
        plt.savefig('results/plots/{}'.format(outname))

    def get_centroids(self, outname=None):
        """
        Calculate the centroid of each segment within a shapefile and convert to the original shapefile's project unit. There is an option for this to be written to file.
        """
        points = self.poly.copy()
        points.geometry = points['geometry'].centroid
        points.crs = self.poly.crs
        if outname:
            points.to_file('src/data/shapes/{}'.format(outname))
            print('Points written to disk.')
        self.points = points

    def reproject(self, new_crs, inplace=True, visualise=False):
        old_crs = self.poly.crs
        self.poly_old = deepcopy(self.poly)
        self.poly = self.poly.to_crs({'init': new_crs})
        if visualise:
            fig, ax = plt.subplots(ncols=2, figsize=(50,50))
            [axes.set_aspect('equal') for axes in ax]            
            self.poly_old.plot(ax=ax[0])
            self.poly.plot(ax=ax[1])
            ax[0].set_title('Old Projection: {}'.format(self.poly_old.crs['datum']), fontsize=60, y=1.05)
            ax[1].set_title('New Projection: {}'.format(new_crs), fontsize=60, y=1.05)
            ax[0].xaxis.label.set_fontsize(60)
            for axes in ax:
                for tick in axes.xaxis.get_major_ticks():
                    tick.label.set_fontsize(30)
                for tick in axes.yaxis.get_major_ticks():
                    tick.label.set_fontsize(30) 
            fig.savefig('results/plots/reprojection.png')


class Locator:
    """
    Determine distances and nearest neighbors with a series of outputs.
    """
    def __init__(self, meta_name):
        self.name = meta_name
        self.reference = None
        self.lookup = None
        self.distances = {}

    def loader(self, reference_data, lookup_data):
        if isinstance(reference_data, pd.DataFrame):
            self.reference = reference_data
            print('Reference Data loaded')
        else:
            raise AttributeError('Reference data must be a Pandas DataFrame.') 
        if isinstance(lookup_data, list):
            self.lookup = lookup_data
            print('Lookup Data loaded')
        elif isinstance(lookup_data, pd.Series):
            self.lookup = lookup_data.tolist()
            print('Lookup Data loaded')
        else:
            raise AttributeError('Lookup data must either be a list or Pandas Series.') 
    
    @staticmethod
    def min_and_idx(ref_point, lookups):
        vals = [ref_point.distance(centroid) for centroid in lookups]
        # Get the smallest element's index
        idx = np.argmin(vals)
        coord = lookups[idx]

        # Get the smallest element's distance
        min_val = np.min(vals)
        return(min_val, coord)

    def calculate_distances(self):
        for i, row in self.reference.iterrows():
            coord = (row['coordinates'].x, row['coordinates'].y)
            if coord in self.distances:
                pass
            else:
                self.distances[coord] = self.min_and_idx(row['coordinates'], self.lookup)

    def get_dict(self):
        return self.distances
    
    def get_df(self):
        lon, lat = map(list, zip(*list(self.distances.keys())))
        distance, measuring_point = map(list, zip(*list(self.distances.values())))
        dist_df = pd.DataFrame({
            'r_lon': lon,
            'r_lat': lat,
            'distance': distance,
            'lookup_point': measuring_point
        })
        self.distances_df = dist_df
        return dist_df


class Dataset:
    def __init__(self):
        self.data = None
        self.val_col = None

    def data_load(self, loc, value_col, verbose=False, units=''):
        """
        Load csv file into memory.
        """
        data_in = pd.read_csv(loc)
        data_in = data_in.dropna()
        if verbose:
            print('County Count: {}\nAverage Value: £{} {}\n'.format(data_in.shape[0], np.round(np.mean(data_in[value_col]), 3), units))
        self.data = data_in
        self.val_col = value_col

    def plot_density(self, view=False, filename=None):
        """
        Plot a generic kernel density of the user supplied value column.
        """
        splot = sns.distplot(self.data[self.val_col], rug=True)
        sns.despine()
        fig = splot.get_figure()
        if view:
            fig.show()
        if filename:
            fig.savefig('results/plots/{}'.format(filename))

    def bin_obs(self, n_bins):
        """
        Group observations into n number of bins. Bins size is driven by the number of percentiles specified in the function call.
        """
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
        """
        Rename a specific column, located by the column's soon-to-be former name.
        """
        self.data.rename(columns={old_name: new_name}, inplace=True)
        if value:
            print('Updating known value variable...')
            self.val_col = new_name

    def write_data(self, output_file):
        """
        Write dataset back to disc.
        """
        self.data.to_csv('src/data/cleaned/{}'.format(output_file), index=False, encoding='utf8')


if __name__ == '__main__':
    # counties = Dataset()
    # counties.data_load(loc='src/data/county_gva.csv', value_col='value_billions', verbose=True, units='billion')
    # counties.bin_obs(8)
    # counties.write_data('counties.csv')

    # msoa = Dataset()
    # msoa.data_load(loc='src/data/msoa_household.csv', value_col='Total annual income (£)', verbose=False, units='')
    # msoa.rename_col('Total annual income (£)', 'toal_income', value=True)
    # msoa.bin_obs(8)
    # msoa.write_data('msoa.csv')

    # Define Centroid spatial df
    msoas_shp = Shapefile(meta_name='MSOA Shapefile')
    msoas_shp.load_shapefile(loc='src/data/shapes/Middle_Layer_Super_Output_Areas_December_2011_Super_Generalised_Clipped_Boundaries_in_England_and_Wales.shp', verbose=False)
    msoas_shp.reproject('epsg:4326', visualise=True)
    msoas_shp.get_centroids(outname='msoa_centroids.shp')
    # msoas_shp.plot_shape('msoa_shapes.png')

    # Define WHO AQ monitoring spatial df
    air_quality = Shapefile('WHO AQ Data')
    air_quality.load_csv('src/data/cleaned/aq.csv', verbose=True)
    air_quality.load_shapefile(loc='src/data/shapes/Middle_Layer_Super_Output_Areas_December_2011_Super_Generalised_Clipped_Boundaries_in_England_and_Wales.shp', verbose=False)
    air_quality.reproject('epsg:4326', visualise=False)
    # air_quality.plot_shape('aq_on_msoa.png', 25)

    dists = Locator('Monitoring Proximities')
    dists.loader(air_quality.points, msoas_shp.points.geometry.tolist())
    dists.calculate_distances()
    res = dists.get_df()
    print(res.head())
