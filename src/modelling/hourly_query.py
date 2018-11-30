import argparse
import iris
from iris.time import PartialDateTime
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.basemap import Basemap
from matplotlib.ticker import MaxNLocator
import cf_units
import imageio
import glob


def iris_loader(cube_dir):
    cubes = iris.load(cube_dir)
    if len(cubes) == 1:
        return cubes[0]
    else:
        return cubes
    
def dt_to_integer(dt_time):
    return 1000000*dt_time.year + 10000*dt_time.month + 100*dt_time.day + dt_time.hour


def cube_plot(cube, extras):
    basemap = extras['bmap']
    lims = extras['lims']
    # Get the cube's timestamp
    full_timestamp = cube.coord('time').units.num2date(cube.coord('time').points[0])
    timestamp = str(full_timestamp).replace(' ', '__').replace('-', '_')[:-6]
    
    # coerce data
    y = cube.coord('latitude').points
    x = cube.coord('longitude').points
    data = cube.data.data
    x, y = np.meshgrid(x,y)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 20))
    levels = MaxNLocator(nbins=250).tick_values(lims[0], lims[1])
    print(basemap)
    print(x.shape)
    print(y.shape)
    print(data.shape)
    pms = basemap.contourf(x, y, data, latlon=True, cmap='hot_r', levels=levels)
    
    # Overlay coastlines
    basemap.drawcoastlines(linewidth = 0.5)
    
    # Add extras
    cbar = fig.colorbar(pms, orientation='horizontal', pad=0.02)
    cbar.set_label('ug/m3 PM2.5 level')
    cbar.set_clim(lims[0], lims[1])
    plt.title('PM2.5 Levels: {}'.format(full_timestamp))
    plt.savefig('results/heatmaps/static/maximum_pm_{}.png'.format(timestamp))


def slicer(cube, slice_list, function, n=None, extras=None):
    j = 0
    for i in cube.slices(slice_list):
        if n:
            if j > n:
                break
            else:
                function(i, extras)
                j += 1
        else:
            function(cube)


def gif_plotter(filenames, outname):
    with imageio.get_writer(outname, mode='I', duration=0.4) as writer:
        for filename in sorted(filenames):
            image = imageio.imread(filename)
            writer.append_data(image)


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-y', '--year', default=2014, type=int, help='Year to extract from cube')
    parser.add_argument('-h', '--hours', default=24, help='Number of hours to cycle through', type=int)
    parser.add_argument('-l', '--ldate', default=1, help='Date(ddmm) the series should start at.', type=int)
    parser.add_argument('-f', '--fdate', default=1, help='Date(ddmm) the series should finish at.', type=int)
    args = parser.parse_args()
    return args


def cleaner(cube):
    n = len(cube.coord('time').points)
    t_unit = cf_units.Unit('hours since {}-01-01 00:00:00'.format(dataset_year), calendar='gregorian')
    rt = iris.coords.DimCoord(range(n), 'time', units=t_unit)
    cube.remove_coord('time')
    cube.add_dim_coord(rt, 0)
    iris.util.demote_dim_coord_to_aux_coord(cube, 'time')
    uk_constraint = iris.Constraint(coord_values={'latitude':lambda cell: 49.7 < cell < 61,
                                                'longitude': lambda cell: -10.651 < cell < 1.87})
    cube_uk = cube.extract(uk_constraint)
    return cube_uk


def constrain(cube, lower, upper):
    lm = lower[]
    # TODO: parse dates, slice cube out on selected date range and return data at these time points

if __name__ == '__main__':
    args = build_parser()
    # Data sourced from http://www.regional.atmosphere.copernicus.eu/index.php?category=data_access&subensemble=reanalysis_products
    cube = iris_loader('src/data/cleaned/pm25_2014_hourly.nc')
    dataset_year = args.year
    cube_uk = cleaner(cube)

    lim = iris.time.PartialDateTime(2014, 1, 1, 1)
    year_constraint = iris.Constraint(coord_values={'time':lambda cell: cell < lim})
    first = cube_uk.extract(year_constraint)

    uk_map = Basemap(projection='merc', lat_0 =  53.866772, lon_0 = -5.23636,resolution = 'i', area_thresh = 0.05,
                    llcrnrlon=-10.65073, llcrnrlat=49.85,urcrnrlon=1.76334, urcrnrlat=60.8)
    j = 0
    ulim = np.ceil(np.amax(cube_uk.data.data)).astype(int)
    llim = np.floor(np.amin(cube_uk.data.data)).astype(int)
    func_args = {'bmap': uk_map, 'lims': (llim, ulim)}

    slicer(cube_uk, slice_list=['latitude', 'longitude'], function=cube_plot, n=args.hours, extras=func_args)
    gif_plotter(glob.glob('results/heatmaps/static/*.png'), 'results/heatmaps/gifs/one_day.gif')
