#!/usr/bin/env python
# coding: utf-8

# In[1]:
print('here loading')


# In[2]:


# IMPORT
import numpy as np
import numpy.ma as ma
print('numpy')
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.ticker import NullFormatter
print('matplot lib')
from netCDF4 import Dataset
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
print('xarray')
import xarray as xr
#import glob
print('datetime')
from datetime import datetime, timedelta
from numba import jit
from obspy.imaging.cm import viridis_white_r
print('past packages')

@jit()
def rms(true,prediction):
    return np.sqrt(np.nanmean((prediction-true)**2))

@jit()
def mean_error(true,prediction):
    return np.nanmean(prediction - true)

@jit()
def rms_tendency(variable,hours):
    variable_tendency = np.zeros((hours))
    variable = np.exp(variable) * 1000.0
    for i in range(hours):
        variable_tendency[i] = np.sqrt(np.mean((variable[i+1] - variable[i])**2.0))
    return variable_tendency

def latituded_weighted_rmse(true,prediction,lats):
    diff = prediction-true
    weights = np.cos(np.deg2rad(lats))
    weights2d = np.zeros(np.shape(diff))
    diff_squared = diff**2.0
    #weights = np.ones((10,96))
    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)
    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
    weighted_average = np.ma.average(masked,weights=weights2d)
    return np.sqrt(weighted_average)

def latituded_weighted_bias(true,prediction,lats):
    diff = prediction-true
#     print(diff[0])
    weights = np.cos(np.deg2rad(lats))
#     print(weights[0])
    weights2d = np.zeros(np.shape(diff))
    diff_squared = diff
    #weights = np.ones((10,96))
    weights2d = np.tile(weights,(96,1))
    weights2d = np.transpose(weights2d)
#     print(weights2d)
    masked = np.ma.MaskedArray(diff_squared, mask=np.isnan(diff_squared))
#     print(masked[0])
    weighted_average = np.ma.average(masked,weights=weights2d)
#     print(weighted_average)
    return weighted_average



# In[103]:


## MAIN SCRIPT
## **** CHANGE FILES AND DATES HERE

def rmse_time_series_plot_and_maps(level_in_speedy,variable_speedy):
    # Define: Initial FILES, dates, Variable, and Level desired

    analysis_file_speedy = '/scratch/user/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/speedy_1_9_20110101_20120101/mean_output/out.nc'    
    #analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_1_9_20110101_20120101/mean_output/out.nc' #speedy_1_9_uniform_20110101_20110501/mean.nc'
    
    '''
    
    
    ANAL
    
    
    '''
    ### ERA5  
    
     
    analysis_file ='/scratch/user/dylanelliott/letkf-hybrid-speedy-from-skynet-2/DATA/uniform_letkf_analysis/ERA5_hybrid_1_9_20110101_20120101/mean_output/out.nc'

    #analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/ERA5_hybrid_1_9_20110101_20120101/mean_output/out.nc' #uniform_letkf_anal_older_stuff/ERA5_1_9/mean_output/out.nc'
    
    ## hybrid 1.9,1.9
    
    hybrid_1_9_1_9_file ='/scratch/user/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/hybrid_1_9_1_9_mem_1_fixed_20110101_20120115/out.nc' 
   # hybrid_1_9_1_9_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_9_mem_1_fixed_20110101_20120115/out.nc'

    #### ACTUALLY 1.9,1.9,1.7
    
    hybrid_1_9_1_9_1_9_file = '/scratch/user/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/hybrid_1_9_1_9_1_7_20110101_20120101/mean_output/out.nc' 
    #hybrid_1_9_1_9_1_9_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_9_1_7_20110101_20120101/mean_output/out.nc'#hybrid_1_9_1_9_1_9_2nd_inter_20110101_20110924/mean_output/out.nc'
    ###### 1.7 ########
    
#     spread_file_ERA5 = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal_older_stuff/ERA5_1_3_6hr_timestep_1_24_24_20110101_20120101/sprd_output/out.nc'
    
    
#     spread_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_cov_1_3_40MEM_individual_ens_member_20110101_20110601/sprd_output/out.nc'
# #     spread_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/40member_speedy_covar1_3_20110101_20120901/sprd.nc' 
    
#     # SPREAD FILES
#     ds_spread_ERA5 = xr.open_dataset(spread_file_ERA5)
#     ds_spread_speedy = xr.open_dataset(spread_file_speedy)
    

    start_year = 2011
    end_year = 2011

    startdate = datetime(2011,1,1,0)
    enddate = datetime(2011,12,31,18)
    time_slice = slice(startdate,enddate)

    #level = 0.95 #0.2#0.95#0.51
    #level_era = 7 #2#7 #4
    
    level = level_in_speedy
    if level_in_speedy == .95:
        level_in_era = 7
    if level_in_speedy == .2:
        level_in_era = 2
    if level_in_speedy == .51:
        level_in_era = 4
        
    level_era = level_in_era

    var_da = variable_speedy
    if variable_speedy == 'q':
        variable_era = 'Specific_Humidity'
    if variable_speedy == 't':
        variable_era = 'Temperature'
    if variable_speedy == 'v':
        variable_era = 'V-wind'
    if variable_speedy == 'u':
        variable_era = 'U-wind'
    if variable_speedy == 'ps':
        variable_era = 'logp'
        
    
#     var_era = variable_era
    print(variable_era)
    var_era = variable_era

    #var_era = 'V-wind'#'Specific_Humidity'#'Temperature' #'V-wind'
    #var_da =  'v'#'q'#'t'#'v'
    print('you selected for variable =',var_era)
    print('at level =',level)
    


    # create empty list to store indiviudal datasets
    era5sets = []
    print('made it to the for loop...')

    # LOAD DATA HERE 
    print('LOADING DATA...')
    # FOR ERA5
    timestep_6hrly = 6
    # loop over the range of years and open each ds
    for year in range(start_year, end_year + 1):
        nature_file = f'/scratch/user/troyarcomano/ERA_5/{year}/era_5_y{year}_regridded_mpi_fixed_var_gcc.nc'
       #scratch/user/troyarcomano/ERA_5/2011 
	# only load var_era selected and only load level_era selected from above
        if variable_speedy == 'ps': # don't select level if variable is 'ps'
            ds_nature = xr.open_dataset(nature_file)[var_era]
        else:
            ds_nature = xr.open_dataset(nature_file)[var_era].sel(Sigma_Level=level_era)
        # Read in every 6th timestep
        ds_nature = ds_nature.isel(Timestep=slice(None, None, timestep_6hrly))
        era5sets.append(ds_nature)

    print('Now its concatinating them all together...')

    ds_nature = xr.concat(era5sets, dim = 'Timestep')
    ds_nature = ds_nature.sortby('Timestep')
    print('Done concat and sortby Timestep...')
#     temp_500_nature = ds_nature.values

    # if surface pressure variable selected, then don't select a level. Theres only one for surface pressure..
    if var_era == 'logp':
        #convert to hPa for era5 ds
        temp_500_nature = np.exp(ds_nature.values) * 1000.0
        #convert to hPa for letkf analysis                         
        ds_analysis_mean = xr.open_dataset(analysis_file)[var_da].sel(time=time_slice) / 100.0
        ds_analysis_mean_speedy = xr.open_dataset(analysis_file_speedy)[var_da].sel(time=time_slice).values / 100.0
        ds_hybrid_1_9_1_9 = xr.open_dataset(hybrid_1_9_1_9_file)[var_da].sel(time=time_slice).values / 100.0
    else:     
        temp_500_nature = ds_nature.values
        ds_analysis_mean = xr.open_dataset(analysis_file)[var_da].sel(lev=level,time=time_slice)
        ds_analysis_mean_speedy = xr.open_dataset(analysis_file_speedy)[var_da].sel(lev=level,time=time_slice)
        ds_hybrid_1_9_1_9 = xr.open_dataset(hybrid_1_9_1_9_file)[var_da].sel(lev=level,time=time_slice).values

    temp_500_analysis = ds_analysis_mean
    # temp_500_analysis = ds_analysis_mean[var_da].sel(lev=level).values
    temp_500_analysis_speedy = ds_analysis_mean_speedy
    # temp_500_analysis_speedy = ds_analysis_mean_speedy[var_da].sel(lev=level,time=time_slice).values
    
    
#     temp_500_spread_era5 = ds_spread_ERA5[var_da].sel(lev=level).values
#     temp_500_spread_speedy = ds_spread_speedy[var_da].sel(lev=level).values
    


    print('era5 shape = ',np.shape(temp_500_nature))
    print('speedy shape = ',np.shape(temp_500_analysis_speedy))
    print('hybrid shape = ',np.shape(temp_500_analysis))

    #find smallest index value to set that as the "length"
    speedy_index = temp_500_analysis_speedy.shape[0]
    nature_index = temp_500_nature.shape[0]
    hybrid_index = temp_500_analysis.shape[0]
    smallest_index = min(speedy_index,nature_index,hybrid_index)

    if smallest_index == speedy_index:
        length = speedy_index #- 1
    elif smallest_index == nature_index:
        length = nature_index
    else:
        length = hybrid_index
    print('the smallest length is',length)
    
    #ps_nature = ds_nature['logp'].values
    #ps_nature = 1000.0 * np.exp(ps_nature)
    #ps_analysis = ds_analysis_mean['ps'].values/100.0

    xgrid = 96
    ygrid = 48
    #length =365*4*2 #1952-7 # 240 for 3 months  #1450 ##338 #160#64#177#1400#455

    analysis_rmse = np.zeros((length))
    analysis_rmse_speedy = np.zeros((length))
    global_average_ensemble_spread_era5 = np.zeros((length))
    global_average_ensemble_spread_speedy = np.zeros((length))
    
    hybrid_1_9_1_9_anal_rmse = np.zeros((length))
    #ps_rmse = np.zeros((length))

    analysis_error = np.zeros((length,ygrid,xgrid))
    analysis_error_speedy = np.zeros((length,ygrid,xgrid))
    
    hybrid_1_9_1_9_anal_error = np.zeros((length,ygrid,xgrid))
    
    analysis_bias = np.zeros((length))
    analysis_bias_speedy = np.zeros((length))

    hybrid_1_9_1_9_bias = np.zeros((length))
    
    
    print(np.shape(analysis_error))
    print(np.shape(analysis_error_speedy))
    
#     print('Test unit check:')
#     print(ds_analysis_mean[0])

    print('Now its calculating analysis RMSE...')
    lats = ds_nature.Lat
    
    
    for i in range(length):
        # TIME AVERAGED ERROR
        analysis_rmse[i] = latituded_weighted_rmse(temp_500_nature[i,:,:],temp_500_analysis[i,:,:],lats)
        analysis_rmse_speedy[i] = latituded_weighted_rmse(temp_500_nature[i,:,:],temp_500_analysis_speedy[i,:,:],lats)
        #ps_rmse[i] = rms(ps_nature[i*6,:,:],ps_analysis[i,:,:])
        hybrid_1_9_1_9_anal_rmse[i] = latituded_weighted_rmse(temp_500_nature[i,:,:], ds_hybrid_1_9_1_9[i,:,:],lats)
        
        # ERROR BY GRIDPOINT
        analysis_error[i,:,:] = temp_500_analysis[i,:,:] - temp_500_nature[i,:,:]
        analysis_error_speedy[i,:,:] = temp_500_analysis_speedy[i,:,:] - temp_500_nature[i,:,:]
        hybrid_1_9_1_9_anal_error[i,:,:] = ds_hybrid_1_9_1_9[i,:,:] - temp_500_nature[i,:,:]
        
        # BIAS FOR MAPS 
        analysis_bias[i] = latituded_weighted_bias(temp_500_nature[i,:,:],temp_500_analysis[i,:,:],lats)
        analysis_bias_speedy[i] = latituded_weighted_bias(temp_500_nature[i,:,:],temp_500_analysis_speedy[i,:,:],lats)
        hybrid_1_9_1_9_bias[i] = latituded_weighted_bias(temp_500_nature[i,:,:],ds_hybrid_1_9_1_9[i,:,:],lats)
        
#         global_average_ensemble_spread_era5[i] = np.average(temp_500_spread_era5[i,:,:])
#         global_average_ensemble_spread_speedy[i] = np.average(temp_500_spread_speedy[i,:,:])

    # print('mean analysis_rmse = ',analysis_rmse)

    print('DONE CALCULATING ERROR AT EVERY GRIDPOINT AT EVERY TIMESTEP.')
    
    ############################
    # LOAD TROYS MEAN ANAL

    # Define the base path for the files
    # base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/40member_hybrid_20110101_2011052906/anal'
    # base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/era5_28_yr_trained_weights/'

    ### ERA 5 crash test
#     base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/troy_test_speedy_trained_12monthrun/'
#     troy_anal_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/troy_test_speedy_trained_12monthrun/mean_output/out.nc'
    ### ERA5 1 year run worked path
#     base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/ERA5_weights_sourcecodeupdated_1_20_24/'
    
    #### HYBRID NEW 1_3_1_3
#     hybrid_base_path = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_3_1_3_original_weights_20110101_20120101/'

    # Define the variable name, level, and time slice
    start_year = 2011
    end_year = 2011

    # startdate = datetime(2011,1,1,0)
    # enddate = datetime(2011,6,1,0)
    time_slice = slice(startdate,enddate)

    # level = 0.95 #0.2#0.95#0.51
    # level_era = 7 #2#7 #4

    # var_era = 'Temperature'#'Specific_Humidity'#'Temperature' #'V-wind'
    # var_da =  't'#'q'#'t'#'v'
    print('you selected for variable =',var_era)
    print('at level =',level)
    
    ## ADDING MEAN of Hybrid
#     path_mean_anal_hybrid_retest = hybrid_base_path + 'mean_output/out.nc'
    
#     path_mean_anal_hybrid_retest = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_3_mem1_fixed_20110101_20120501/mean_output/out.nc'
    path_mean_anal_hybrid_retest = hybrid_1_9_1_9_1_9_file


    if variable_speedy == 'ps':
        ds_mean_anal_hybrid_1_9_1_9_1_9 = xr.open_dataset(path_mean_anal_hybrid_retest)[var_da].sel(time=time_slice) / 100.0
    else:    
        ds_mean_anal_hybrid_1_9_1_9_1_9 = xr.open_dataset(path_mean_anal_hybrid_retest)[var_da].sel(lev=level, time=time_slice)

    # SPREAD OF HYBRID
#     spread_file_hybrid = hybrid_base_path + 'sprd_output/out.nc'
#     ds_spread_hybrid = xr.open_dataset(spread_file_hybrid)
#     temp_500_spread_hybrid = ds_spread_hybrid[var_da].sel(lev=level).values
#     global_average_ensemble_spread_hybrid = np.zeros((length))
    
#     # Create an empty list to store the datasets
#     ds_list = []

#     # ens member list
#     ens_member_list = range(1,40+1)
# #     print("opening all files...")
#     # Loop through the member numbers and read in the corresponding files
#     for member_number in ens_member_list:
#         file_path = f'{base_path}/{member_number:03d}_output/out.nc'
#         # file_path = f'{base_path}/{member_number:03d}_output/{member_number:03d}.nc'
#         ds = xr.open_dataset(file_path)[var_da].sel(lev=level, time=time_slice)
#         ds_list.append(ds)
#     print('shape test =', np.shape(ds_list))
    
    
    
#     ds_anal_troy = xr.open_dataset(troy_anal_path)[var_da].sel(lev=level, time=time_slice)
#     print('shape test =', np.shape(ds_anal_troy))
    
    # print('ds_list[0] =',ds_list[0])

    # Assign each element in ds_list to be called ds_member_{i}
    # for i, ds in enumerate(ds_list, start=1):
    #     globals()[f'ds_member_{i}'] = ds

    # print(ds_member_1) 

    lats = ds_nature.Lat

    # analysis_error is for maps, see its a 3d array, 
    # MAKING analysis_rmse now

    print('MAKING zeros arrays..')
#     quantity_of_ens_members = 40
#     analysis_rmse_object = np.zeros((quantity_of_ens_members, length))
    
#     anal_rmse_troy = np.zeros((length))
#     anal_error_troy = np.zeros((length,ygrid,xgrid))
    anal_mean_error_hybrid_1_9_1_9_1_9 = np.zeros((length,ygrid,xgrid))
    anal_mean_rmse_hybrid_1_9_1_9_1_9 = np.zeros((length))
    
    anal_bias_hybrid_1_9_1_9_1_9 = np.zeros((length))
    # check shape, yes they are equal and vibing
    # print('new = ',np.shape(analysis_rmse_object[0]))
    # print('old = ', np.shape(analysis_rmse_1))

#     print('looping through each ERA5 ens_member at every timestep..')
#     counter = 0
#     for member_number in range(0,40):
#         # loop through each timestep
#         counter = counter + 1
#         print('MEM counter = ',counter)
#         for l in range(length):
#     #     for l in range(length):
#             analysis_rmse_object[member_number,l] = latituded_weighted_rmse(temp_500_nature[l,:,:],ds_list[member_number][l,:,:],lats)
    
        
    ##### and calc for mem 1 with length == 3:
    # analysis_rmse_mem1 = latituded_weighted_rmse(temp_500_nature[3,:,:],ds_list[1][3,:,:],lats)
    # print(analysis_rmse_mem1)
    #####

    # print('analysis_rmse_object[0] =',analysis_rmse_object[0])
    # print('analysis_rmse_object[39] =',analysis_rmse_object[39])    
    print('Calc hybrid analysis_error and bias')
    
    for i in range(length):
        
        anal_mean_error_hybrid_1_9_1_9_1_9[i,:,:] = ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:] - temp_500_nature[i,:,:]
        anal_mean_rmse_hybrid_1_9_1_9_1_9[i] = latituded_weighted_rmse(temp_500_nature[i,:,:], ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:],lats)
        anal_bias_hybrid_1_9_1_9_1_9[i] = latituded_weighted_bias(temp_500_nature[i,:,:],ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:],lats)
        
    print('Done with speedy trained hybrid analysis.')
    
#     # take average of anal_bias_hybrid_1_9_1_9_1_9
#     global_mean_anal_bias_hybrid_1_9_1_9_1_9 = np.average(anal_bias_hybrid_1_9_1_9_1_9[24::])
    
#     print('global mean bias of speedy trained hybrid analysis = ', global_mean_anal_bias_hybrid_1_9_1_9_1_9)
    
#     # take avg of speedy_anal_bias 
#     global_mean_anal_bias_speedy = np.average(analysis_bias_speedy[24::])
#     print('global mean bias of speedy analysis = ', global_mean_anal_bias_speedy)
    
#     if var_era == 'Specific_Humidity': #convert to g/kg
#         global_mean_anal_bias_hybrid_1_9_1_9_1_9 = global_mean_anal_bias_hybrid_1_9_1_9_1_9*1000
#         global_mean_anal_bias_speedy = global_mean_anal_bias_speedy*1000
        
    
    
    #########################################################
    # MAKE MAP FROM ABOVE 

    ''' 24(below) instead of 28 to cut transient event (ML spin up) out in first few weeks '''
    ####### WHICH AVGERAGE ERROR DO YOU WANT?? I want the analysis error of the mean ERA5
    if var_era == 'Temperature':
        units='(K)'
        units_nopar='K'
    if var_era == 'Specific_Humidity':
        units='(g/kg)'
        units_nopar ='g/kg'
    if var_era == 'V-wind':
        units='(m/s)'
        units_nopar='m/s'
    if var_era == 'U-wind':
        units='(m/s)'
        units_nopar='m/s' 
    if var_era == 'logp':
        units='(hPa)' # converted above
        units_nopar='hPa' 
    print(units)
    if level == .95:
#         title_level = 'Low Level '
        title_level = '.95 Sigma Level '
    if level == .2:
#         title_level = '200 hPa '
        title_level = '.2 Sigma Level '
    if level == .51: 
        title_level = '.51 Sigma Level '
    print(title_level)
    if var_era == 'Specific_Humidity':
        title_var_era = 'Specific Humidity'
    if var_era == 'V-wind':
        title_var_era = "Meridional Wind"
    if var_era == 'Temperature':
        title_var_era = 'Temperature'
    if var_era == 'U-wind':
        title_var_era = 'Zonal Wind'
    if var_era == 'logp':
        title_var_era = 'Surface Pressure'
    print(title_var_era)
    
#     obs_network_file = '/skydata2/troyarcomano/letkf-hybrid-speedy/obs/networks/uniform.txt'
#     network_ij = np.loadtxt(obs_network_file,skiprows=2,dtype=int)
#     network_ij = network_ij - 1
    
    
    #### TAKING OUT ABSOLUTE VALUE 
#     averaged_error = np.average(abs(analysis_error[24::,:,:]),axis=0)
    
    averaged_error = np.average((analysis_error[24::,:,:]),axis=0)
    averaged_error_speedy = np.average((analysis_error_speedy[24::,:,:]),axis=0)
    
  
    # SeeeEEeee no abs value taken so its the bias
    
    lat = ds_analysis_mean.lat.values
    lon = ds_analysis_mean.lon.values  
#     lons2d, lats2d = np.meshgrid(lon,lat)

    print('Now plotting and meshing...')
    
    # data for plot Hybrid and Speedy Map
    if var_era == 'Specific_Humidity':
        cyclic_data, cyclic_lons = add_cyclic_point(averaged_error*1000, coord=lon)
        cyclic_data_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy*1000, coord=lon)
    else: 
        cyclic_data, cyclic_lons = add_cyclic_point(averaged_error, coord=lon)
        cyclic_data_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy, coord=lon)
    
    # data for plot difference 
    diff = averaged_error - averaged_error_speedy
    if var_era == 'Specific_Humidity':
        cyclic_data_diff, cyclic_lons = add_cyclic_point(diff*1000, coord=lon)
    else:
        cyclic_data_diff, cyclic_lons = add_cyclic_point(diff, coord=lon)
        
#     print('cyclic_data ', cyclic_data, np.shape(cyclic_data))
#     print('cyclic_lons ', cyclic_lons, np.shape(cyclic_lons))
    
    lons2d,lats2d = np.meshgrid(cyclic_lons,lat)

#     lons2d, lats2d = np.meshgrid(lon,lat)
    
    ## SET VMIN AND VMAX using old code
    if level != .2 and var_era == 'Temperature':
        adapted_range = np.arange(-5.05,5.05,.05)
        adapted_difference_range = np.arange(-5,5,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)

    if level == .2 and var_era == 'Temperature':
#         adapted_range = np.arange(0,.1,.001)
#         adapted_difference_range = np.arange(-.05,.05,.001)
        adapted_range = np.arange(-5.05,5.05,.05)
        adapted_difference_range = np.arange(-5,5,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
        
    if level != .2 and var_era == 'Specific_Humidity':
        adapted_range = np.arange(-3,3,.05)
        adapted_difference_range = np.arange(-2,2,.001)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
        
    if level == .2 and var_era == 'Specific_Humidity':
        adapted_range = np.arange(-.1,.1,.001)
        adapted_difference_range = np.arange(-.05,.05,.001)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level == .2 and var_era == 'V-wind':
        adapted_range = np.arange(-10.05,10.05,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level != .2 and var_era == 'V-wind':
        adapted_range = np.arange(-5,5,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level != .2 and var_era == 'U-wind':
        adapted_range = np.arange(-10.05,10.05,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)
    if level == .2 and var_era == 'U-wind':
        adapted_range = np.arange(-10.05,10.05,.05)
        adapted_difference_range = np.arange(-2,2,.05)
        range_min = min(adapted_range)
        range_max = max(adapted_range)
        diff_min = min(adapted_difference_range)
        diff_max = max(adapted_difference_range)

    
    
    averaged_error_speedyhybrid = np.average((anal_mean_error_hybrid_1_9_1_9_1_9[24::,:,:]),axis=0) # this is average at each grid point. all I need to do is subtract 1 number from all these
    

    
    # Now plot speedy but first calculate
    
    #bias
#     analysis_error_speedy[i,:,:] = temp_500_analysis_speedy[i,:,:] - temp_500_nature[i,:,:]

    
#     print('bias squared and in g/kg ..\n', squared_bias_g_kg)
    
#     print(np.shape(squared_bias_g_kg))
    #average all the squared differences through time
    if variable_speedy == 'q':
        MSE_speedy = np.average((((analysis_error_speedy[24::,:,:])*1000)**2.0),axis=0)
    else: 
        MSE_speedy = np.average((((analysis_error_speedy[24::,:,:]))**2.0),axis=0)
    
#     print("MSE speedy \n", MSE_speedy)
    # make points for plotting using cyclic point
    cyclic_data_MSE_speedy, cyclic_lons = add_cyclic_point(MSE_speedy, coord=lon)
    
#     print('cyclic_data_MSE \n', cyclic_data_MSE_speedy)
    
    # take an average of the biases
    averaged_error_speedy = np.average((analysis_error_speedy[24::,:,:]),axis=0)
    
#     if variable_speedy == 'q':
#         bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 
#     # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
#     else:
#         bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0 # g/kg
        
    cyclic_data_bias_speedy, cyclic_lons = add_cyclic_point(averaged_error_speedy, coord=lon)
    # square each bias value
    if variable_speedy == 'q':
        Bias_squared_speedy = (cyclic_data_bias_speedy*1000)**2.0
#         Bias_squared_speedy = ((np.average()))
    else:
        Bias_squared_speedy = cyclic_data_bias_speedy**2.0
    #calculate variance
    Variance_speedy = cyclic_data_MSE_speedy - Bias_squared_speedy
    #calculate std devia
    standard_deviation_speedy = np.sqrt(Variance_speedy)
    
    # Plot of Mean Square Error, Bias**2, and Variance of Speedy trained Hybrid
#     print('variable_speedy = ', variable_speedy)
#     print('type of variable_speedy == ', type(variable_speedy))
#     print('type of level_in_speedy == ', type(level_in_speedy))
    
    if variable_speedy == 'q': # type ===== string
        if level_in_speedy == .95: # type === float 
            imshow_v_max = 8 
        elif level_in_speedy == .51:
            imshow_v_max = 4
        elif level_in_speedy == .2:
            imshow_v_max = .005
    elif variable_speedy == 'v':
        imshow_v_max = 30
    elif variable_speedy == 'u':
        imshow_v_max = 30
    elif variable_speedy == 't':
        imshow_v_max = 16
    elif variable_speedy == 'ps':
        imshow_v_max = 100
        
#     print('imshow_v_max = ', imshow_v_max)
    imshowsquared_units = units_nopar + '²'
#     print('imshowsquared_units ', imshowsquared_units)
    #hPa"R\u00b2 score EO: {:0.2f}".format(r2_train_EO)
    
    fs = 14
    fs_cbar =12
        
    '''REDO SPEEDY CALC HERE'''
    
    bias_speedy_1_9_redo = np.zeros((length,ygrid,xgrid))
    for i in range(length):
        bias_speedy_1_9_redo[i,:,:] = ds_analysis_mean_speedy[i,:,:] - temp_500_nature[i,:,:]
    if variable_speedy == 'q':
        bias_speedy_1_9_redo_squared = ((np.average(bias_speedy_1_9_redo[24::,:,:],axis=0))*1000)**2.0
    else:
        bias_speedy_1_9_redo_squared = ((np.average(bias_speedy_1_9_redo[24::,:,:],axis=0)))**2.0
    
    cyclic_data_bias_squared_speedy_1_9_redo, cyclic_lons = add_cyclic_point(bias_speedy_1_9_redo_squared, coord=lon)
    ###
    if variable_speedy == 'q':
        MSE_speedy_1_9_redo = np.average(((bias_speedy_1_9_redo[24::,:,:]*1000)**2.0), axis = 0)
    else:
        MSE_speedy_1_9_redo = np.average(((bias_speedy_1_9_redo[24::,:,:])**2.0), axis = 0)

    cyclic_data_mse_speedy_1_9_redo, cyclic_lons = add_cyclic_point(MSE_speedy_1_9_redo, coord=lon)
    ###
    variance_speedy_1_9_redo = cyclic_data_mse_speedy_1_9_redo - cyclic_data_bias_squared_speedy_1_9_redo
    
    '''DONE REDO SPEEDY CALC'''
    
    # Plot SPEEDY
    
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot letkf-speedy
    ipickcolormap = 'gist_heat_r' #'magma_r'#'viridis'
#     img1 = axs[0].imshow(cyclic_data_MSE_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    img1 = axs[0].imshow(cyclic_data_mse_speedy_1_9_redo, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize=fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('PHYSICS\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error',fontsize = fs)
    else:
        axs[0].set_title('PHYSICS\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error',fontsize = fs)

    
#     img2 = axs[1].imshow(Bias_squared_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax=imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    img2 = axs[1].imshow(bias_speedy_1_9_redo_squared, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax=imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[1].coastlines()
#     axs[1].gridlines()
    cbar2 = plt.colorbar(img2, ax=axs[1], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar2.ax.tick_params(labelsize=fs_cbar)
    cbar2.set_label(imshowsquared_units,fontsize = fs)  # Change label for Data 2
    axs[1].set_title('Bias Squared',fontsize = fs)

    # PLOT variance = MS - Bias**2.0

    img3 = axs[2].imshow(Variance_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize = fs)
    plt.tight_layout()
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_SPEEDY_LETKF_paper_figures_10_15_24.pdf"
    print(filename)
    #plt.savefig('paper_figures_10_15_24/' + filename, dpi = 300)
    #plt.show()
    
    
    
    
    
    ############# hybrid iter 2
    bias_hybrid_1_9_1_9_1_9 = np.zeros((length,ygrid,xgrid))
    
    for i in range(length):
        bias_hybrid_1_9_1_9_1_9[i,:,:] = ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:] - temp_500_nature[i,:,:]
    if variable_speedy == 'q':
        bias_hybrid_1_9_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 
    # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
    else:
        bias_hybrid_1_9_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9_1_9[24::,:,:],axis=0)))**2.0 # g/kg

    
    cyclic_data_bias_sqaured_iter1_now, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_1_9_squared, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW_iter2 = np.average(((bias_hybrid_1_9_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else:
        MSE_NOW_iter2 = np.average(((bias_hybrid_1_9_1_9_1_9[24::,:,:])**2.0),axis=0)
    
    cyclic_data_mse_iter1_now, cyclic_lons = add_cyclic_point(MSE_NOW_iter2,coord=lon)
    
    #####
    variance_now_iter1 = cyclic_data_mse_iter1_now - cyclic_data_bias_sqaured_iter1_now
#     std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of Speedy trained Hybrid
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID 2
    img1 = axs[0].imshow(cyclic_data_mse_iter1_now, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize =fs)
    if variable_speedy == 'ps':
        axs[0].set_title('HYBRID-2\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize=fs)
    else:
        axs[0].set_title('HYBRID-2\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize=fs)
    
    
    img2 = axs[1].imshow(cyclic_data_bias_sqaured_iter1_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_iter1,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize = fs)
    
#     std_dev_now = np.sqrt(variance_now)
    
#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_Hybrid_1_9_1_9_1_7_date_10_15_24.pdf"
    print(filename)
    # Save the figure with the generated filename
    #plt.savefig('paper_figures_10_15_24/' + filename,dpi=300) #300
    #plt.show()
    
       ############# ERA5 hybrid
    bias_hybrid_era5_1_9 = np.zeros((length,ygrid,xgrid))
    
    for i in range(length):
        bias_hybrid_era5_1_9[i,:,:] = ds_analysis_mean[i,:,:] - temp_500_nature[i,:,:]
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_era5_1_9_squared = ((np.average(bias_hybrid_era5_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_era5_1_9_squared = ((np.average(bias_hybrid_era5_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_era5_1_9_squared ', bias_hybrid_era5_1_9_squared)
    
    cyclic_data_bias_squared_now_era5, cyclic_lons = add_cyclic_point(bias_hybrid_era5_1_9_squared, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW_era5 = np.average(((bias_hybrid_era5_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW_era5 = np.average(((bias_hybrid_era5_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now_era5, cyclic_lons = add_cyclic_point(MSE_NOW_era5,coord=lon)
    
    #####
    variance_now_era5 = cyclic_data_mse_now_era5 - cyclic_data_bias_squared_now_era5
#     std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of ERA5 Hybrid
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now_era5, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('HYBRID-OPT\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('HYBRID-OPT\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_squared_now_era5,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_era5,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    
#     std_dev_now = np.sqrt(variance_now)
    
#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_ERA5_trained_Hybrid_10_15_24.pdf"
    print(filename)
    # Save the figure with the generated filename
    #plt.savefig('paper_figures_10_15_24/' + filename,dpi=300)
    #plt.show()
    
    
    #### MAKE IMSHOW MAP FOR 1.9,1.9
    bias_hybrid_1_9_1_9 = np.zeros((length,ygrid,xgrid))
    
    for i in range(length):
        bias_hybrid_1_9_1_9[i,:,:] = ds_hybrid_1_9_1_9[i,:,:] - temp_500_nature[i,:,:]
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_1_9_1_9_squared = ((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_1_9_1_9_squared ', bias_hybrid_1_9_1_9_squared)
    
    cyclic_data_bias_sqaured_now, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_squared, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW = np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW = np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now, cyclic_lons = add_cyclic_point(MSE_NOW,coord=lon)
    
    #####
    variance_now = cyclic_data_mse_now - cyclic_data_bias_sqaured_now
    std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of Speedy trained Hybrid
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    
    if variable_speedy == 'ps':
        axs[0].set_title('HYBRID-1\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('HYBRID-1\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_sqaured_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_1st_iter_10_15_24.pdf"
    print(filename)
    # Save the figure with the generated filename
    #plt.savefig('paper_figures_10_15_24/' + filename,dpi=300)
    #plt.show()
    #################DONE MAP
    
    # *** IM WORKING HERE 6/11/24***
    # fixing 6/10/24 to be 2nd iter - 1st iter
    
    ipickcolormap = 'seismic'
    bias_difference_now = np.zeros((ygrid,xgrid))
    
    bias_difference_now =  bias_hybrid_1_9_1_9_1_9 - bias_hybrid_1_9_1_9  # hybrid 1.9,1.9,1.9 - hybrid 1.9,1.9
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_1_9_1_9_squared_difference = -bias_hybrid_1_9_1_9_squared + bias_hybrid_1_9_1_9_1_9_squared #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_1_9_1_9_squared_difference = -bias_hybrid_1_9_1_9_squared + bias_hybrid_1_9_1_9_1_9_squared         #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_1_9_1_9_squared_difference ', bias_hybrid_1_9_1_9_squared_difference)
    
    cyclic_data_bias_sqaured_now_difference, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_squared_difference, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW_difference = -MSE_NOW + MSE_NOW_iter2 # np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW_difference = -MSE_NOW + MSE_NOW_iter2 # np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now_difference, cyclic_lons = add_cyclic_point(MSE_NOW_difference,coord=lon)
    
    #####
    # Below is wrong
#     variance_now_difference = -cyclic_data_mse_now_difference + cyclic_data_bias_sqaured_now_difference
    # CORRECTION HERE
    variance2nditeration = variance_now_iter1
    variance1stiteration = variance_now
    
    variance_now_difference = variance2nditeration - variance1stiteration
#     std_dev_now = np.sqrt(variance_now)
    
    ########################### Plot of Mean Square Error, Bias**2, and Variance of DIFFERENCE OF 2nd iteration Speedy trained Hybrid - 1st iteration
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now_difference, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('Difference from\nHYBRID-2 - HYBRID-1\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('Difference from\nHYBRID-2 - HYBRID-1\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_sqaured_now_difference,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_difference,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    

    
#     std_dev_now = np.sqrt(variance_now)
    
#     img4 = axs[3].imshow(std_dev_now,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = 0, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
#     axs[3].coastlines()
#     cbar4 = plt.colorbar(img4, ax=axs[3],orientation='vertical',fraction=0.03,pad=0.005)
#     cbar4.set_label(units)
#     axs[3].set_title('Standard Deviation')
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_2iter_minus_1stiter_10_15_24.pdf"
    print(filename)
    # Save the figure with the generated filename
    #plt.savefig('paper_figures_10_15_24/' + filename,dpi=300)
    #plt.show()
    
    
    #DONE MAP
    
    
    
    
    # 8 18 24
    #
    # 
    ############################# MAP OF DIFFERENCE FROM HYBRID 1st iter to SPEEDY
    ipickcolormap = 'seismic'
#     bias_difference_now = np.zeros((ygrid,xgrid))
    
#     bias_difference_now =  bias_hybrid_1_9_1_9 - bias_speedy_1_9_redo_squared  # hybrid 1.9,1.9,1.9 - hybrid 1.9,1.9
    
    print('hybrid shape, speedy shape ', np.shape(bias_hybrid_1_9_1_9_1_9_squared), np.shape(Bias_squared_speedy))
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_1_9_1_9_squared_difference_minus_speedy = bias_hybrid_1_9_1_9_squared - bias_speedy_1_9_redo_squared #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_1_9_1_9_squared_difference_minus_speedy = bias_hybrid_1_9_1_9_squared - bias_speedy_1_9_redo_squared  ###(48,96 and 48,97????)       #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_1_9_1_9_squared_difference ', bias_hybrid_1_9_1_9_squared_difference_minus_speedy)
    
    cyclic_data_bias_squared_now_difference_1stiter_minus_speedy, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_squared_difference_minus_speedy, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW_difference_1stiter_minus_speedy = MSE_NOW - MSE_speedy # np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW_difference_1stiter_minus_speedy = MSE_NOW - MSE_speedy # np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now_difference_1stiter_minus_speedy, cyclic_lons = add_cyclic_point(MSE_NOW_difference_1stiter_minus_speedy,coord=lon)
    
    #####
    # Below is wrong
#     variance_now_difference = -cyclic_data_mse_now_difference + cyclic_data_bias_sqaured_now_difference
    # CORRECTION HERE
#     variance2nditeration = variance_now_iter1
#     variance1stiteration = variance_now
    
    variance_now_difference_1stiter_minus_speedy = variance1stiteration - Variance_speedy #variance2nditeration - variance1stiteration
#     std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of DIFFERENCE OF Hybrid 1st iter to SPEEDY
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now_difference_1stiter_minus_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('Difference from\nHYBRID-1 - PHYSICS\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('Difference from\nHYBRID-1 - PHYSICS\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_squared_now_difference_1stiter_minus_speedy,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_difference_1stiter_minus_speedy,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_1stiter_minus_speedy_10_15_24.pdf"
    print(filename)
    # Save the figure with the generated filename
    #plt.savefig('paper_figures_10_15_24/' + filename,dpi=300)
    #plt.show()
    # DONE MAP
    
    
    
    
    
    
    
    
    

    # 8 18 24
    #
    # 
    ######################### MAP OF DIFFERENCE FROM HYBRID 2nd iter to SPEEDY
    ipickcolormap = 'seismic'
#     bias_difference_now = np.zeros((ygrid,xgrid))
    
#     bias_difference_now =  bias_hybrid_1_9_1_9 - bias_speedy_1_9_redo_squared  # hybrid 1.9,1.9,1.9 - hybrid 1.9,1.9
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_hybrid_1_9_1_9_1_7_squared_difference_minus_speedy = bias_hybrid_1_9_1_9_1_9_squared - bias_speedy_1_9_redo_squared #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_hybrid_1_9_1_9_1_7_squared_difference_minus_speedy = bias_hybrid_1_9_1_9_1_9_squared - bias_speedy_1_9_redo_squared         #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_1_9_1_9_1_7_squared_difference_minus_speedy ', bias_hybrid_1_9_1_9_1_7_squared_difference_minus_speedy)
    
    cyclic_data_bias_squared_now_difference_2nditer_minus_speedy, cyclic_lons = add_cyclic_point(bias_hybrid_1_9_1_9_1_7_squared_difference_minus_speedy, coord=lon)
    ####
    if variable_speedy == 'q':
        MSE_NOW_difference_2nditer_minus_speedy = MSE_NOW_iter2 - MSE_speedy # np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW_difference_2nditer_minus_speedy = MSE_NOW_iter2 - MSE_speedy # np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now_difference_2nditer_minus_speedy, cyclic_lons = add_cyclic_point(MSE_NOW_difference_2nditer_minus_speedy,coord=lon)
    
    #####
    # Below is wrong
#     variance_now_difference = -cyclic_data_mse_now_difference + cyclic_data_bias_sqaured_now_difference
    # CORRECTION HERE
#     variance2nditeration = variance_now_iter1
#     variance1stiteration = variance_now
    
    variance_now_difference_2nditer_minus_speedy = variance2nditeration - Variance_speedy #variance2nditeration - variance1stiteration
#     std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of DIFFERENCE OF Hybrid 1st iter to SPEEDY
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now_difference_2nditer_minus_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('Difference from\nHYBRID-2 - PHYSICS\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('Difference from\nHYBRID-2 - PHYSICS\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_squared_now_difference_2nditer_minus_speedy,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_difference_2nditer_minus_speedy,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_2iter_minus_speedy_10_15_24.pdf"
    print(filename)
    # Save the figure with the generated filename
    #plt.savefig('paper_figures_10_15_24/' + filename,dpi=300)
    #plt.show()
    # DONE MAP
    
    
    
    
    
    
    
    
    # 8 18 24
    #
    # 
    ############################# MAP OF DIFFERENCE FROM HYBRID ERA5 Trained to SPEEDY
    ipickcolormap = 'seismic'
#     bias_difference_now = np.zeros((ygrid,xgrid))
    
#     bias_difference_now =  bias_hybrid_1_9_1_9 - bias_speedy_1_9_redo_squared  # hybrid 1.9,1.9,1.9 - hybrid 1.9,1.9
    
    if variable_speedy == 'q':
        # now i have the difference in kg/kg of every gridpoint at every time step. now to convert to time avg in g/kg
        bias_era5_hybrid_1_9_squared_difference_minus_speedy = bias_hybrid_era5_1_9_squared - bias_speedy_1_9_redo_squared #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0))*1000)**2.0 # g/kg
        
    else: 
        bias_era5_hybrid_1_9_squared_difference_minus_speedy = bias_hybrid_era5_1_9_squared - bias_speedy_1_9_redo_squared         #((np.average(bias_hybrid_1_9_1_9[24::,:,:],axis=0)))**2.0

    print('bias_hybrid_era5_squared_difference_minus_speedy ', bias_era5_hybrid_1_9_squared_difference_minus_speedy)
    
    cyclic_data_bias_squared_now_difference_era5_1_9_minus_speedy, cyclic_lons = add_cyclic_point(bias_era5_hybrid_1_9_squared_difference_minus_speedy, coord=lon)
    
    ####
    if variable_speedy == 'q':
        MSE_NOW_difference_era5_1_9_minus_speedy = MSE_NOW_era5 - MSE_speedy # np.average(((bias_hybrid_1_9_1_9[24::,:,:]*1000)**2.0),axis=0) # g/kg
    else: 
        MSE_NOW_difference_era5_1_9_minus_speedy = MSE_NOW_era5 - MSE_speedy # np.average(((bias_hybrid_1_9_1_9[24::,:,:])**2.0),axis=0) 
        
    cyclic_data_mse_now_difference_era5_1_9_minus_speedy, cyclic_lons = add_cyclic_point(MSE_NOW_difference_era5_1_9_minus_speedy,coord=lon)
    
    #####
    # Below is wrong
#     variance_now_difference = -cyclic_data_mse_now_difference + cyclic_data_bias_sqaured_now_difference
    # CORRECTION HERE
#     variance2nditeration = variance_now_iter1
#     variance1stiteration = variance_now
    
    variance_now_difference_era5_1_9_minus_speedy = variance_now_era5 - Variance_speedy #variance2nditeration - variance1stiteration
#     std_dev_now = np.sqrt(variance_now)
    
    # Plot of Mean Square Error, Bias**2, and Variance of DIFFERENCE OF Hybrid 1st iter to SPEEDY
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), subplot_kw={'projection': ccrs.PlateCarree()})
    # Plot HYBRID
    img1 = axs[0].imshow(cyclic_data_mse_now_difference_era5_1_9_minus_speedy, extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none')
    axs[0].coastlines()
#     axs[0].gridlines()
    cbar1 = plt.colorbar(img1, ax=axs[0], orientation='vertical', fraction=0.0244, pad=0.005)
    cbar1.ax.tick_params(labelsize = fs_cbar)
    cbar1.set_label(imshowsquared_units, fontsize = fs)
    if variable_speedy == 'ps':
        axs[0].set_title('Difference from\nHYBRID-OPT - PHYSICS\n'+ title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
    else:
        axs[0].set_title('Difference from\nHYBRID-OPT - PHYSICS\n'+ title_level  + title_var_era +'\nJan 1, 2011 to Jan 1, 2012\n'+ 'Mean Square Error', fontsize = fs)
        
    
    img2 = axs[1].imshow(cyclic_data_bias_squared_now_difference_era5_1_9_minus_speedy,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[1].coastlines()
    cbar2 = plt.colorbar(img2, ax=axs[1],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar2.ax.tick_params(labelsize = fs_cbar)
    cbar2.set_label(imshowsquared_units, fontsize = fs)
    axs[1].set_title('Bias Squared', fontsize = fs)
    
    img3 = axs[2].imshow(variance_now_difference_era5_1_9_minus_speedy,extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()),vmin = -imshow_v_max, vmax= imshow_v_max, cmap=ipickcolormap, transform=ccrs.PlateCarree(), interpolation='none' )
    axs[2].coastlines()
    cbar3 = plt.colorbar(img3, ax=axs[2],orientation='vertical',fraction=0.0244,pad=0.005)
    cbar3.ax.tick_params(labelsize= fs_cbar)
    cbar3.set_label(imshowsquared_units, fontsize = fs)
    axs[2].set_title('Variance', fontsize =fs)
    
    plt.tight_layout()
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_era5hybrid_minus_speedy_10_15_24.pdf"
    print(filename)
    # Save the figure with the generated filename
    #plt.savefig('paper_figures_10_15_24/' + filename,dpi=300)
    #plt.show()
    
    # DONE MAP    
    
     
   ############ # MAKE TIME SERIES NOW  ################
    
    x = np.arange(0,length)
    base = datetime(2011,1,1,0)

    plt.figure(figsize=(16,8))
    date_list = [base + timedelta(days=x/4) for x in range(length)]
    ### PLOT ENS MEMBERS 
    
    # make colors for 40 member ens
#     from matplotlib.colors import LinearSegmentedColormap
#     start_color = np.array([1.0, 0.8, 0.8])  # Light Red (RGB values)
#     end_color = np.array([0.5, 0.0, 0.0])   # Dark Red (RGB values)
#     # Create a colormap with 40 colors by linearly interpolating between start and end colors
#     cmap = LinearSegmentedColormap.from_list("custom_colormap", np.linspace(start_color, end_color, 40))
#     # Get a list of 40 different colors from the colormap
#     num_colors = 40
#     colors = [cmap(i / (num_colors - 1)) for i in range(num_colors)]

#     for each_member in range(0,40):
#         plt.plot(date_list,analysis_rmse_object[each_member],linewidth=.45,color=colors[each_member],label=each_member+1)
    # ALSO Average of MEAN LINE
    
    width = 1
    
    if var_era == 'Specific_Humidity':
        # convert (kg/kg) to (g/kg)
        
        analysis_rmse = analysis_rmse*1000
        analysis_rmse_speedy = analysis_rmse_speedy*1000
        anal_mean_rmse_hybrid_1_9_1_9_1_9 = anal_mean_rmse_hybrid_1_9_1_9_1_9*1000
        hybrid_1_9_1_9_anal_rmse = hybrid_1_9_1_9_anal_rmse*1000
        

    
#     plt.plot(date_list,anal_mean_rmse_hybrid_1_9_1_9_1_9,color='red',lw=width,label="SPEEDY trained Hybrid 1.9,1.3")
#     plt.axhline(y=np.average(anal_mean_rmse_hybrid_1_9_1_9_1_9[20::]),color='red',lw=width, linestyle='--',label="Average SPEEDY trained Hybrid 1.9,1.3")
    
    #### TROY
#     plt.plot(date_list,anal_rmse_troy,linewidth=.45,color='red',label='TROY NEW HYBRID 1.5,1.3 Mean')
#     plt.axhline(y=np.average(anal_rmse_troy[20::]), color='red',lw=.6, linestyle='--',label="Average TROY NEW HYBRID 1.5,1.3 Mean")
    
    plt.plot(date_list,analysis_rmse_speedy,label='PHYSICS',linewidth=width,color='blue')    
    plt.axhline(y=np.average(analysis_rmse_speedy[24::]), color='blue',lw=width, linestyle='--',label="Average PHYSICS")
    
    plt.plot(date_list,analysis_rmse,label='HYBRID-OPT',linewidth=width,color='black')
    plt.axhline(y=np.average(analysis_rmse[24::]), color='black',lw=width, linestyle='--',label="Average HYBRID-OPT")
    
    
    plt.plot(date_list,hybrid_1_9_1_9_anal_rmse,label='HYBRID-1',linewidth=width,color='red')
    plt.axhline(y=np.average(hybrid_1_9_1_9_anal_rmse[24::]),color= 'red', lw=width,linestyle='--',label='Average HYBRID-1')
    
    plt.plot(date_list,anal_mean_rmse_hybrid_1_9_1_9_1_9, label='HYBRID-2', linewidth=width, color= 'green')
    plt.axhline(y=np.average(anal_mean_rmse_hybrid_1_9_1_9_1_9[24::]), color ='green', lw=width, linestyle='--', label='Average HYBRID-2')
    
    
#     print(np.average(hybrid_1_9_1_9_anal_rmse[24::]))
#     plt.plot(date_list,analysis_rmse_object[0],linewidth=.45,color='green',label='MEM1 in GREEN')
#     plt.axhline(y=np.average(analysis_rmse[20::]), color='r',lw=width, linestyle='--',label="Average RMSE 1st Iteration Hybrid 1.5,1.3")
#     plt.axhline(y=np.average(analysis_rmse_speedy[20::]), color='b',lw=width, linestyle='--',label="Average RMSE SPEEDY 1.3")

    plt.title('LETKF Analysis RMS Error\n'+ title_level + title_var_era,fontsize = fs+3)
    if variable_speedy == 'ps':
        plt.title('LETKF Analysis RMS Error\n'+ title_var_era,fontsize = fs+3)
        
    plt.xlabel('Date',fontsize = fs+3)
    plt.ylabel(f'Global root-mean-square error {units}',fontsize = fs+3)
    plt.xticks(fontsize=fs)
    plt.yticks(fontsize=fs)
    plt.xlim([startdate,datetime(2012,1,1,0)])
    
    if variable_speedy == 'q':
        if level == .95:
            plt.ylim(.5,2.25)
        elif level == .2:
            plt.ylim(0,.07)
    elif variable_speedy == 'v':
        if level == .95:
            plt.ylim(1.5,3.75)
        elif level == .2:
            plt.ylim(2,6)
    elif variable_speedy == 'u':
        if level == .95:
            plt.ylim(1.5,4)
        elif level == .2:
            plt.ylim(2,6)
    elif variable_speedy == 't':
        if level == .95:
            plt.ylim(.75, 4)
        elif level == .2:
            plt.ylim(.5,3)
    elif variable_speedy == 'ps':
        plt.ylim(0,19)
        

#     handles, labels = plt.get_legend_handles_labels()
#     lgd = plt.legend(handles, labels, ncol=1,loc='center right', bbox_to_anchor=(1.7, 0.5),fontsize = 15)


#     plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.23,0.5),fontsize=fs)
    
    if variable_speedy == 'ps':
        plt.legend(ncol=2,loc='center right',fontsize=fs)
    else:
        plt.legend(ncol=2,loc='upper right',fontsize=fs)
#     if variable_speedy == 'q':
#         if level == .2:
#             plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
#         else:
#             plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
#     elif variable_speedy == 'v':
#         if level == .95:
#             plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
#         elif level == .2:
#             plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
#     else:
#         plt.legend(ncol=1,loc='center right', bbox_to_anchor=(1.3, 0.5), fontsize = 13)
#     plt.legend()
    plt.grid(color='grey', linestyle='--', linewidth=.2)
    
    # Generate a filename based on level and variable_name level_in_speedy,variable_speedy
    
    # Save the figure with the generated filename
    plt.tight_layout()
#     if variable_speedy == 'ps':
#         filename2 = f"Time_Series_of_3_models_for_level_{level_in_speedy}_variable_{variable_speedy}_reduced_ylim.pdf"
#     else:
#         filename2 = f"Time_Series_of_3_models_for_level_{level_in_speedy}_variable_{variable_speedy}.pdf"
    filename2 = f"Time_Series_of_3_models_for_level_{level_in_speedy}_variable_{variable_speedy}.pdf"
    
    plt.savefig('figures_paper_version_2_11_7_24/'+filename2,dpi=300) # bbox_inches='tight')
    #plt.show()
    
#     # print the percent improvment from SPEEDY
#         plt.plot(date_list,analysis_rmse_speedy,label='PHYSICS',linewidth=width,color='blue')    
#     plt.axhline(y=np.average(analysis_rmse_speedy[24::]), color='blue',lw=width, linestyle='--',label="Average PHYSICS")
    
#     plt.plot(date_list,analysis_rmse,label='HYBRID-OPT',linewidth=width,color='black')
#     plt.axhline(y=np.average(analysis_rmse[24::]), color='black',lw=width, linestyle='--',label="Average HYBRID-OPT")
    
    
#     plt.plot(date_list,hybrid_1_9_1_9_anal_rmse,label='HYBRID-1',linewidth=width,color='red')
#     plt.axhline(y=np.average(hybrid_1_9_1_9_anal_rmse[24::]),color= 'red', lw=width,linestyle='--',label='Average HYBRID-1')
    
#     plt.plot(date_list,anal_mean_rmse_hybrid_1_9_1_9_1_9, label='HYBRID-2', linewidth=width, color= 'green')
#     plt.axhline(y=np.average(anal_mean_rmse_hybrid_1_9_1_9_1_9[24::]), color ='green', lw=width, linestyle='--', label='Average HYBRID-2')
    print(f'values of percent change in time series for {variable_speedy} at {level_in_speedy} : ')
    
    hybrid_1_avg_value =  np.average(hybrid_1_9_1_9_anal_rmse[24::])
    hybrid_2_avg_value = np.average(anal_mean_rmse_hybrid_1_9_1_9_1_9[24::])
    hybrid_opt_avg_value = np.average(analysis_rmse[24::])
    physics_avg_value = np.average(analysis_rmse_speedy[24::])
    
    def calc_speedy_change(x):
        
        percent_change = - ( ((x - physics_avg_value )/ physics_avg_value ) * 100) 
        
        return percent_change
    
    avg_value_list = [hybrid_1_avg_value, hybrid_2_avg_value, hybrid_opt_avg_value, physics_avg_value]
    
#     for i in avg_value_list:
#         print(f'{i} % improvement: ', calc_speedy_change(i))
    
    avg_value_dict = {
    "hybrid_1_avg_value": hybrid_1_avg_value,
    "hybrid_2_avg_value": hybrid_2_avg_value,
    "hybrid_opt_avg_value": hybrid_opt_avg_value,
    "physics_avg_value": physics_avg_value
    }

    for name, i in avg_value_dict.items():
        print(f'{name} % improvement: ', calc_speedy_change(i))
    
    # CLose previous plots 
    plt.close('all')

    # Make experiment PHYSICS and HYBRID-OPT in the same figure
    fig = plt.figure(figsize=(9.3, 7.5))  # Adjusted size to maintain aspect ratio
    gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[.972, 1])

    ipickcolormap = 'gist_heat_r'  # Your colormap choice

    if variable_speedy == 'ps':
        fig.suptitle(title_var_era + '\nJan 1, 2011 to Jan 1, 2012', fontsize=fs)
    else:
        fig.suptitle(title_level + title_var_era + '\nJan 1, 2011 to Jan 1, 2012', fontsize=fs)

    # Plot PHYSICS
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    img1 = ax1.imshow(cyclic_data_mse_speedy_1_9_redo, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=0, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax1.coastlines()
    ax1.text(-0.16, 0.5, 'Mean-square-error', transform=ax1.transAxes,
             fontsize=fs, verticalalignment='center', horizontalalignment='right',rotation=90)
    ax1.set_title('PHYSICS',fontsize=fs)

    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    img2 = ax2.imshow(bias_speedy_1_9_redo_squared, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=0, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax2.coastlines()
    ax2.text(-0.16, 0.5, 'Bias Squared', transform=ax2.transAxes,
             fontsize=fs, verticalalignment='center', horizontalalignment='right',rotation=90)

    ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    img3 = ax3.imshow(Variance_speedy, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=0, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax3.coastlines()
    ax3.text(-0.16, 0.5, 'Variance', transform=ax3.transAxes,
             fontsize=fs, verticalalignment='center', horizontalalignment='right',rotation=90)

    # Plot HYBRID-OPT
    ax4 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    img4 = ax4.imshow(cyclic_data_mse_now_era5, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=0, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax4.coastlines()
    cbar4 = plt.colorbar(img4, orientation='vertical', fraction=0.0244, pad=0.005)
    cbar4.ax.tick_params(labelsize=fs_cbar)
    cbar4.set_label(imshowsquared_units, fontsize=fs)
    ax4.set_title('HYBRID-OPT',fontsize=fs)

    ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    img5 = ax5.imshow(cyclic_data_bias_squared_now_era5, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=0, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax5.coastlines()
    cbar5 = plt.colorbar(img5, orientation='vertical', fraction=0.0244, pad=0.005)
    cbar5.ax.tick_params(labelsize=fs_cbar)
    cbar5.set_label(imshowsquared_units, fontsize=fs)

    ax6 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
    img6 = ax6.imshow(variance_now_era5, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=0, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax6.coastlines()
    fontsize_labels = 10     
    #for ax in [ax3,ax6]:
    ax3.set_xticks([-180, -90, 0, 90, 180])
    ax3.set_xticklabels(['180\u00B0', '90\u00B0 W', '0\u00B0', '90\u00B0 E', ''], fontsize = fontsize_labels)
    ax6.set_xticks([-180, -90, 0, 90, 180])
    ax6.set_xticklabels(['180\u00B0', '90\u00B0 W', '0\u00B0', '90\u00B0 E', '180\u00B0'], fontsize = fontsize_labels)

    for ax in [ax1,ax2,ax3]: 
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_yticklabels(['90\u00B0 S', '45\u00B0 S', '0\u00B0', '45\u00B0 N', '90\u00B0 N'],fontsize = fontsize_labels)    	
  

    cbar6 = plt.colorbar(img6, orientation='vertical', fraction=0.0244, pad=0.005)
    cbar6.ax.tick_params(labelsize=fs_cbar)
    cbar6.set_label(imshowsquared_units, fontsize=fs)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.006)
    plt.subplots_adjust(wspace=0.01)
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_PHYSICS_AND_HYBRID_OPT.pdf"
    # print(filename)
    plt.savefig('figures_paper_version_2_11_7_24/' + filename, dpi=300)
    #plt.show()
    
    
    
    # Make DIFF Hybrid-1 - PHYSICS and then HYBRID-2 - HYBRID-1
    
    fig = plt.figure(figsize=(9.3, 7.5))  # Adjusted size to maintain aspect ratio
    gs = gridspec.GridSpec(nrows=3, ncols=2, width_ratios=[.972, 1])

    ipickcolormap = 'seismic'  # Your colormap choice

    if variable_speedy == 'ps':
        fig.suptitle(title_var_era + '\nJan 1, 2011 to Jan 1, 2012', fontsize=fs)
    else:
        fig.suptitle(title_level + title_var_era + '\nJan 1, 2011 to Jan 1, 2012', fontsize=fs)

    # Plot PHYSICS
    ax1 = fig.add_subplot(gs[0, 0], projection=ccrs.PlateCarree())
    img1 = ax1.imshow(cyclic_data_mse_now_difference_1stiter_minus_speedy, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=-imshow_v_max, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax1.coastlines()
    ax1.text(-0.16, 0.5, 'Mean-square-error', transform=ax1.transAxes,
             fontsize=fs, verticalalignment='center', horizontalalignment='right',rotation=90)
    ax1.set_title('HYBRID-1 - PHYSICS',fontsize=fs)

    ax2 = fig.add_subplot(gs[1, 0], projection=ccrs.PlateCarree())
    img2 = ax2.imshow(cyclic_data_bias_squared_now_difference_1stiter_minus_speedy, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=-imshow_v_max, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax2.coastlines()
    ax2.text(-0.16, 0.5, 'Bias Squared', transform=ax2.transAxes,
             fontsize=fs, verticalalignment='center', horizontalalignment='right',rotation=90)

    ax3 = fig.add_subplot(gs[2, 0], projection=ccrs.PlateCarree())
    img3 = ax3.imshow(variance_now_difference_1stiter_minus_speedy, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=-imshow_v_max, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax3.coastlines()
    ax3.text(-0.16, 0.5, 'Variance', transform=ax3.transAxes,
             fontsize=fs, verticalalignment='center', horizontalalignment='right',rotation=90)

    # Plot HYBRID-2 - HYBRID-1
    ax4 = fig.add_subplot(gs[0, 1], projection=ccrs.PlateCarree())
    img4 = ax4.imshow(cyclic_data_mse_now_difference, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=-imshow_v_max, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax4.coastlines()
    cbar4 = plt.colorbar(img4, orientation='vertical', fraction=0.0244, pad=0.005)
    cbar4.ax.tick_params(labelsize=fs_cbar)
    cbar4.set_label(imshowsquared_units, fontsize=fs)
    ax4.set_title('HYBRID-2 - HYBRID-1',fontsize=fs)

    ax5 = fig.add_subplot(gs[1, 1], projection=ccrs.PlateCarree())
    img5 = ax5.imshow(cyclic_data_bias_sqaured_now_difference, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=-imshow_v_max, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax5.coastlines()
    cbar5 = plt.colorbar(img5, orientation='vertical', fraction=0.0244, pad=0.005)
    cbar5.ax.tick_params(labelsize=fs_cbar)
    cbar5.set_label(imshowsquared_units, fontsize=fs)

    ax6 = fig.add_subplot(gs[2, 1], projection=ccrs.PlateCarree())
    img6 = ax6.imshow(variance_now_difference, 
                      extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                      vmin=-imshow_v_max, vmax=imshow_v_max, cmap=ipickcolormap, 
                      transform=ccrs.PlateCarree(), interpolation='none')
    ax6.coastlines()
    fontsize_labels = 10     
    ax3.set_xticks([-180, -90, 0, 90, 180])
    ax3.set_xticklabels(['180\u00B0', '90\u00B0 W', '0\u00B0', '90\u00B0 E', ''], fontsize = fontsize_labels)
    ax6.set_xticks([-180, -90, 0, 90, 180])
    ax6.set_xticklabels(['180\u00B0', '90\u00B0 W', '0\u00B0', '90\u00B0 E', '180\u00B0'], fontsize = fontsize_labels)

    for ax in [ax1,ax2,ax3]: 
        ax.set_yticks([-90, -45, 0, 45, 90])
        ax.set_yticklabels(['90\u00B0 S', '45\u00B0 S', '0\u00B0', '45\u00B0 N', '90\u00B0 N'],fontsize = fontsize_labels)    	
  
    cbar6 = plt.colorbar(img6, orientation='vertical', fraction=0.0244, pad=0.005)
    cbar6.ax.tick_params(labelsize=fs_cbar)
    cbar6.set_label(imshowsquared_units, fontsize=fs)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.006)
    plt.subplots_adjust(wspace=0.01)
    filename = f"Map_of_level_{level_in_speedy}_variable_{variable_speedy}_HYBRID1_minus_PHYSICS_and_HYBRID2_minus_HYBRID1.pdf"
    # print(filename)
    plt.savefig('figures_paper_version_2_11_7_24/' + filename, dpi=300)
    #plt.show()
    plt.close('all')
    print('DONE')
    
    


# In[104]:


# TEST CELL an individual variable/level combination
#rmse_time_series_plot_and_maps(.51,'t')


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:



## LOOP THROUGH FUNCTION
# FUNCTION INPUTS
level_list_speedy = [.95, .2] #[.95,.51, .2]
variable_list_speedy = ['t', 'q','ps'] #['u','t','v','q','ps'] 
# variable_list_speedy = ['t','v','q','ps']
# variable_list_speedy = ['u']


for level in level_list_speedy:
    for variable in variable_list_speedy:
        if level != .95 and variable =='ps':
            break
        rmse_time_series_plot_and_maps(level,variable)
        


# In[ ]:


