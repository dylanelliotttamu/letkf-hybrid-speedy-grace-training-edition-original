# Make forecast error calculations for 0 h, 6 h, 12 h , 24 h, 48 h, 72 h, 120 h, 240 h                                   
     
# import json
import numpy as np
import xarray as xr
# import pandas as pd
import matplotlib.pyplot as plt
import os
from datetime import datetime, timedelta
import sys 
import glob 
import re
from netCDF4 import Dataset
# IMPORT

# import numpy.ma as ma
print('numpy')
# import matplotlib as mpl
import matplotlib.gridspec as gridspec
# from matplotlib.ticker import NullFormatter
print('matplot lib')
from netCDF4 import Dataset
import cartopy.crs as ccrs
from cartopy.util import add_cyclic_point
from numba import jit
# from obspy.imaging.cm import viridis_white_r
print('past packages')
        
# CORE FUNCTIONS 
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
'''


Begin 0 hr forecast calculation


'''
print('\nBegin 0 hr forecast calculations\n')

print('here loading')

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

## MAIN SCRIPT
## **** CHANGE FILES AND DATES HERE

def rmse_spatial_and_temporal_avg(level_in_speedy,variable_speedy):
    # Define: Initial FILES, dates, Variable, and Level desired

    # analysis_file_speedy = '/scratch/user/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/speedy_1_9_20110101_20120101/mean_output/out.nc'    
    analysis_file_speedy = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/speedy_1_9_20110101_20120101/mean_output/out.nc' #speedy_1_9_uniform_20110101_20110501/mean.nc'

    ### ERA5  hybrid
    # analysis_file ='/scratch/user/dylanelliott/letkf-hybrid-speedy-from-skynet-2/DATA/uniform_letkf_analysis/ERA5_hybrid_1_9_20110101_20120101/mean_output/out.nc'
    analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/ERA5_hybrid_1_9_20110101_20120101/mean_output/out.nc' #uniform_letkf_anal_older_stuff/ERA5_1_9/mean_output/out.nc'
    
    ## hybrid 1.9,1.9
    
    # hybrid_1_9_1_9_file ='/scratch/user/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/hybrid_1_9_1_9_mem_1_fixed_20110101_20120115/out.nc' 
    hybrid_1_9_1_9_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_9_mem_1_fixed_20110101_20120115/out.nc'

    #### ACTUALLY 1.9,1.9,1.7
    # hybrid_1_9_1_9_1_9_file = '/scratch/user/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/hybrid_1_9_1_9_1_7_20110101_20120101/mean_output/out.nc' 
    hybrid_1_9_1_9_1_9_file = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_9_1_7_20110101_20120101/mean_output/out.nc'

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
        nature_file = f'/skydata2/troyarcomano/ERA_5/{year}/era_5_y{year}_regridded_mpi_fixed_var.nc'#f'/scratch/user/troyarcomano/ERA_5/{year}/era_5_y{year}_regridded_mpi_fixed_var_gcc.nc'
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
    
    global ds_analysis_mean

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
    
    # make them global to be added to dictionary
    # temporal
    global analysis_rmse_speedy
    global hybrid_1_9_1_9_anal_rmse
    global anal_mean_rmse_hybrid_1_9_1_9_1_9
    global anal_rmse_era5hybrid
    # spatial
    global analysis_error_speedy
    global hybrid_1_9_1_9_anal_error
    global anal_mean_error_hybrid_1_9_1_9_1_9
    global anal_error_era5hybrid

    analysis_rmse_ = np.zeros((length))
    analysis_rmse_speedy = np.zeros((length))
    global_average_ensemble_spread_era5 = np.zeros((length))
    global_average_ensemble_spread_speedy = np.zeros((length))
    
    hybrid_1_9_1_9_anal_rmse = np.zeros((length))

    anal_rmse_era5hybrid = np.zeros((length))
    #ps_rmse = np.zeros((length))

    analysis_error = np.zeros((length,ygrid,xgrid))
    analysis_error_speedy = np.zeros((length,ygrid,xgrid))
    
    hybrid_1_9_1_9_anal_error = np.zeros((length,ygrid,xgrid))
    anal_error_era5hybrid = np.zeros((length,ygrid,xgrid))
    
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
        # analysis_rmse[i] = latituded_weighted_rmse(temp_500_nature[i,:,:],temp_500_analysis[i,:,:],lats)
        analysis_rmse_speedy[i] = latituded_weighted_rmse(temp_500_nature[i,:,:],temp_500_analysis_speedy[i,:,:],lats)
        hybrid_1_9_1_9_anal_rmse[i] = latituded_weighted_rmse(temp_500_nature[i,:,:], ds_hybrid_1_9_1_9[i,:,:],lats)
        anal_rmse_era5hybrid[i] = latituded_weighted_rmse(temp_500_nature[i,:,:], ds_analysis_mean[i,:,:],lats)
        
        # ERROR BY GRIDPOINT
        # analysis_error[i,:,:] = temp_500_analysis[i,:,:] - temp_500_nature[i,:,:]
        analysis_error_speedy[i,:,:] = temp_500_analysis_speedy[i,:,:] - temp_500_nature[i,:,:]
        hybrid_1_9_1_9_anal_error[i,:,:] = ds_hybrid_1_9_1_9[i,:,:] - temp_500_nature[i,:,:]
        anal_error_era5hybrid[i,:,:] = ds_analysis_mean[i,:,:] - temp_500_nature[i,:,:]
        
        # BIAS FOR MAPS 
        # analysis_bias[i] = latituded_weighted_bias(temp_500_nature[i,:,:],temp_500_analysis[i,:,:],lats)
        # analysis_bias_speedy[i] = latituded_weighted_bias(temp_500_nature[i,:,:],temp_500_analysis_speedy[i,:,:],lats)
        # hybrid_1_9_1_9_bias[i] = latituded_weighted_bias(temp_500_nature[i,:,:],ds_hybrid_1_9_1_9[i,:,:],lats)
        
#         global_average_ensemble_spread_era5[i] = np.average(temp_500_spread_era5[i,:,:])
#         global_average_ensemble_spread_speedy[i] = np.average(temp_500_spread_speedy[i,:,:])

    # print('mean analysis_rmse = ',analysis_rmse)

    print('DONE CALCULATING ERROR AT EVERY GRIDPOINT AT EVERY TIMESTEP.')
    ## ADDING MEAN of Hybrid
#     path_mean_anal_hybrid_retest = hybrid_base_path + 'mean_output/out.nc'
    
#     path_mean_anal_hybrid_retest = '/skydata2/dylanelliott/letkf-hybrid-speedy/DATA/uniform_letkf_anal/hybrid_1_9_1_3_mem1_fixed_20110101_20120501/mean_output/out.nc'
    path_mean_anal_hybrid_retest = hybrid_1_9_1_9_1_9_file


    if variable_speedy == 'ps':
        ds_mean_anal_hybrid_1_9_1_9_1_9 = xr.open_dataset(path_mean_anal_hybrid_retest)[var_da].sel(time=time_slice) / 100.0
    else:    
        ds_mean_anal_hybrid_1_9_1_9_1_9 = xr.open_dataset(path_mean_anal_hybrid_retest)[var_da].sel(lev=level, time=time_slice)

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
        
    ##### and calc for mem 1 with length == 3:
    # analysis_rmse_mem1 = latituded_weighted_rmse(temp_500_nature[3,:,:],ds_list[1][3,:,:],lats)
    # print(analysis_rmse_mem1)
    #####

    print('Calc hybrid analysis_error and bias')
    
    for i in range(length):
        
        anal_mean_error_hybrid_1_9_1_9_1_9[i,:,:] = ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:] - temp_500_nature[i,:,:]
        anal_mean_rmse_hybrid_1_9_1_9_1_9[i] = latituded_weighted_rmse(temp_500_nature[i,:,:], ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:],lats)
        # anal_bias_hybrid_1_9_1_9_1_9[i] = latituded_weighted_bias(temp_500_nature[i,:,:],ds_mean_anal_hybrid_1_9_1_9_1_9[i,:,:],lats)
        
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

    ''' 24(below) instead of 28 to cut transient event (ML spin up) out in first few weeks '''
    #### TAKING OUT ABSOLUTE VALUE 
#     averaged_error = np.average(abs(analysis_error[24::,:,:]),axis=0)
    
    # we want abs value again (RMSE) so lets put abs value back in.. 1/20/25
    print('Take spatial averages..')
    # averaged_error = np.average(abs(analysis_error[24::,:,:]),axis=0)
    averaged_error_speedy = np.average(abs(analysis_error_speedy[24::,:,:]),axis=0)
    print('average_error_speedy = ',averaged_error_speedy)

    averaged_error_hybrid_1_9_1_9 = np.average(abs(hybrid_1_9_1_9_anal_error[24::,:,:]),axis=0) # this is average at each grid point. all I need to do is subtract 1 number from all these
    averaged_anal_mean_error_hybrid_1_9_1_9_1_9 = np.average(abs(anal_mean_error_hybrid_1_9_1_9_1_9[24::,:,:]),axis=0)

    averaged_anal_error_era5hybrid = np.average(abs(anal_error_era5hybrid[24::,:,:]),axis=0)
    
    print('Take temporal averages..')
    averaged_rmse_speedy = np.average(abs(analysis_rmse_speedy[24::]))
    print('average_rmse_speedy = ',averaged_rmse_speedy)
    
    averaged_rmse_hybrid_1_9_1_9 = np.average(abs(hybrid_1_9_1_9_anal_rmse[24::]))
    averaged_rmse_hybrid_1_9_1_9_1_7 = np.average(abs(anal_mean_rmse_hybrid_1_9_1_9_1_9[24::]))
    averaged_rmse_era5hybrid = np.average(abs(anal_rmse_era5hybrid[24::]))
    
    if variable_speedy == 't':
        avg_gp_error_physics_0h_t.append(averaged_error_speedy)
        avg_gp_error_hybrid_1_0h_t.append(averaged_error_hybrid_1_9_1_9)
        avg_gp_error_hybrid_2_0h_t.append(averaged_anal_mean_error_hybrid_1_9_1_9_1_9)
        avg_gp_error_hybrid_opt_0h_t.append(averaged_anal_error_era5hybrid)
        avg_global_error_physics_0h_t.append(averaged_rmse_speedy)
        avg_global_error_hybrid_1_0h_t.append(averaged_rmse_hybrid_1_9_1_9)
        avg_global_error_hybrid_2_0h_t.append(averaged_rmse_hybrid_1_9_1_9_1_7)
        avg_global_error_hybrid_opt_0h_t.append(averaged_rmse_era5hybrid)
    if variable_speedy == 'v':
        avg_gp_error_physics_0h_v.append(averaged_error_speedy)
        avg_gp_error_hybrid_1_0h_v.append(averaged_error_hybrid_1_9_1_9)
        avg_gp_error_hybrid_2_0h_v.append(averaged_anal_mean_error_hybrid_1_9_1_9_1_9)
        avg_gp_error_hybrid_opt_0h_v.append(averaged_anal_error_era5hybrid)
        avg_global_error_physics_0h_v.append(averaged_rmse_speedy)
        avg_global_error_hybrid_1_0h_v.append(averaged_rmse_hybrid_1_9_1_9)
        avg_global_error_hybrid_2_0h_v.append(averaged_rmse_hybrid_1_9_1_9_1_7)
        avg_global_error_hybrid_opt_0h_v.append(averaged_rmse_era5hybrid)
    if variable_speedy == 'u':
        avg_gp_error_physics_0h_u.append(averaged_error_speedy)
        avg_gp_error_hybrid_1_0h_u.append(averaged_error_hybrid_1_9_1_9)
        avg_gp_error_hybrid_2_0h_u.append(averaged_anal_mean_error_hybrid_1_9_1_9_1_9)
        avg_gp_error_hybrid_opt_0h_u.append(averaged_anal_error_era5hybrid)
        avg_global_error_physics_0h_u.append(averaged_rmse_speedy)
        avg_global_error_hybrid_1_0h_u.append(averaged_rmse_hybrid_1_9_1_9)
        avg_global_error_hybrid_2_0h_u.append(averaged_rmse_hybrid_1_9_1_9_1_7)
        avg_global_error_hybrid_opt_0h_u.append(averaged_rmse_era5hybrid)
    if variable_speedy == 'q':
        avg_gp_error_physics_0h_q.append(averaged_error_speedy)
        avg_gp_error_hybrid_1_0h_q.append(averaged_error_hybrid_1_9_1_9)
        avg_gp_error_hybrid_2_0h_q.append(averaged_anal_mean_error_hybrid_1_9_1_9_1_9)
        avg_gp_error_hybrid_opt_0h_q.append(averaged_anal_error_era5hybrid)
        avg_global_error_physics_0h_q.append(averaged_rmse_speedy)
        avg_global_error_hybrid_1_0h_q.append(averaged_rmse_hybrid_1_9_1_9)
        avg_global_error_hybrid_2_0h_q.append(averaged_rmse_hybrid_1_9_1_9_1_7)
        avg_global_error_hybrid_opt_0h_q.append(averaged_rmse_era5hybrid)
    if variable_speedy == 'ps':
        avg_gp_error_physics_0h_ps.append(averaged_error_speedy)
        avg_gp_error_hybrid_1_0h_ps.append(averaged_error_hybrid_1_9_1_9)
        avg_gp_error_hybrid_2_0h_ps.append(averaged_anal_mean_error_hybrid_1_9_1_9_1_9)
        avg_gp_error_hybrid_opt_0h_ps.append(averaged_anal_error_era5hybrid)
        avg_global_error_physics_0h_ps.append(averaged_rmse_speedy)
        avg_global_error_hybrid_1_0h_ps.append(averaged_rmse_hybrid_1_9_1_9)
        avg_global_error_hybrid_2_0h_ps.append(averaged_rmse_hybrid_1_9_1_9_1_7)
        avg_global_error_hybrid_opt_0h_ps.append(averaged_rmse_era5hybrid)
    print('DONE get 0hr forecast error')
    
# create a list to store 0 h variable and level combinations 
# Temperature
# average gridpoint
avg_gp_error_physics_0h_t = []
avg_gp_error_hybrid_1_0h_t = []
avg_gp_error_hybrid_2_0h_t = []
avg_gp_error_hybrid_opt_0h_t = []
avg_global_error_physics_0h_t = []
avg_global_error_hybrid_1_0h_t = []
avg_global_error_hybrid_2_0h_t = []
avg_global_error_hybrid_opt_0h_t = []
# V-wind
avg_gp_error_physics_0h_v = []
avg_gp_error_hybrid_1_0h_v = []
avg_gp_error_hybrid_2_0h_v = []
avg_gp_error_hybrid_opt_0h_v = []
avg_global_error_physics_0h_v = []
avg_global_error_hybrid_1_0h_v = []
avg_global_error_hybrid_2_0h_v = []
avg_global_error_hybrid_opt_0h_v = []
# U-wind
avg_gp_error_physics_0h_u = []
avg_gp_error_hybrid_1_0h_u = []
avg_gp_error_hybrid_2_0h_u = []
avg_gp_error_hybrid_opt_0h_u = []
avg_global_error_physics_0h_u = []
avg_global_error_hybrid_1_0h_u = []
avg_global_error_hybrid_2_0h_u = []
avg_global_error_hybrid_opt_0h_u = []
# Specific Humidity
avg_gp_error_physics_0h_q = []
avg_gp_error_hybrid_1_0h_q = []
avg_gp_error_hybrid_2_0h_q = []
avg_gp_error_hybrid_opt_0h_q = []
avg_global_error_physics_0h_q = []
avg_global_error_hybrid_1_0h_q = []
avg_global_error_hybrid_2_0h_q = []
avg_global_error_hybrid_opt_0h_q = []
# Surface Pressure
avg_gp_error_physics_0h_ps = []
avg_gp_error_hybrid_1_0h_ps = []
avg_gp_error_hybrid_2_0h_ps = []
avg_gp_error_hybrid_opt_0h_ps = []
avg_global_error_physics_0h_ps = []
avg_global_error_hybrid_1_0h_ps = []
avg_global_error_hybrid_2_0h_ps = []
avg_global_error_hybrid_opt_0h_ps = []


# TEST CELL an individual variable/level combination
# rmse_spatial_and_temporal_avg(.51,'t')

# ## LOOP THROUGH FUNCTION

# # FUNCTION INPUTS
# level_list_speedy = [.95,.51, .2]
# variable_list_speedy = ['t','u','v','q','ps'] #['u','t','v','q','ps'] 
level_list_speedy = [.95] 
variable_list_speedy = ['v'] #['u','t','v','q','ps'] 


for level in level_list_speedy:
    for variable in variable_list_speedy:
        if level != .95 and variable =='ps':
            break
        rmse_spatial_and_temporal_avg(level,variable)


print('avg_gp_error_physics_0h_t ', avg_gp_error_physics_0h_t)
print('avg_gp_error_hybrid_1_0h_t ', avg_gp_error_hybrid_1_0h_t)
print('avg_gp_error_hybrid_2_0h_t ', avg_gp_error_hybrid_2_0h_t)
print('avg_gp_error_hybrid_opt_0h_t ', avg_gp_error_hybrid_opt_0h_t)
print('\n')

print('avg_global_error_physics_0h_t ', avg_global_error_physics_0h_t)
print('avg_global_error_hybrid_1_0h_t ', avg_global_error_hybrid_1_0h_t)
print('avg_global_error_hybrid_2_0h_t ', avg_global_error_hybrid_2_0h_t)
print('avg_global_error_hybrid_opt_0h_t ', avg_global_error_hybrid_opt_0h_t)

print('avg_global_error_physics_0h_v ', avg_global_error_physics_0h_v)
print('avg_global_error_hybrid_1_0h_v ', avg_global_error_hybrid_1_0h_v)
print('avg_global_error_hybrid_2_0h_v ', avg_global_error_hybrid_2_0h_v)
print('avg_global_error_hybrid_opt_0h_v ', avg_global_error_hybrid_opt_0h_v)

print('avg_global_error_physics_0h_u ', avg_global_error_physics_0h_u)
print('avg_global_error_hybrid_1_0h_u ', avg_global_error_hybrid_1_0h_u)
print('avg_global_error_hybrid_2_0h_u ', avg_global_error_hybrid_2_0h_u)
print('avg_global_error_hybrid_opt_0h_u ', avg_global_error_hybrid_opt_0h_u)

print('avg_global_error_physics_0h_q ', avg_global_error_physics_0h_q)
print('avg_global_error_hybrid_1_0h_q ', avg_global_error_hybrid_1_0h_q)
print('avg_global_error_hybrid_2_0h_q ', avg_global_error_hybrid_2_0h_q)
print('avg_global_error_hybrid_opt_0h_q ', avg_global_error_hybrid_opt_0h_q)

print('avg_global_error_physics_0h_ps ', avg_global_error_physics_0h_ps)
print('avg_global_error_hybrid_1_0h_ps ', avg_global_error_hybrid_1_0h_ps)
print('avg_global_error_hybrid_2_0h_ps ', avg_global_error_hybrid_2_0h_ps)
print('avg_global_error_hybrid_opt_0h_ps ', avg_global_error_hybrid_opt_0h_ps)



# stop script here
print('\nFinished 0 hr forecast calculations\n')

# sys.exit()

'''


End 0 hr forecast calculation


Begin All other forecast calculations










Go 




'''

def extract_datetime_from_filename(filename):
    patterns = [
        # Matches '2011_2_10_0' or '2011_02_10_00' and similar SPEEDY format
        r'(\d{4})[_-](\d{1,2})[_-](\d{1,2})[_-](\d{1,2})',
        
        # Matches '08_09_2011_00' or '05_14_2011_12' from hybrid files
        r'(\d{1,2})[_-](\d{1,2})[_-](\d{4})[_-](\d{1,2})'
    ]
    
    for pattern in patterns:
        match = re.search(pattern, filename)
        if match:
            groups = match.groups()
            #print(f"Regex match for {filename}: {groups}")  # Debugging print
            
            if len(groups) == 4:
                try:
                    # Handle month/day order correctly for the SPEEDY case
                    # Ensure single digits are treated as zero-padded when necessary
                    if len(groups[0]) == 4:  # If month is a single digit, assume it's month
                        year, month, day, hour = groups
                    else:  # Otherwise, it's the format with zero-padded month
                        month, day, year, hour = groups
                    
                    # Check if the month and day are valid values
                    month = int(month)
                    day = int(day)
                    year = int(year)
                    hour = int(hour)

                    # Check if month is valid (1 to 12)
                    if not (1 <= month <= 12):
                        raise ValueError(f"Invalid month: {month}")

                    # Check if day is valid (1 to 31) based on the month
                    if not (1 <= day <= 31):
                        raise ValueError(f"Invalid day: {day}")

                    # Return a datetime object
                    return datetime(year, month, day, hour)
                except ValueError as e:
                    #print(f"Error in parsing datetime for {filename}: {e}")
                    continue
    
    # If no valid datetime is found
    return None



def get_files_by_datetime(directories, target_datetime):
    matching_files = []
    
    # Remove microseconds from target datetime
    target_datetime = target_datetime.replace(microsecond=0)
    
    # Iterate over each directory
    for directory in directories:
        #print(f"Scanning directory: {directory}")  # Debugging print
        # List all files in the directory
        for filename in os.listdir(directory):
            if "era_truth" in filename.lower():  # case-insensitive check
                continue  # Skip the current iteration and move to the next file
            file_path = os.path.join(directory, filename.strip())  # Stripping any whitespace

            # Check if it's a file
            if os.path.isfile(file_path):
                #print(f"Processing file: {filename.strip()}")  # Debugging print
                # Extract datetime from filename
                file_datetime = extract_datetime_from_filename(filename)
                
                #if file_datetime:
                    #print(f"Found datetime in file: {file_datetime}")  # Debugging print
                    #print(f"Target datetime: {target_datetime}")  # Debugging print
                
                # Remove microseconds from file datetime
                file_datetime = file_datetime.replace(microsecond=0) if file_datetime else None

                # Print the datetime comparison for each file
                #print(f"Comparing {file_datetime} with {target_datetime}")  # Debugging print

                # If the datetime matches the target, add the file to the list
                if file_datetime and file_datetime == target_datetime:
                    print(f"Match found: {file_path}")  # Debugging print
                    matching_files.append(file_path)
    
    return matching_files
        
        
#################### FORECASTS ######################
# load speedy files                                    
speedy_forecast_error_path = '/skydata2/dylanelliott/Predictions/SPEEDY_FORECASTS'
hybrid_1_9_forecast_path = '/skydata2/dylanelliott/Predictions/Hybrid/hybrid_1_9_forecasts' 
hybrid_1_9_1_9_forecast_path ='/skydata2/dylanelliott/Predictions/Hybrid/hybrid_1_9_1_9_forecasts'
era5_hybrid_forecast_path ='/skydata2/dylanelliott/Predictions/Hybrid/era_5_1_9_forecasts_retry_2_16_25'#'/skydata2/dylanelliott/Predictions/Hybrid/era_5_1_9_forecasts' 
        
        
# speedy_analysis_file = '/skydata2/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/speedy_1_9_20110101_20120101/mean_output/out.nc'
# hybrid_1_9_1_9_analysis_file = '/skydata2/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/hybrid_1_9_1_9_mem_1_fixed_20110101_20120115/out.nc'
# hybrid_1_9_1_9_1_7_analysis_file = '/skydata2/dylanelliott/backup_letkf_data_from_skynet/uniform_letkf_anal/hybrid_1_9_1_9_1_7_20110101_20120101/mean_output/out.nc'
# era5_hybrid_1_9_analysis_file = '/skydata2/dylanelliott/letkf-hybrid-speedy-from-Skynet-2/DATA/uniform_letkf_analysis/ERA5_hybrid_1_9_20110101_20120101/mean_output/out.nc'

######### now glob in the forecast files
# PATTERN file names
speedy_forecast_filenames = '/SPEEDY_10_day_forecast_*.nc'
hybrid_1_9_forecast_filenames = '/hybrid_prediction_era6000_20_20_20_beta_res1.0_beta_model_0.01_prior_0.0_overlap1_vertlevel_1_era5_uniform_29yr_training_cov_1_9_speedytrial_*.nc'
era5_hybrid_forecast_filenames = '/hybrid_prediction_era6000_20_20_20_beta_res1.0_beta_model_0.01_prior_0.0_overlap1_vertlevel_1_era5_29yr_training_6hr_timesteptrial_*.nc'
hybrid_1_9_1_9_forecast_filenames = '/hybrid_prediction_era6000_20_20_20_beta_res1.0_beta_model_0.01_prior_0.0_overlap1_vertlevel_1_era5_uniform_29yr_training_cov_1_9_1_9_hybridtrial_*.nc'

# creates a list of files
speedy_files = glob.glob(speedy_forecast_error_path + speedy_forecast_filenames)
hybrid_1_9_files = glob.glob(hybrid_1_9_forecast_path + hybrid_1_9_forecast_filenames)
era5_hybrid_forecast_files = glob.glob(era5_hybrid_forecast_path + era5_hybrid_forecast_filenames)
hybrid_1_9_1_9_files = glob.glob(hybrid_1_9_1_9_forecast_path + hybrid_1_9_1_9_forecast_filenames)

#print('speedy_files',speedy_files)
#print('hybrid_1_9_files',hybrid_1_9_files)
#print('era5_hybrid_forecast_files \n',era5_hybrid_forecast_files)

# speedy_files  # GOES OUT TO 2011- 12-31-12
# ['/skydata2/dylanelliott/Predictions/SPEEDY_FORECASTS/SPEEDY_10_day_forecast_2011_2_10_0.nc',

# hybrid_1_9_files  # GOES OUT to ONLY 12_22_2011_00
# ['/skydata2/dylanelliott/Predictions/Hybrid/hybrid_1_9_forecasts/hybrid_prediction_era6000_20_20_20_beta_res1.0_beta_model_0.01_prior_0.0_overlap1_vertlevel_1_era5_uniform_29yr_training_cov_1_9_speedytrial_05_14_2011_12.nc',

# era5_hybrid_forecast_files # GOES OUT to ONLY 12_22_2011_00
# '/skydata2/dylanelliott/Predictions/Hybrid/era_5_1_9_forecasts/hybrid_prediction_era6000_20_20_20_beta_res1.0_beta_model_0.01_prior_0.0_overlap1_vertlevel_1_era5_29yr_training_6hr_timesteptrial_08_09_2011_00.nc'

# hybrid_1_9_1_9_files # GOES OUT to ONLY 12_22_2011_00
# ['/skydata2/dylanelliott/Predictions/Hybrid/hybrid_1_9_1_9_forecasts/hybrid_prediction_era6000_20_20_20_beta_res1.0_beta_model_0.01_prior_0.0_overlap1_vertlevel_1_era5_uniform_29yr_training_cov_1_9_1_9_hybridtrial_05_14_2011_12.nc']

# LOAD in..
# VARIABLES AND LEVELS



# NO ! we want a general era5 verification array that we can access for all variables and levels later


# variable_speedy = 't'
   
# if variable_speedy == 't':
#     var_era = 'Temperature'
# else:
#     raise ValueError('Variable not supported')
# level = 0.2
# level_era = 2

## LOAD ERA5 VERIFICATION DATA
start_time = datetime(2011, 1, 1, 0)  
end_time = datetime(2011, 12, 31, 18)  
        
current_time = start_time
time_list_2011 = []
        
while current_time <= end_time:
    time_list_2011.append(current_time)
    current_time += timedelta(hours=6) # 6 for the truth data
                                                             
# for time in time_list_2011:
#     print(time.strftime('%Y-%m-%d %H:%M:%SZ'))

era5sets = []
timestep_6hrly = 6
for year in range(start_time.year, end_time.year + 1):
    nature_file = f'/skydata2/troyarcomano/ERA_5/{year}/era_5_y{year}_regridded_mpi_fixed_var.nc'
       #skydata2/troyarcomano/ERA_5/2011 
        # only load var_era selected and only load level_era selected from above
    # if variable_speedy == 'ps': # don't select level if variable is 'ps'
    #     ds_truth = xr.open_dataset(nature_file)[var_era]
    # else:
    ds_truth = xr.open_dataset(nature_file) #[var_era].sel(Sigma_Level=level_era)
    # Read in every 6th timestep
    ds_truth = ds_truth.isel(Timestep=slice(None, None, timestep_6hrly))
    era5sets.append(ds_truth)
   
print('Now its concatinating them all together...')
ds_truth = xr.concat(era5sets, dim = 'Timestep')
ds_truth = ds_truth.sortby('Timestep')
print('Done concat and sortby Timestep...')                       
# print('ds_truth inital_shape ', np.shape(ds_truth))
# assign the time dimension to the datetime values
ds_truth['Timestep'] = time_list_2011
# ds_truth = ds_truth.rename({'Timestep':'time'})
# ds_truth['time'] = time_list_2011
# print('ds_truth ', ds_truth) # Looks good!
# print('ds_truth q values ', ds_truth['Specific_Humidity']) # kg/kg

## DONE ERA5 VERIFICATION DATA

forecast_directories = [
    '/skydata2/dylanelliott/Predictions/SPEEDY_FORECASTS',
    '/skydata2/dylanelliott/Predictions/Hybrid/hybrid_1_9_forecasts',
    '/skydata2/dylanelliott/Predictions/Hybrid/era_5_1_9_forecasts_retry_2_16_25', #RETRY added
    '/skydata2/dylanelliott/Predictions/Hybrid/hybrid_1_9_1_9_forecasts'
]   

# A loop would start here to loop through the forecast times and calculate the forecast error

# trim time_list_2011 to Jan 7 @ 0 h,  to Dec 22 @ 0 h
trim_start = datetime(2011, 1, 7, 0)
trim_end = datetime(2011, 12, 22, 0) # DYLAN prob forgot to change this to run whole year 4.16 

main_time_list_2011 = []
while trim_start <= trim_end:
    main_time_list_2011.append(trim_start)
    trim_start += timedelta(hours=12) # 12 for the forecast start time file names

# print('main_time_list_2011[0:40] ', main_time_list_2011[0:40])



def forecast_error(lead_time,level_in_speedy,variable_speedy):
    
    if variable_speedy == 't':
        var_era = 'Temperature'
        var_era_hybrid = var_era
    elif variable_speedy == 'q':
        var_era = 'Specific_Humidity'
        var_era_hybrid = 'Specific-Humidity'
    elif variable_speedy == 'u':
        var_era = 'U-wind'
        var_era_hybrid = var_era
    elif variable_speedy == 'v':
        var_era = 'V-wind'
        var_era_hybrid = var_era
    elif variable_speedy == 'ps':
        var_era = 'logp'
        var_era_hybrid = var_era
    else:
        raise ValueError('Variable not supported')
    if level_in_speedy == 0.95:
        level_era = 7
    elif level_in_speedy == 0.51:
        level_era = 4
    elif level_in_speedy == 0.2:
        level_era = 2
    else:
        raise ValueError('Level not supported')
    
    # need inside loop to go from leadtime to cycle through 10 days
    inside_loop_lead_times = []
    start_inside_loop_lead_times = lead_time #+ timedelta(hours=6)DYLANCCHANGE HERE
    # bc each file is actually 6 h ahead, it just started from that time.
    end_inside_loop_lead_times = start_inside_loop_lead_times + timedelta(days=10)
    

    while start_inside_loop_lead_times <= end_inside_loop_lead_times:
        inside_loop_lead_times.append(start_inside_loop_lead_times)
        start_inside_loop_lead_times += timedelta(hours=6)
    # Check here if it worked
    inside_loop_lead_times = inside_loop_lead_times[:40]
    # print('inside_loop_lead_times ', inside_loop_lead_times) # looks good! but has minutes but whatever
    # print('np.shape(inside_loop_lead_times) ', np.shape(inside_loop_lead_times))
    # print('lead_time ', lead_time)
    # print('main_time_list_2011[0] ', main_time_list_2011[0])

    # open Truth (era5 file) at that time for 10 days
    # truth_file = ds_truth.sel(Sigma_Level=level_era,time = lead_time) #.sel(time = time_period)[var_era]

    matching_files = get_files_by_datetime(forecast_directories, lead_time) 
    # # Output the matching files
    # for file in matching_files:
    #     print(file)

    print('var era ========= ', var_era)
    print('level_era ======== ', level_era)
    print()

    # if variable_speedy == 'ps':
    #     print('speedy check xr array ', xr.open_dataset(matching_files[0])[var_era])
    #     print('speedy check xr array ', xr.open_dataset(matching_files[0])[var_era].values)
    # else:
    #     print('speedy check xr array ', xr.open_dataset(matching_files[0])[var_era].sel(Sigma_Level=level_era))  
    #     print('speedy values ', xr.open_dataset(matching_files[0])[var_era].sel(Sigma_Level=level_era).values)

    # print('speedy check np.exp1000', np.exp(xr.open_dataset(matching_files[0])[var_era]) * 1000.0)

    # print('hybrid_1_9_forecast check xr array ', xr.open_dataset(matching_files[1]) )

    # print('hybrid_1_9_forecast check xr array ', xr.open_dataset(matching_files[1]) )
    if variable_speedy == 'ps':
        speedy_forecast = np.exp ( xr.open_dataset(matching_files[0])[var_era]) * 1000.0
        hybrid_1_9_forecast = np.exp ( xr.open_dataset(matching_files[1])[var_era_hybrid] ) * 1000.0
        era5_hybrid_forecast = np .exp (xr.open_dataset(matching_files[2])[var_era_hybrid]) * 1000.0
        hybrid_1_9_1_9_forecast = np.exp(xr.open_dataset(matching_files[3])[var_era_hybrid]) * 1000.0
        
        truth_data = np.exp(ds_truth[var_era]) * 1000.0

    elif variable_speedy == 'q':
        speedy_forecast = xr.open_dataset(matching_files[0])[var_era].sel(Sigma_Level=level_era) 
        hybrid_1_9_forecast = xr.open_dataset(matching_files[1])[var_era_hybrid].sel(Sigma_Level=level_era) 
        era5_hybrid_forecast = xr.open_dataset(matching_files[2])[var_era_hybrid].sel(Sigma_Level=level_era) 
        hybrid_1_9_1_9_forecast = xr.open_dataset(matching_files[3])[var_era_hybrid].sel(Sigma_Level=level_era) 
        
        truth_data = ds_truth[var_era].sel(Sigma_Level=level_era) * 1000.0
    else:
        speedy_forecast = xr.open_dataset(matching_files[0])[var_era].sel(Sigma_Level=level_era)
        hybrid_1_9_forecast = xr.open_dataset(matching_files[1])[var_era_hybrid].sel(Sigma_Level=level_era)
        era5_hybrid_forecast = xr.open_dataset(matching_files[2])[var_era_hybrid].sel(Sigma_Level=level_era)
        hybrid_1_9_1_9_forecast = xr.open_dataset(matching_files[3])[var_era_hybrid].sel(Sigma_Level=level_era)
        
        truth_data = ds_truth[var_era].sel(Sigma_Level=level_era)

    # print('speedy_forecast before ', speedy_forecast)
    speedy_forecast = speedy_forecast.rename({'time': 'Timestep'})
    # print('speedy_forecast ', speedy_forecast)

    speedy_forecast.coords['Timestep'] = inside_loop_lead_times
    hybrid_1_9_forecast.coords['Timestep'] = inside_loop_lead_times
    era5_hybrid_forecast.coords['Timestep'] = inside_loop_lead_times
    hybrid_1_9_1_9_forecast.coords['Timestep'] = inside_loop_lead_times

    # print('hybrid_1_9_forecast AFTER ', hybrid_1_9_forecast)
    # Create a NumPy array from the list
    forecast_index_desired = [4,8,12,20,40]#[1,2,4,8,12]# error maps -> [4,8,12,20,40]#[1,2,4,8,12,20,40] #np.array([6, 12, 24, 48, 72, 120, 240])
    # all_forecast_indices = np.arange(1,41)

    i = 1 
    for inner_lead_time in inside_loop_lead_times:

        # if i not in all_forecast_indices:    
        #     print(f"Not a forecast time I want, skipping {i}..")
        #     # continue
        #     i += 1
        if i not in forecast_index_desired: # only forecast times I want 
            # print(f"Not a forecast time I want, skipping {i}..")
            i += 1
        else:
        
            # call on that specific time we NEED

            print('inner_lead_time ', inner_lead_time)
            speedy_forecast_inside = speedy_forecast.sel(Timestep = inner_lead_time)
            hybrid_1_9_forecast_inside = hybrid_1_9_forecast.sel(Timestep = inner_lead_time)
            era5_hybrid_forecast_inside = era5_hybrid_forecast.sel(Timestep = inner_lead_time)
            hybrid_1_9_1_9_forecast_inside = hybrid_1_9_1_9_forecast.sel(Timestep = inner_lead_time)
            # adjust here for lowest rmse
            truth_data_inside = truth_data.sel(Timestep = inner_lead_time +timedelta(hours=6)) # makes sense because speedy 0h is included, (41 timesteps)
            truth_data_inside_for_hybrid = truth_data.sel(Timestep = inner_lead_time - timedelta(hours=12)) # o.g. -12 hours

            # speedy is correct at 2.87 (+ 6 h)
            # v_0.95_6h, speedy = 2.8772564

            # hybrid is wrong at 5.23       (+ 6h)
            # v_0.95_6h, hybrid_1_9 = 4.5298634 (+ 0 h)
            # v_0.95_6h, hybrid_1_9 = 3.5691423 (- 6 h)
            # BOOM ITS - 12 h
            # v_0.95_6h, hybrid_1_9 = 2.834367 (- 12 h)
            # v_0.95_6h, hybrid_1_9 = 3.301713 (- 18 h)

            lats = truth_data.Lat
            xgrid = 96
            ygrid = 48

            # calculate spatial and global rmse for each
            speedy_rmse_inside = latituded_weighted_rmse(truth_data_inside, speedy_forecast_inside,lats)
            hybrid_1_9_rmse_inside = latituded_weighted_rmse(truth_data_inside_for_hybrid, hybrid_1_9_forecast_inside,lats)
            era5_hybrid_rmse_inside = latituded_weighted_rmse(truth_data_inside_for_hybrid, era5_hybrid_forecast_inside,lats)
            hybrid_1_9_1_9_rmse_inside = latituded_weighted_rmse(truth_data_inside_for_hybrid, hybrid_1_9_1_9_forecast_inside,lats)
            
            speedy_error_inside = np.zeros((ygrid,xgrid))
            hybrid_1_9_error_inside = np.zeros((ygrid,xgrid))
            era5_hybrid_error_inside = np.zeros((ygrid,xgrid))
            hybrid_1_9_1_9_error_inside = np.zeros((ygrid,xgrid))

            # print('speedy_rmse_inside ', speedy_rmse_inside)
            # print('hybrid_1_9_rmse_inside ', hybrid_1_9_rmse_inside)
            # print('era5_hybrid_rmse_inside ', era5_hybrid_rmse_inside)
            # print('hybrid_1_9_1_9_rmse_inside ', hybrid_1_9_1_9_rmse_inside)

            speedy_error_inside[:,:] = np.sqrt((speedy_forecast_inside[:,:] - truth_data_inside[:,:])**2.0)
            hybrid_1_9_error_inside[:,:] = np.sqrt((hybrid_1_9_forecast_inside[:,:] - truth_data_inside_for_hybrid[:,:])**2.0)
            era5_hybrid_error_inside[:,:] = np.sqrt((era5_hybrid_forecast_inside[:,:] - truth_data_inside_for_hybrid[:,:])**2.0)
            hybrid_1_9_1_9_error_inside[:,:] = np.sqrt((hybrid_1_9_1_9_forecast_inside[:,:] - truth_data_inside_for_hybrid[:,:])**2.0)

            # print('speedy_error_inside ', speedy_error_inside)
            # print('hybrid_1_9_error_inside ', hybrid_1_9_error_inside)
            # print('era5_hybrid_error_inside ', era5_hybrid_error_inside)
            # print('hybrid_1_9_1_9_error_inside ', hybrid_1_9_1_9_error_inside)
            
            index_in_hours = i*6

            key = f'{variable_speedy}_{level_in_speedy}_{index_in_hours}h'
            print('key ', key)

            # allow list
            if key not in rmse_dict:
                rmse_dict[key] = {
                    'speedy': [],
                    'hybrid_1_9': [],
                    'era5_hybrid': [],
                    'hybrid_1_9_1_9': []
                }
            if key not in error_dict:
                error_dict[key] = {
                    'speedy': [],
                    'hybrid_1_9': [],
                    'era5_hybrid': [],
                    'hybrid_1_9_1_9': []
                }

            # append the rmse values to the dictionary
            rmse_dict[key]['speedy'].append(speedy_rmse_inside)
            rmse_dict[key]['hybrid_1_9'].append(hybrid_1_9_rmse_inside)
            rmse_dict[key]['era5_hybrid'].append(era5_hybrid_rmse_inside)
            rmse_dict[key]['hybrid_1_9_1_9'].append(hybrid_1_9_1_9_rmse_inside)

            error_dict[key]['speedy'].append(speedy_error_inside)
            error_dict[key]['hybrid_1_9'].append(hybrid_1_9_error_inside)
            error_dict[key]['era5_hybrid'].append(era5_hybrid_error_inside)
            error_dict[key]['hybrid_1_9_1_9'].append(hybrid_1_9_1_9_error_inside)

            # now at the end of the inside loop, store the rmse values in a dictionary
            print('Inside loop lead time ', inner_lead_time, ' complete')
            i += 1

    print(f'\nDone with lead time: {lead_time}. \n')


variable_list_speedy = ['t','q','u','v','ps']

# variable_list_speedy = ['v']
# variable_list_era = ['Temperature','Specific_humidity','U_wind','V_wind','Pressure']

level_list = [0.95, 0.51, 0.2]
# level_list = [0.95]#, 0.51, 0.2]
# level_era_list = [7, 4, 2]

# Initialize dictionaries to store RMSE and errors
rmse_dict = {}
error_dict = {}

# BEGIN MAIN LOOP
enddate_datetime = datetime(2011,12,22,0) #datetime(2011, 12, 22, 0) # 12 22 is furthest out we can go 
# dont have start dates past that because hybrid code stopped at 12/22/2011 12 h

for time in main_time_list_2011:
    if time == datetime(2011,1,14,0): # have to skip because SPEEDY crashed at this time
        continue
    if time == enddate_datetime: # works!
        break
    for variable in variable_list_speedy:
        for level in level_list:
            if level != .95 and variable =='ps':
                break
            forecast_error(time,level,variable)

#             # and then at this time t We need to loop through 10 days of times and
            # get the forecast error w.r.t. the truth @ 6 h, 12 h, 24 h, 48 h, 72 h, 120 h, 240 h


# ### Run one time to test ###
# lead_time = main_time_list_2011[10]
# print('lead_time ', lead_time)
# level_in_speedy = 0.95
# variable_speedy = 't'
# forecast_error(lead_time,level_in_speedy,variable_speedy) # works!
###########################

# print('rmse_dict ', rmse_dict)
# print out dictionary in a very organized way
# for key in rmse_dict:
#     print(key)
#     for model in rmse_dict[key]:
#         print(model, rmse_dict[key][model])
#     print('\n')

print('dylan here avg test')

#save those averages to a dictionary
rmse_dict_avg = {}
error_dict_avg = {}

print('rmse_dict \n')
for key in rmse_dict:
    rmse_dict_avg[key] = {}
    for model in rmse_dict[key]:
        rmse_dict_avg[key][model] = np.average(rmse_dict[key][model])
        print(f'{key}, {model} =', rmse_dict_avg[key][model])

# print('error_dict \n')
for key in error_dict:
    error_dict_avg[key] = {}
    print('DYLAN key ', key)

    for model in error_dict[key]:
        # error_dict_avg[key][model] = np.average(error_dict[key][model])
        print(type(error_dict[key][model]))
        try:
            print(len(error_dict[key][model]))
        except:
            print('fail get len')
        # print('error_dict[key][model] ', error_dict[key][model])
        error_dict[key][model] = np.array(error_dict[key][model])
        # print('afterchange to numpy array')
        # print('error_dict[key][model] ', error_dict[key][model])


        error_dict_avg[key][model] = np.average(abs(error_dict[key][model][::,:,:]),axis=0) # rmse
        
        # print('error_dict_avg[key][model] ', error_dict_avg[key][model])
        # print('np.shape() error_dict_avg[key][model] ', np.shape(error_dict_avg[key][model]))
        
# take avg of each error_dict_avg key
print(f'{key}, {model} =', error_dict_avg[key][model])
# print('np.shape error_dict_avg[key][model] ', np.shape(error_dict_avg[key][model]))


# print('avg(error_dict_avg test\n')

for key in error_dict_avg:
    print(key,'\n')
    for model in error_dict_avg[key]:
        print(model,'average axis = 0 ,', np.average(np.average(error_dict_avg[key][model], axis=0),axis=0))
        # print('np.shape() error_dict_avg[key][model] ', np.shape(error_dict_avg[key][model]))


    


# print('error_dict ', error_dict)
# print('rmse_dict keys ', rmse_dict.keys())
# # print('error_dict keys ', error_dict.keys())

# print('rmse_dict[t_0.95_6h] ', rmse_dict['t_0.95_6h'])

# print('rmse_dict[t_0.95_12h][speedy] ', rmse_dict['t_0.95_12h']['speedy'])

# print('rmse_dict[t_0.95_24h][speedy] ', rmse_dict['t_0.95_24h']['speedy'])
# print('rmse_dict[t_0.95_24h][hybrid_1_9] ', rmse_dict['t_0.95_24h']['hybrid_1_9'])
# print('rmse_dict[t_0.95_24h][era5_hybrid] ', rmse_dict['t_0.95_24h']['era5_hybrid'])
# print('rmse_dict[t_0.95_24h][hybrid_1_9_1_9] ', rmse_dict['t_0.95_24h']['hybrid_1_9_1_9'])

# print('error_dict[t_0.95_24h][speedy] ', error_dict['t_0.95_24h']['speedy'])
# print('error_dict[t_0.95_24h][hybrid_1_9] ', error_dict['t_0.95_24h']['hybrid_1_9'])
# print('error_dict[t_0.95_24h][era5_hybrid] ', error_dict['t_0.95_24h']['era5_hybrid'])
# print('error_dict[t_0.95_24h][hybrid_1_9_1_9] ', error_dict['t_0.95_24h']['hybrid_1_9_1_9'])


# print('rmse_dict[t_0.95_12h] ', rmse_dict['t_0.95_12h'])

# print('averages')

# print('rmse_dict_avg[v_0.95_6h][speedy] ', rmse_dict_avg['v_0.95_6h']['speedy'])
# print('rmse_dict_avg[v_0.95_6h][hybrid_1_9] ', rmse_dict_avg['v_0.95_6h']['hybrid_1_9'])
# print('rmse_dict_avg[v_0.95_6h][era5_hybrid] ', rmse_dict_avg['v_0.95_6h']['era5_hybrid'])
# print('rmse_dict_avg[v_0.95_6h][hybrid_1_9_1_9] ', rmse_dict_avg['v_0.95_6h']['hybrid_1_9_1_9'])


# print('check DYLAN 4.15 grid point errors')

# print('error_dict_avg[v_0.95_6h][speedy] ', error_dict_avg['v_0.95_6h']['speedy'])
# print('error_dict_avg[v_0.95_6h][hybrid_1_9] ', error_dict_avg['v_0.95_6h']['hybrid_1_9'])
# print('error_dict_avg[v_0.95_6h][era5_hybrid] ', error_dict_avg['v_0.95_6h']['era5_hybrid'])
# print('error_dict_avg[v_0.95_6h][hybrid_1_9_1_9] ', error_dict_avg['v_0.95_6h']['hybrid_1_9_1_9'])

print('\n')
# print all available keys
print('error_dict keys ', error_dict.keys())
for key in error_dict:
    print(key)
    for model in error_dict[key]:
        print(model, error_dict[key][model])
    print('\n')

############################################
# Plot Maps of RMSE
# import matplotlib.pyplot as plt
# import numpy as np
# import cartopy.crs as ccrs
# from matplotlib import gridspec
# from matplotlib import cm
print('dylan here before plot')

############################################


def plot_rmse_maps_new(selected_level, variable):
    # Coordinates for lat and lon
    lat = ds_analysis_mean.lat.values
    lon = ds_analysis_mean.lon.values

    # Create the grid for latitudes and longitudes
    lons2d, lats2d = np.meshgrid(lon, lat)

    # List of hours to plot
    # hours = [6, 12, 24, 48, 72] # Dylan debug here [1,2,4,8,12]
    hours = [24, 48, 72, 120, 240]
    # error plot -> hours = [24, 48, 72, 120, 240]
    
    #USING ORDERED KEYS HERE NOW, defined below
    models = ['PHYSICS', 'HYBRID 1', 'HYBRID 2','HYBRID-OPT']

    fig = plt.figure(figsize=(9.3, 6.5)) #7.5
    width_ratio_set = .972
    gs = gridspec.GridSpec(nrows=len(hours), ncols=len(models), width_ratios=[width_ratio_set,  width_ratio_set, width_ratio_set, 1])

    ipickcolormap = 'viridis'#'magma_r' #'gist_heat_r'
    fs = 12  # Font size for titles and labels
    fontsize_labels = 9

    # Set the title based on the variable and level
    if variable == 'ps':
        fig.suptitle(f'Global RMSE of Surface Pressure', fontsize=fs)
    elif variable == 't':
        fig.suptitle(f'Global RMSE of Temperature at {selected_level} Sigma', fontsize=fs)
    elif variable == 'q':
        fig.suptitle(f'Global RMSE of Specific Humidity at {selected_level} Sigma', fontsize=fs)
    elif variable == 'u':
        fig.suptitle(f'Global RMSE of U-wind at {selected_level} Sigma', fontsize=fs)
    elif variable == 'v':
        fig.suptitle(f'Global RMSE of V-wind at {selected_level} Sigma', fontsize=fs)
    else:
        raise


    # Loop through each hour and plot the RMSE for each model
    for i, hour in enumerate(hours):
        # Fetch the model data for this hour
        key = f'{variable}_{selected_level}_{hour}h'
        print('key ', key)  
        if key not in error_dict_avg:
            print(f"No data for {key}")
            continue
        
        print('error_dict_avg[key] ', error_dict_avg[key])
        data = error_dict_avg[key] 
        # print('data', data)
         # Assuming error_dict_avg[hour] contains the RMSE data
        #print keys in data
        print('data keys ', data.keys())

        # REORDER THE KEYS
        ordered_keys = [key for key in data.keys() if key != 'era5_hybrid'] + ['era5_hybrid']
        print('ordered_keys ', ordered_keys)

        # define the max value for the imshow
        
        if variable == 't':
            if selected_level == .95:
                imshow_v_max = 8
            elif selected_level == .51:
                imshow_v_max = 8
            elif selected_level == .2:
                imshow_v_max = 10
            imshow_units = 'K' 
        elif variable == 'q':
            imshow_units = 'g/kg'
            if selected_level == .95:
                imshow_v_max = 4
            elif selected_level == .51:
                imshow_v_max = 2
            elif selected_level == .2:
                imshow_v_max = 0.2
        elif variable == 'u':
            if selected_level == .2:
                imshow_v_max = 50
            if selected_level == .95:
                imshow_v_max = 10
            else:
                imshow_v_max = 20
            imshow_units = 'm/s'
        elif variable == 'v':
            if selected_level == .2:
                imshow_v_max = 50
            if selected_level == .95:
                imshow_v_max = 10
            else:           
                imshow_v_max = 20
            imshow_units = 'm/s'
        elif variable == 'ps':
            imshow_v_max = 100
            imshow_units = 'hPa'
        else:
            raise ValueError('Variable not supported')

        # Loop through models and plot

        for j, model in enumerate(ordered_keys): #enumerate(data.keys()):
            # if variable == 'v' or 't':
            #     print( 'j', j)
            #     print('model ', model)

            model_data = data[model]

            # if variable == 'v' or 't':
            #     print('model_data ', model_data)

            cyclic_data, cyclic_lons = add_cyclic_point(model_data, coord=lon)
            lons2d, lats2d = np.meshgrid(cyclic_lons, lat)

            ax = fig.add_subplot(gs[i, j], projection=ccrs.PlateCarree())
            img = ax.imshow(cyclic_data, 
                            extent=(lons2d.min(), lons2d.max(), lats2d.max(), lats2d.min()), 
                            vmin=0, vmax=imshow_v_max, cmap=ipickcolormap, 
                            transform=ccrs.PlateCarree(), interpolation='none')
            
            ax.coastlines()

            if i == 0:
                ax.set_title(f'{models[j]}', fontsize=fs)

            if j == 0:
                ax.set_ylabel(f'{hour} h', fontsize=fs)   
                ax.set_yticks([-90, -45, 0, 45, 90])
                ax.set_yticklabels(['90\u00B0 S', '45\u00B0 S', '0\u00B0', '45\u00B0 N', '90\u00B0 N'],fontsize = fontsize_labels)
            
            if i == len(hours) - 1:
                if j != len(models) - 1:
                    ax.set_xticks([-180, -90, 0, 90, 180])
                    ax.set_xticklabels(['180\u00B0', '90\u00B0 W', '0\u00B0', '90\u00B0 E', ''], fontsize = fontsize_labels)
                else:
                    ax.set_xticks([-180, -90, 0, 90, 180])
                    ax.set_xticklabels(['180\u00B0', '90\u00B0 W', '0\u00B0', '90\u00B0 E', '180\u00B0'], fontsize = fontsize_labels)
            
            if j == len(models) - 1:
                cb = plt.colorbar(img, ax=ax,orientation='vertical', fraction=0.0244, pad=0.005)
                cb.set_label(imshow_units, fontsize=fs)
                num_ticks = 3
                ticks = np.linspace(0, imshow_v_max, num_ticks)
                cb.set_ticks(ticks)
                cb.ax.tick_params(labelsize=fontsize_labels)
                # plt.colorbar(img, ax=ax, orientation='vertical')#, pad=0.05, aspect=50)

    plt.tight_layout()
    plt.subplots_adjust(hspace=0.01) #0.006
    plt.subplots_adjust(wspace=0.01)
    # get todays datetime
    today = datetime.today().strftime('%Y-%m-%d')
    save_directory = f'/skydata2/dylanelliott/plotting_scripts_forecast_results_paper/forecast_error_maps_{today}'
    if not os.path.exists(save_directory):
        os.makedirs(save_directory)
    plt.savefig(f'{save_directory}/rmse_{variable}_{selected_level}.pdf', dpi=300)
    # plt.show()
    plt.close()

# plot_rmse_maps_new(0.2, 't')

# loop it

for variable in variable_list_speedy:
    for level in level_list:
        print('variable ', variable, '\n')
        print('level ', level, '\n')
        if level != .95 and variable =='ps':
            break
        #plot_rmse_maps_new(level, variable)
        continue
        
############################################
print('plotting done')


print('\nDONE with py script.\n')


