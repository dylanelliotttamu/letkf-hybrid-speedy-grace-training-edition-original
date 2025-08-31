import sys
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os

# load data
forecast_path = '/Users/dylanelliott/Desktop/plots_for_paper/rmse_6hrly_globally_averaged.txt' #rmse_whole_year.txt'

df = pd.read_csv(forecast_path, sep='\t', header=None)

#print(df.head())
#print(df.tail())

df_split = df[0].str.split('=', expand=True)

df['forecast_label'] = df_split[0]

df['rmse'] = df_split[1].astype(str)

# Now, #print the updated DataFrame to check the changes
#print(df.head())


import re

def extract_components(forecast_label):
    # Clean up any trailing spaces or escape characters
    forecast_label = forecast_label.strip().replace("\\", "")
    
    # Regex to match variable, level, hour, and model
    # ps_0.95_240h, speedy
    pattern = r'([a-zA-Z]+)_(\d+\.\d+)_(\d+h?)(?:, ([a-zA-Z_0-9]+))?'
    
    match = re.match(pattern, forecast_label)
    if match:
        variable = match.group(1)
        level = match.group(2)
        hour = match.group(3)
        model = match.group(4) if match.group(4) else None
        return variable, level, hour, model
    return None, None, None, None  # If no match, return None for all components

df[['variable', 'level', 'hour', 'model']] = df['forecast_label'].apply(lambda x: pd.Series(extract_components(x)))

#print(df)

#print(df['variable'])

# clean rmse into float
# Remove any trailing spaces or escape characters
#print(df['rmse'])

df['rmse'] = df['rmse'].str.replace(r'\\$', '', regex=True)


df['rmse'] = df['rmse'].astype(float)
# print(df['rmse'])
# print(df.head())

df = df.drop(columns=[0, 'forecast_label'])
# print(df.head())

model_colors = {
    "speedy": "blue",
    "hybrid_1_9": "red",
    "era5_hybrid": "black",
    "hybrid_1_9_1_9": "green"
}

plot_labels = {
    "speedy": "PHYSICS",
    "hybrid_1_9": "HYBRID 1",
    "era5_hybrid": "HYBRID-OPT",
    "hybrid_1_9_1_9": "HYBRID 2"
}


avg_global_error_physics_0h_t = [2.2202023610597204, 1.1779967758980967, 1.5733649225122086]
avg_global_error_hybrid_1_0h_t = [2.2731451093320394, 1.1448758511822203, 1.2911185026168823]
avg_global_error_hybrid_2_0h_t = [2.519356803060574, 1.159647818063295, 1.2598075706480605]
avg_global_error_hybrid_opt_0h_t = [1.192508150649602, 0.9102834808046107, 1.0181153005295145]
avg_global_error_physics_0h_v = [2.762787265697894, 3.0877849174077134, 3.5216254482362266]
avg_global_error_hybrid_1_0h_v = [2.724988056258571, 2.880736772562468, 3.0970945134136336]
avg_global_error_hybrid_2_0h_v = [2.7944294715658202, 2.8900185229054403, 3.0630123781294545]
avg_global_error_hybrid_opt_0h_v = [1.8542371090906244, 2.64725668881929, 2.7547882752166153]
avg_global_error_physics_0h_u = [2.7194891216695143, 3.056445935144398, 3.6961653325551067]
avg_global_error_hybrid_1_0h_u = [2.6837959868994266, 2.8695884413373838, 3.2956424423579054]
avg_global_error_hybrid_2_0h_u = [2.756721772191252, 2.8744754867633406, 3.2556930019994965]
avg_global_error_hybrid_opt_0h_u = [1.7748983694152247, 2.6044155365909374, 2.862445031201939]

avg_global_error_physics_0h_q = [0.0014313763706499975, 0.0006457215566920962, 4.048066854051802e-05]
avg_global_error_hybrid_1_0h_q = [0.0014305536028498414, 0.0006184144574914693, 3.959279744515781e-05]
avg_global_error_hybrid_2_0h_q = [0.0015496386620112302, 0.0005985946191586995, 4.0637614513299246e-05]
avg_global_error_hybrid_opt_0h_q = [0.0007371786352886473, 0.0005489390657449072, 9.96244204206529e-06]
avg_global_error_physics_0h_q = [x * 1000 for x in avg_global_error_physics_0h_q]
avg_global_error_hybrid_1_0h_q = [x * 1000 for x in avg_global_error_hybrid_1_0h_q]
avg_global_error_hybrid_2_0h_q = [x * 1000 for x in avg_global_error_hybrid_2_0h_q]
avg_global_error_hybrid_opt_0h_q = [x * 1000 for x in avg_global_error_hybrid_opt_0h_q]

avg_global_error_physics_0h_ps = [17.548820044004817]
avg_global_error_hybrid_1_0h_ps = [17.575358511677692]
avg_global_error_hybrid_2_0h_ps = [17.63742478909931]
avg_global_error_hybrid_opt_0h_ps = [1.2329773954362258]


    

# need to add those 0h errors to df
# 0th index is 0.95 sigma, 1st is 0.51, 2nd is 0.2

print('BEFORE')
print(df.head())
print(df.tail())

model_names = ['physics', 'hybrid_1', 'hybrid_2', 'hybrid_opt']
convert_name = ['speedy', 'hybrid_1_9', 'hybrid_1_9_1_9', 'era5_hybrid']
variables = ['t', 'v', 'u', 'q', 'ps']
row_index = [0, 1, 2]
sigma_level = ['0.95', '0.51', '0.2']

for variable in variables:
    for i, mn in enumerate(model_names):
        if variable == 'ps':
            error_variable = f'avg_global_error_{mn}_0h_{variable}'
            error_value = globals()[error_variable][0]
            df = pd.concat([df, 
                            pd.DataFrame({'variable': [variable], 
                                          'level': [sigma_level[0]],
                                            'hour': ['0h'], 'model': [convert_name[i]],
                                              'rmse': [error_value]})], ignore_index=True)
        else:
            for row in row_index:
                error_variable = f'avg_global_error_{mn}_0h_{variable}'

                error_value = globals()[error_variable][row]
                
                df = pd.concat([df, pd.DataFrame({'variable': [variable],
                                                   'level': [sigma_level[row]], 'hour': ['0h'],
                                                     'model': [convert_name[i]], 'rmse': [error_value]})], ignore_index=True)

print('AFTER')
print(df.head())
print(df.tail())

df['hour'] = df['hour'].astype(str)
df['rmse'] = df['rmse'].astype(float)


unique_combinations = df[['variable', 'level']].drop_duplicates()
print('unique_combinations ', unique_combinations)

print('df.iloc[0]\n', df.iloc[0])
print('df.iloc[-1]\n', df.iloc[-4])

# Convert the 'hour' column to a numeric format (remove 'h' and convert to integer)
df['hour_numeric'] = ( df['hour'].str.replace('h', '').astype(int) )
# print('df[hour_numeric]\n', df['hour_numeric'])

# Now, in your plotting code, make sure you sort the DataFrame by 'hour_numeric' before plotting
for _, row in unique_combinations.iterrows():
    variable = row['variable']
    level = row['level']
    
    subset = df[(df['variable'] == variable) & (df['level'] == level)]

    # Sort by the new 'hour_numeric' column to ensure the correct order
    subset = subset.sort_values(by='hour_numeric')

    plt.figure(figsize=(10, 6))

    for model in subset['model'].unique():
        print(model)
        model_subset = subset[subset['model'] == model]
        print(model_subset)
        color = model_colors.get(model, "gray")  # Default to gray if model not in color map
        label = plot_labels.get(model, model)  # Default to model name if not in label map
        # plt.plot(model_subset['hour'], model_subset['rmse'], label=label, color=color)
        plt.plot(model_subset['hour_numeric'] , model_subset['rmse'], label=label, color=color)
    
    plt.xlabel('Hour')

    if variable == 'ps':
        plt.ylabel('RMSE (hPa)')
        plt.title(f'Global RMSE of Surface Pressure')
    elif variable == 'q':
        plt.ylabel('RMSE (g/kg)')
        plt.title(f'Global RMSE of Specific Humidity at Sigma {level}')
    elif variable == 'u' or variable == 'v':
        plt.ylabel('RMSE (m/s)')
        if variable == 'u':
            plt.title(f'Global RMSE of U-wind at Sigma {level}')
        if variable == 'v':
            plt.title(f'Global RMSE of V-wind at Sigma {level}')
    elif variable == 't':
        plt.ylabel('RMSE (K)')
        plt.title(f'Global RMSE of Temperature at Sigma {level}')
    else:
        raise

    # show only every other x-tick
    plt.xticks(np.arange(0,246,step=12), rotation=45)
    #change x-tick labels to be equal to the hour

    # plt.xticks(rotation=45)
    plt.xlim(0, 240)
    plt.ylim(0, ) # auto max
    plt.grid(color='grey', linestyle='--', linewidth=0.1)
    plt.legend()

    plt.tight_layout()
    # plt.show()
    # os make directory if doesn't exist
    # os.makedirs(f'/Users/dylanelliott/Desktop/plots_6hrly_4_1_2025', exist_ok=True)
    # plt.savefig(f'/Users/dylanelliott/Desktop/plots_6hrly_4_1_2025/forecast_error_{variable}_{level}.pdf', dpi=300)
    plt.close()



# Assuming model_colors and plot_labels are already defined
# df, model_colors, and plot_labels are already defined

# Define lists for each variable's subsets by level
temp_list = [
    df[(df['variable'] == 't') & (df['level'] == '0.2')],
    df[(df['variable'] == 't') & (df['level'] == '0.51')],
    df[(df['variable'] == 't') & (df['level'] == '0.95')]
]
v_list = [
    df[(df['variable'] == 'v') & (df['level'] == '0.2')],
    df[(df['variable'] == 'v') & (df['level'] == '0.51')],
    df[(df['variable'] == 'v') & (df['level'] == '0.95')]
]
u_list = [
    df[(df['variable'] == 'u') & (df['level'] == '0.2')],
    df[(df['variable'] == 'u') & (df['level'] == '0.51')],
    df[(df['variable'] == 'u') & (df['level'] == '0.95')]
]
q_list = [
    df[(df['variable'] == 'q') & (df['level'] == '0.2')],
    df[(df['variable'] == 'q') & (df['level'] == '0.51')],
    df[(df['variable'] == 'q') & (df['level'] == '0.95')]
]

def make_subplots(list_of_df_combos):
    if list_of_df_combos is temp_list:
        variable = 't'
    elif list_of_df_combos is v_list:
        variable = 'v'
    elif list_of_df_combos is u_list:
        variable = 'u'
    elif list_of_df_combos is q_list:
        variable = 'q'
                         
                                                 
    # Create a figure for the subplots (3 rows, 1 column)
    fig, axs = plt.subplots(3, 1, figsize=(10, 18))  # Adjust size as necessary
    fig.tight_layout(pad=4.0)  # Adjust padding between subplots


    # Iterate over the temp_list for each level
    for idx, subset in enumerate(list_of_df_combos):#enumerate(temp_list):
        # Determine the axis for the current subplot (0: top, 1: middle, 2: bottom)
        ax = axs[idx]
        
        level = subset['level'].iloc[0]  # Get level from the first row (since all rows have the same level)
        # variable = 't'  # As we're working with temperature ('t')
        
        # Sort by the new 'hour_numeric' column to ensure the correct order
        subset = subset.sort_values(by='hour_numeric')

        # Plot data for each model
        for model in subset['model'].unique():
            model_subset = subset[subset['model'] == model]
            color = model_colors.get(model, "gray")  # Default to gray if model not in color map
            label = plot_labels.get(model, model)  # Default to model name if not in label map
            ax.plot(model_subset['hour_numeric'], model_subset['rmse'], label=label, color=color)
        
        sup_level = .99
        # Set titles and labels based on the variable
        if variable == 'ps':
            ax.set_ylabel('RMSE (hPa)')
            ax.set_title(f'Global RMSE of Surface Pressure')
        elif variable == 'q':
            if idx == 1:
                plt.suptitle('Global RMSE of Specific Humidity', y=sup_level)
            ax.set_ylabel('RMSE (g/kg)')
            ax.set_title(f'Sigma {level}')
        elif variable == 'u' or variable == 'v':
            ax.set_ylabel('RMSE (m/s)')
            if variable == 'u':
                ax.set_title(f'Sigma {level}')
                if idx == 1:
                    plt.suptitle('Global RMSE of U-wind', y=sup_level)
            if variable == 'v':
                ax.set_title(f'Sigma {level}')
                if idx == 1:
                    plt.suptitle('Global RMSE of V-wind', y=sup_level)
        elif variable == 't':
            ax.set_ylabel('RMSE (K)')
            ax.set_title(f'Sigma {level}')
            if idx == 1:
                plt.suptitle('Global RMSE of Temperature',y=sup_level)
        else:
            raise ValueError(f"Unexpected variable: {variable}")

        ax.set_xticks(np.arange(0, 246, step=12))
        ax.set_xticklabels([f'{int(i)}h' for i in np.arange(0, 246, step=12)], rotation=45)

        ax.set_xlim(0, 240)
        ax.set_ylim(0, )  # Auto max y-limit

        # Add grid, legend, and labels
        ax.grid(color='grey', linestyle='--', linewidth=0.1)
        ax.legend()
    # plt.show()

    plt.tight_layout()
    output_dir = '/Users/dylanelliott/Desktop/subplots_test_4_1_2025'
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f'{output_dir}/forecast_error_{variable}.pdf', dpi=300)

    plt.close()
    print(f'Finished plotting {variable}')

make_subplots(temp_list)
make_subplots(v_list)
make_subplots(u_list)
make_subplots(q_list)
