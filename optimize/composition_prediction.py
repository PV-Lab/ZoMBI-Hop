import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import simpson
from scipy import signal
import pandas as pd
import math
import os
import statistics
from scipy.ndimage import median_filter
import traceback
from scipy.signal import find_peaks
import glob
import re
from scipy.signal import iirnotch, filtfilt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
# ipywidgets / IPython are only needed by the interactive notebook predictor
# (launch_physics_interactive_predictor). Guard the imports so this module can be
# imported as a plain physics library from headless scripts (run_mobo, evaluate,
# app, interactive_test) where those packages may be absent.
try:
    import ipywidgets as widgets
    from ipywidgets import interact
    from IPython.display import display, HTML
except Exception:  # pragma: no cover - notebook-only dependency
    widgets = None
    interact = None
    display = HTML = None
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import LeaveOneOut, cross_val_predict
from numpy.lib.stride_tricks import as_strided
from scipy.interpolate import interp1d
from scipy import integrate
import json

## System Parameters ##
# Load the hardware/print config relative to THIS file (not the caller's cwd) so
# the physics model works regardless of where the importing script is run from.
_CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config.json")
with open(_CONFIG_PATH, "r") as file:
        data_config = json.load(file)
syringe_radius = data_config['syringe_radius']
lead =  data_config['lead'] # [mm]
micro_steps =  data_config['micro_steps']
total_num_drops =  data_config['total_num_drops'] # determined by Computer Vision
grad_steps= data_config['grad_steps'] 
max_print_time =  data_config['max_print_time']  # seconds
# Physics Deterministic Parameters 
# ─── Fluid & Pipe Parameters (tune these) ────────────────────────────────────
FLUID = {
    'rho': 1000,       # kg/m³       — water density
    'mu':  1e-3,       # Pa·s        — water dynamic viscosity
}
PIPE = {
    'D':       0.79375e-3,   # m           — pipe inner diameter (1 mm)
    'length':  1,    # m           — pipe length
    'C':       2e-14,  # m³/Pa       — compliance (increase for softer/longer pipes)
    'A':       np.pi * (0.79375e-3)**2 / 4,  # m² — cross-sectional area (auto from D)
}

# Shared Functions #
def norm_padd_flows(junction_volume,flowrates,time_seconds,start_time,first_comp,platej0=None,use_set_j0=False,verbose=True):
    # Normalize the flow rates 
    global init_jv
    flowrates[np.where(flowrates<0)] = 0 
    # lets trim it to exclude anything before the start of the gradient 
    # try:
    #     rows_tbd = np.where(time_seconds>=start_time)[0][0] 
    #     flowrates_trimmed = flowrates[rows_tbd:-1,:]
    #     time_trimmed = time_seconds[rows_tbd:-1]
    # except:
    flowrates_trimmed = flowrates
    time_trimmed = time_seconds
    
    norm_vec = np.sum(flowrates_trimmed,axis=1,keepdims=True) 
    if verbose:
        fig,ax=plt.subplots(1) 
        ax.plot(time_trimmed,norm_vec)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("FLow rate [ul/s]") 
        # fig.tight_layout()
        plt.title("Sum of all Flow Rates at Time t") 
        plt.show()
    norm_vec_safe = np.where(norm_vec == 0, np.nan, norm_vec)
    norm_flow_timewise = np.divide(flowrates_trimmed, norm_vec_safe)
    norm_flow_timewise = np.nan_to_num(norm_flow_timewise, nan=0.0)
    if verbose:
        fig,ax=plt.subplots(1) 
        ax.plot(time_trimmed,norm_flow_timewise)
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Compositional Fraction") 
        # fig.tight_layout()
        plt.title("Time Point Normalized Flow Rate") 
        plt.show()

    # Estimate the total_flow_rate
    wind = max(1, int(len(time_trimmed)/5))
    total_flow_rate = np.nanmean(norm_vec_safe[wind:wind*2])
    if verbose:
        print("Time_Points:",len(time_trimmed))
        print("Wind:",wind)
        print(norm_vec[wind:2*wind])
        print("Flow_rate:",total_flow_rate,"ul/s")

    # Pad the flow rate 
    time_step = max(time_seconds)/len(time_seconds)
    # Estimate the time it takes to clear the junction
    lib_pl={'Front':85,'Back':66}
    if not use_set_j0:
        time_to_clear = junction_volume/total_flow_rate
    else:
        time_to_clear = lib_pl[platej0]/total_flow_rate
    # print("Total Flow Rate:", total_flow_rate)
    # print("Clear:", time_to_clear)
    # make time-vectroa and compotitional padding
    time_points = np.arange(start_time-time_to_clear,start_time,time_step) 
    repeat_rows = np.tile(first_comp,(len(time_points),1))
    # Add start of gradient to flowrates
    flow_trimmed_expended = np.insert(norm_flow_timewise,0,repeat_rows,axis=0)
    time_trimmed_extended = np.insert(time_trimmed,0,time_points,axis=0) 
    if verbose:
        fig,ax=plt.subplots(1)
        for i,line in enumerate(flow_trimmed_expended.T):
            ax.plot(time_trimmed_extended,line,label=f"Module {i}") 
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Compositional Fraction")
        ax.legend()
        # This plot represents all of the dispensed fluid in the print with each time point
        # corresponding to the volume placed inside the mixing Volume 
        plt.title("Full Print Volume Visualization") 
        plt.show()
    return flow_trimmed_expended,time_trimmed_extended,total_flow_rate,flowrates_trimmed,time_trimmed

def set_volume_packets(droplets_made,flows,times,drop_time,verbose=True):
    # Now place out all droplets time points
    # Plot the Graph 
    if verbose:
        fig,ax=plt.subplots(1)
        for i,line in enumerate(flows.T):
            ax.plot(times,line,label=f"Module {i}") 
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Compositional Fraction")
        ax.legend()
    # This plot represents all of the dispensed fluid in the print with each time point
    # corresponding to the volume placed inside the mixing Volume 

    # Now Divide the plot into Grids 
    x_pts=[]
    y_pts=[]
    k=0
    first_time= times[0] 
    x_pts.append(first_time)
    y_pts.append(flows[0,:])
    # iterate over the droplets_made until you find the first droplet made 
    for m,drop in enumerate(droplets_made):
        k+=1 # count this droplet volume 
        if drop ==1: # if the droplet exists set it as the time limit for the droplet volume
            limit = first_time+(k*drop_time) 
            first_drop=m 
            break 
    # print("First k:",k, "at ",limit)
    for i,value in enumerate(times): 
        if value>=limit: # find the nearest time point to that droplet volume time window 
            x_pts.append(value)
            y_pts.append(flows[i,:])
            # push up your counter based on the next set of droplets 
            if first_drop+1<len(droplets_made):
                for g,drop in enumerate(droplets_made[first_drop+1:]):
                    k+=1 
                    if drop ==1: 
                        first_drop= first_drop+1+g
                        break 
            else: 
                k+=1
            # augment your limit
            limit = first_time+(k*drop_time)  
            # print("Next ks:",k, "at ",limit)

    # Plot those Mark points
    # ax.scatter(x_pts,y_pts,color='r
    y_pts=np.array(y_pts)
    if verbose:
        for i in range(y_pts.shape[1]):
            ax.scatter(x_pts, y_pts[:, i], label=f'Col {i}',color='r')
        plt.title(f"Individual Droplet Volume Visualization for {len(x_pts)} Drop Volumes") 
        plt.show()
    return y_pts,x_pts

def est_composition(times,flows,junction_volume,diffusion_factor,drop_time,flow_rate,first_comp,total_num_drops,x_pts,sequential=None,df_eds=None,verbose = False,plot_final=True):
    # Now Predict the Droplet Composiitons 
    # print(f"JV {junction_volume},FR {flow_rate},df {diffusion_factor} \n")

    # estimate the number of droplets included in the diffusion volume
    # blocks are the number of droplets inside the junction volume that get mixed together due to diffusion 
    # The higher the diffusion factor the more of the junction volume that is taken into account
    if flow_rate!=0:
        blocks = junction_volume*diffusion_factor/(drop_time*flow_rate) 
    else: 
        blocks=1
    # ie blocks = 6 
    # Round up or down to ensure blocks is an integer 
    if blocks-int(blocks)>=0.5:
        blocks=int(blocks)+1
    elif blocks-int(blocks)<0.5:
        blocks=int(blocks) 
    if blocks <1: 
        blocks = 1 # if the diffusion is very low, you only take one droplet volume into account to calculate the composition. 
    # print(blocks)
    # Now go through every droplet and estimate the composition
    # Define your Variables / Initital Conditions 
    prev_comp = first_comp 
    ind_lim = len(x_pts)-1  
    if total_num_drops>=len(x_pts): #The only case that should create a problem
        # clip total_num_droplets and output what we can....should always be one less than len(x_pts) ie for 24 droplets need 25 x_pts 
        total_num_drops = len(x_pts)-1
    comps = []
    for i,val in enumerate(x_pts): # for every droplet volume on your print trace 
        if verbose:
            print(i)
        if blocks>1 and i < total_num_drops: # if you have more than one droplet volume to take into account and still have more compositions to calculate
            # extract the index 
            index = blocks-1 
            # ie index = 4 
            prev_time = x_pts[i] # the time at the start of your droplet volume 
            # extract all time points in the compositional representation during the mixed volume interval 
            start_index = np.where(times==prev_time)[0][0]
            
            # Ensure that you have enoough droplet volumes remaining in the encoder data traces 
            if i+index+1 <= ind_lim: # if yes, get the data
                end_index = np.where(times==x_pts[i+index])[0][0]
                end_index_add = np.where(times==x_pts[i+index+1])[0][0]
            else: # if no, only consider the last remaining droplet volumes 
                #(We should keep an eye on this, to ensure the tail of data is long enough to capture the dilutions at the end)
                end_index = np.where(times==x_pts[-1])[0][0]
                end_index_add = np.where(times==x_pts[-1])[0][0]
            # Extract the time points during your diffusion volume 
            # print(end_index,start_index)
            time_arr =times[start_index:end_index] 
            # Padd the begining of the volume with the last composition ( what was in the mixing volume before)
            rows = np.tile(prev_comp,(len(time_arr),1))
            # Extract the flow_rates and time values during the last (most recent added) droplet volume 
            comp_arr_add =flows[end_index:end_index_add,:] 
            time_arr_add =times[end_index:end_index_add] 
            # Add the diluted volume from the previous diluted droplet volumes to the new droplet volume
            # print(rows.shape,comp_arr_add.shape)
            y_vals = np.concatenate((rows,comp_arr_add),axis=0)
            x_vals = np.concatenate((time_arr,time_arr_add),axis=0)
            if verbose: 
                fig,ax=plt.subplots(1)  
            volumes = []
            for j,line in enumerate(y_vals.T):
                if verbose: 
                    ax.plot(x_vals,line,label=f"Module {j}") 
                # integrate for the total volume 
                volume = np.trapezoid(line, x_vals)
                volumes.append(volume) 
            volumes_arr = np.array(volumes)
            total_volume = np.sum(volumes_arr) # Volume added by all ten modules during the diluted and new droplet volumes
            comp = volumes_arr/total_volume # extract composition by normalizing individual module volumes with the total volume 
            if verbose:
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Compositional Fraction")
                ax.legend()
            comp_strs = [f"{x:.2f}" for x in comp]
            line_len = 5  # number of values per line
            lines = [
                ", ".join(comp_strs[a:a + line_len])
                for a in range(0, len(comp_strs), line_len)
            ]
            wrapped_title = "\n".join(lines)
            if verbose: 
                plt.title(f"Drop {i} Estimated Composition:\n{wrapped_title}")
                plt.tight_layout()
                plt.show() 
            prev_comp = comp 
            comps.append(comp) 
        elif blocks == 1 and i < total_num_drops: 
            # extract just the composition of the droplet volumre 
                        # extract the index 
            index = 1 
            # ie index = 4 
            prev_time = x_pts[i] # the time at the start of your droplet volume 
            # extract all time points in the compositional representation during the mixed volume interval 
            start_index = np.where(times==prev_time)[0][0]
            
            # Ensure that you have enoough droplet volumes remaining in the encoder data traces 
            if i+1 <= ind_lim: # if yes, get the data
                end_index = start_index 
                end_index_add = np.where(times==x_pts[i+1])[0][0]
            else: 
                end_index = start_index 
                end_index_add = np.where(times==x_pts[-1])[0][0]

            # Extract the flow_rates and time values during the last (most recent added) droplet volume 
            comp_arr_add =flows[end_index:end_index_add,:] 
            time_arr_add =times[end_index:end_index_add] 
            # Add the diluted volume from the previous diluted droplet volumes to the new droplet volume
            # print(rows.shape,comp_arr_add.shape)
            y_vals = comp_arr_add
            x_vals = time_arr_add
            if verbose: 
                fig,ax=plt.subplots(1)  
            volumes = []
            for j,line in enumerate(y_vals.T):
                if verbose: 
                    ax.plot(x_vals,line,label=f"Module {j}") 
                # integrate for the total volume 
                volume = np.trapezoid(line, x_vals)
                volumes.append(volume) 
            volumes_arr = np.array(volumes)
            total_volume = np.sum(volumes_arr) # Volume added by all ten modules during the diluted and new droplet volumes
            comp = volumes_arr/total_volume # extract composition by normalizing individual module volumes with the total volume 
            if verbose:
                ax.set_xlabel("Time [s]")
                ax.set_ylabel("Compositional Fraction")
                ax.legend()
            comp_strs = [f"{x:.2f}" for x in comp]
            line_len = 5  # number of values per line
            lines = [
                ", ".join(comp_strs[a:a + line_len])
                for a in range(0, len(comp_strs), line_len)
            ]
            wrapped_title = "\n".join(lines)
            if verbose: 
                plt.title(f"Drop {i} Estimated Composition:\n{wrapped_title}")
                plt.tight_layout()
                plt.show() 
            prev_comp = comp 
            comps.append(comp) 

    
    # Plot the estimated composition over the real ones 
    if plot_final:
        fig,ax=plt.subplots(1) 
        x_labels = np.arange(0,total_num_drops,1) 
        # print(comps)
        comps=np.array(comps)
        for i,line in enumerate(comps.T): 
            # print(line)
            line = np.array(line) 
            line = line.reshape(x_labels.shape) 
            ax.scatter(x_labels,line,label=f'Predicted {i}')

            
        # ax1=ax.twiny()
        # plot the measured Br and I values 
        # add mod 2 and 9 together always 
        if df_eds: 
            measured_Br = np.array(df_eds['Measured Br (at%)'])
            measured_I = np.array(df_eds['Measured I (at%)'])
            if sequential:
                ax.scatter(df_eds["Sequential"],df_eds['Measured Br (at%)'],color='lightblue',label="Measured Br") 
                ax.scatter(df_eds['Sequential'],df_eds['Measured I (at%)'],color='lightgreen',label="Measured I") 
            else: 
                ax.scatter(df_eds["Droplet #"],df_eds['Measured Br (at%)'],color='lightblue',label="Measured Br") 
                ax.scatter(df_eds['Droplet #'],df_eds['Measured I (at%)'],color='lightgreen',label="Measured I") 
        # calculate error Between predicted vs measured values 
        # predicted-measured/measured
        name='Measured vs Predicted'
        ax.plot()
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.title(f"{name}")
        plt.show()  
        
        # name_poped = name[:-4] 
        # fig.savefig( f"./prediction_plots/{name_poped}.png") 
        # plt.close()
    else: 
        comps=np.array(comps)
    if df_eds:
        return comps,measured_Br,measured_I
    else: 
        return comps

def steps_to_flow(steps_s, units):
    r              = 5.15
    steps_per_rev  = 25600
    lead           = 2
    mm_per_s       = steps_s * lead / steps_per_rev
    ul_per_s       = mm_per_s * math.pi * r**2
    if units == 'ml_per_min':
        return ul_per_s * 60 / 1000
    return ul_per_s

def mod_grad(grad_steps,grad_interval,start_step,max_flow,start_comp,end_comp,padding=0,guess_padding=True,axes = None,align=None,verbose=False):
    # # Remove nans
    # end_comp=np.nan_to_num(end_comp,nan=0.0)
    # start_comp =np.nan_to_num(start_comp,nan=0.0)
    est_time = grad_steps*grad_interval
    if guess_padding and est_time<max_flow: 
        padding = round((max_print_time-est_time)/grad_interval)
    start_time=0
    comp_increment = (start_comp-end_comp)/grad_steps
    total_print_time = grad_steps*grad_interval
    x = np.linspace(start_time,total_print_time+padding)
    x_steps = np.arange(0,grad_steps,1)
    x_steps_padded = np.arange(0,grad_steps+padding,1)
    x_timing = x_steps_padded*grad_interval
    if align==None:
        x_timing_offset=x_timing
    else: 
        x_timing_offset=x_timing+align
    increment = max_flow*comp_increment
    counter = 0
    y_start = start_comp*max_flow
    y_steps=np.empty([grad_steps+padding,10])
    y_steps_comp=np.empty([grad_steps+padding,10])
    # print(x_steps)
    if verbose:
        print("y_start",y_start)
        print("increment",increment)
    j=0

    for i in x_steps: # x_steps goes from 0-79 
        counter+=1
        # add an increment for 
        # if you start at the last step 
        if counter==1 and start_step==grad_steps: #80=80
            y = y_start-increment*grad_steps
            y_comp = y/max_flow
            y_flow = steps_to_flow(y,'')
            y_steps[j]=y_flow
            y_steps_comp[j]=y_comp
            if verbose: 
                    print(i,counter,j)
                    print("y",y)
                    print("y_flow",y_flow)
                    print("Y_steps",y_steps[j])
            j+=1
        elif i>=start_step: #you are at the right steps 
            y = y_start-increment*counter
            y_comp = y/max_flow
            y_flow = steps_to_flow(y,'')
            y_steps[j]=y_flow
            y_steps_comp[j]=y_comp
            # print(y_flow.shape)
            # print(y_steps[j])
            if verbose: 
                    print(i,counter,j)
                    print("y",y)
                    print("y_flow",y_flow)
                    print("Y_steps",y_steps[j])
            j+=1
        else: 
            continue
    y_steps[j:,:]=y_flow
    y_steps_comp[j:,:]=y_comp
    if verbose: 
        print("Y_steps",y_steps[0:j,:])
    if axes == None and verbose == True:
        fig,ax=plt.subplots(1)
        for i in range(10):
            ax.plot(x_timing_offset,y_steps[:,i],label=f"MOD{i}")
        plt.ylabel("Flow Rate [ul/s]")
        plt.xlabel("Time [s]")
        plt.legend()
        plt.show()
    elif axes !=None and verbose==True:
        for i in range(10):
            axes.plot(x_timing_offset,y_steps[:,i],label=f"MOD{i}")
    return y_steps,x_timing_offset

# Method Specific # 

# Encoder # 
def take_derivative(syringe_radius,df_encoder,time_seconds,title,ylabel,input_title,input_ylabel,type,use_high_pass=False,verbose=True,calc_flow=True):
    derivatives=np.zeros((len(time_seconds),10))
    if calc_flow:
        flow_rates=np.zeros((len(time_seconds),10))
        num_figs = 3
    else:
        num_figs=2
    # Set the Encoder or Potentiometer Data
    if type =='encoder':
        idk = 1
        wind=10
    elif type == 'potentiometer':
        idk = 11 
        wind= 50
    else: 
        idk = 1
        wind = 1 
    # Set of Figures
    
    if verbose:
        fig,ax=plt.subplots(num_figs)
    
    for i in range(10):
        # Extract Encdoer data for this module
        y=df_encoder.iloc[:,i+idk].astype(float).values
        # REMOVE Sudden jumps in data using median filtering
        filt_y = median_filter(y,size=wind)
        # Perform low pass filtering     
        # Take, Store, and Plot Derivative
        derivative = np.gradient(filt_y,time_seconds,edge_order=2)
        if use_high_pass:
            derivative=high_pass(time_seconds,derivative)  
        derivative = np.where(derivative < 0, 0, derivative)
        derivatives[:,i]=derivative  
        # Plot the Encoder Data
        if verbose:
            ax[0].plot(time_seconds,filt_y)
            ax[1].plot(time_seconds,derivative)
        # Calculate the flow rate if needed 
        if calc_flow:
            flow_rate_arr= derivative*math.pi*syringe_radius**2 
            flow_rates[:,i]=flow_rate_arr
            if verbose:
                ax[2].plot(time_seconds,flow_rate_arr,label=f"MOD {i}")
    
    
    if calc_flow and verbose:
        ax[2].set_title("Calculated Flow Rate")
        ax[2].set_ylabel("Flow Rate [ul/s]")
    
    if verbose:
        ax[0].set_title(input_title)
        ax[0].set_ylabel(input_ylabel)
        ax[1].set_title(title)
        ax[1].set_ylabel(ylabel)
        fig.tight_layout()
        plt.legend()
        plt.show()
    
    if calc_flow:
        return derivatives,flow_rates
    else: 
        return derivatives

def high_pass(t, y, noise_band=(0.02, 0.35), quality_factor=1, verbose=False):
    """
    Only notches frequencies within the known noise band.
    noise_band: (min_freq, max_freq) in cycles/s to target
    """
    dt = np.mean(np.diff(t))
    fs = 1/dt

    yhat = np.fft.fft(y - np.mean(y))
    fcycles = np.fft.fftfreq(len(t), d=dt)
    
    pos_mask = fcycles > 0
    fcycles_pos = fcycles[pos_mask]
    power_pos = normalize(np.abs(yhat[pos_mask]))
    
    # Only look for peaks within the noise band
    band_mask = (fcycles_pos >= noise_band[0]) & (fcycles_pos <= noise_band[1])
    fcycles_band = fcycles_pos[band_mask]
    power_band = power_pos[band_mask]
    
    peaks, _ = find_peaks(power_band, prominence=0.1)
    
    if len(peaks) == 0:
        # print("No noise peaks found in band, returning unchanged")
        return y
    
    noise_freqs = fcycles_band[peaks]
    if verbose:
        print(f"Notching out: {[f'{f:.4f}' for f in noise_freqs]} cycles/s")
    
    # Check signal is long enough for filtfilt
    min_len = 3 * max(len(iirnotch(w0=f, Q=quality_factor, fs=fs)[0]) for f in noise_freqs)
    if len(y) < min_len:
        print("Signal too short for filtfilt, returning unchanged")
        return y
    
    y_filtered = y.copy()
    for noise_freq in noise_freqs:
        b, a = iirnotch(w0=noise_freq, Q=quality_factor, fs=fs)
        y_filtered = filtfilt(b, a, y_filtered)
    
    return y_filtered

def normalize(arr):
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))

def process_data(syringe_radius,df_encoder,time_lim=84,limit_range=False,use_high_pass=False,verbose=True):
    # Change the first COlumn from Pure Time to relative time 
    time_seconds=[]
    for row in df_encoder.iloc[:,0]:
        #print(row)
        row_str=str(row)
        parts=row_str.split(":")
        if len(parts)>=2:
            if len(parts)==3:
                hour,minute,second=parts 
                hour_seconds = float(hour)*60*60
                minute_seconds = float(minute)*60
                seconds = float(second)
                time_total = hour_seconds+minute_seconds+seconds
            elif len(parts)==2:
                minute,second=parts 
                minute_seconds = float(minute)*60
                seconds = float(second)
                time_total = minute_seconds+seconds
            time_seconds.append(time_total)
        else: 
            time_seconds.append(None)
    # Guard for time_seconds having None or Nan 
    time_seconds = np.array(time_seconds)
    valid_mask = ~pd.isnull(time_seconds)
    time_seconds = time_seconds[valid_mask]
    df_encoder = df_encoder[valid_mask.tolist()]
    # Start time_seconds at 0 
    time_seconds = time_seconds - time_seconds[0]
    title = "Calculated Velocity"
    ylabel = "Velocity [mm/s]"
    input_ylabel = "Displacement [mm]"
    # Extract Flow rates from Encoder data 
    input_title = "Encoder Displacement"
    if limit_range:
        mask = time_seconds < time_lim + 10
        time_seconds = time_seconds[mask]
        df_encoder = df_encoder[mask]
    derivatives_enc,flowrates_enc = take_derivative(syringe_radius,df_encoder,time_seconds,title,ylabel,input_title,input_ylabel,type='encoder',use_high_pass=use_high_pass,verbose=verbose) 

    # Extract the start point of the gradient
    df_rates = pd.DataFrame(flowrates_enc) 
    df_rates.insert(0,'time[s]',time_seconds)
    df_velocity = pd.DataFrame(derivatives_enc)
    df_velocity.insert(0,'time[s]',time_seconds)
    acceleration=take_derivative(syringe_radius, df_velocity,time_seconds,title="Acceleration",ylabel='Acceleration [mm/s^2]',
                                input_title = "Velocity",
                                input_ylabel = "Velocity [mm/s]",
                                type='neither',
                                calc_flow=False,
                                verbose=verbose) 
    # Get the index of the max value in the flattened array
    try:
        max_index_flat = np.argmax(acceleration[np.where(time_seconds<20)]) #Find the max in the first 20 seconds
    except Exception as e: 
        print(f"Tried to get acceleration but could not {e}. Set start time to first time point")
        start_time=time_seconds[0]
    # Lets not just get the max value but the first value that is much greater than everything else
    # Convert flat index to (row, col)
    row, col = np.unravel_index(max_index_flat, acceleration.shape)
    start_time=time_seconds[row]
    
    # start_time=time_seconds[0] +2 # over_rode the acceleration dependent start_time determination because the real data will start at the right time
    # add 2 seconds to get rid of the initial noise of starting the print

    if verbose:
        # print(f"Max value is {acceleration[row, col]} at row {row}, column {col}")
        print("Start_Time:",start_time)
    return derivatives_enc,flowrates_enc,start_time,time_seconds

# Physics # 
def friction_factor(Re):
    if Re < 2300:
        return 64 / Re          # laminar — Hagen-Poiseuille
    else:
        return 0.316 / Re**0.25 # turbulent — Blasius

def effective_params(Q_ul_s, fluid=FLUID, pipe=PIPE):
    """
    Compute zeta and omega_n from physical parameters at a given flow rate.
    Q_ul_s: flow rate in ul/s (converts internally to m³/s)
    """
    Q   = Q_ul_s * 1e-9          # ul/s → m³/s
    A   = pipe['A']
    u   = Q / A if Q > 1e-12 else 1e-12 # set a minimum so Q never goes to 0 

    Re  = fluid['rho'] * u * pipe['D'] / fluid['mu']
    Re  = max(Re, 1e-6)

    f   = friction_factor(Re)
    R   = f * (pipe['length'] / pipe['D']) * (fluid['rho'] * u / 2) / max(Q, 1e-12)
    L   = fluid['rho'] * pipe['length'] / A

    omega_n = 1.0 / np.sqrt(L * pipe['C'])
    zeta    = (R / 2) * np.sqrt(pipe['C'] / L)

    return zeta, omega_n

def clamped_effective_params(Q_ul_s, fluid=FLUID, pipe=PIPE):
    Q   = max(Q_ul_s, 0.01) * 1e-9   # clamp floor — prevents Re→0, zeta→inf
    A   = pipe['A']
    u   = Q / A

    Re  = fluid['rho'] * u * pipe['D'] / fluid['mu']
    Re  = max(Re, 1.0)

    f   = friction_factor(Re)
    R   = f * (pipe['length'] / pipe['D']) * (fluid['rho'] * u / 2) / Q
    L   = fluid['rho'] * pipe['length'] / A

    omega_n = 1.0 / np.sqrt(L * pipe['C'])
    zeta    = min((R / 2) * np.sqrt(pipe['C'] / L), 5.0)   # cap at 5

    return zeta, omega_n

def print_params(Q_ul_s):
    """Helper to inspect parameters at a given flow rate."""
    Q   = Q_ul_s * 1e-9
    A   = PIPE['A']
    u   = Q / A
    Re  = FLUID['rho'] * u * PIPE['D'] / FLUID['mu']
    z, w = effective_params(Q_ul_s)
    print(f"  Q={Q_ul_s:.2f} ul/s | u={u:.4f} m/s | Re={Re:.1f} | zeta={z:.3f} | omega_n={w:.3f}")

def slow_decay_ode_ramp_physics(X, t_ode, ramp_func, fluid=FLUID, pipe=PIPE):
    """
    Ramp ODE where zeta and omega_n are computed from physics at each timestep.
    """
    x, dotx = X
    u       = ramp_func(t_ode)
    u_ul_s  = max(abs(u), 1e-6)       # avoid zero flow — use current target as operating point
    zeta, omega_n = effective_params(u_ul_s, fluid, pipe)
    ddotx   = -2*zeta*omega_n*dotx - omega_n**2*x + omega_n**2*u
    return [dotx, ddotx]

def ode_ramp_physics(X, t_ode, ramp_func, fluid=FLUID, pipe=PIPE,
                     decay_boost=4.0):
    x, dotx = X
    u       = ramp_func(t_ode)
    q_now   = max(abs(x), 1e-6)
    zeta, omega_n = effective_params(q_now, fluid, pipe)
    
    error = u - x
    if error < 0:                  # commanded flow is DROPPING → decay phase
        omega_n *= decay_boost     # faster response on the way down
    
    ddotx = -2*zeta*omega_n*dotx - omega_n**2*x + omega_n**2*u
    return [dotx, ddotx]

def update_ramp_physics(X0_list, t_ramp, y_ramp,experiment, fluid=FLUID, pipe=PIPE, verbose=False):
    """
    Physics-based ramp response — zeta and omega_n computed from flow rate at each step.
    """
    solns = []

    if verbose:
        fig, ax = plt.subplots(1)
        print("\n── Physics parameters at min/max flow ──")
        print_params(y_ramp.max())
        print_params(max(y_ramp[y_ramp > 0].min(), 1e-6))

    for i, X0 in enumerate(X0_list):
        y         = y_ramp[:, i]
        ramp_func = interp1d(t_ramp, y, bounds_error=False, fill_value=(y[0], y[-1]))
        # sol       = integrate.odeint(ode_ramp_physics, X0, t_ramp, args=(ramp_func, fluid, pipe))
        sol = integrate.odeint(
            ode_ramp_physics, X0, t_ramp,
            args   = (ramp_func, fluid, pipe),
            hmax   = (t_ramp[-1] - t_ramp[0]) / len(t_ramp),  # limit step size
            mxstep = 5000,
            rtol   = 1e-4,
            atol   = 1e-6,
        )
        # ── catch NaN/inf before plotting ──────────────────────────────────────
        if not np.all(np.isfinite(sol[:, 0])):
            n_bad = np.sum(~np.isfinite(sol[:, 0]))
            print(f"  WARNING: module {i} has {n_bad} non-finite values — clamping")
            sol[:, 0] = np.nan_to_num(sol[:, 0], nan=0.0, posinf=0.0, neginf=0.0)
        if verbose:
            ax.plot(t_ramp, sol[:, 0], label=f"IC {i+1}: x0={X0[0]:.2f}, v0={X0[1]:.2f}")
        solns.append(sol[:, 0])
    if verbose:
        ax.plot(t_ramp, y_ramp, 'k--', linewidth=1.5, label="ramp input")
        plt.grid()
        plt.xlabel("Time, $t$")
        plt.ylabel("Flow Rate [ul/s]")
        plt.legend(fontsize=7)
        plt.title("Physics-based ramp response (friction model)")
        plt.show()
    # # debug — print zeta and omega_n along module 7's ramp
    # boundary = (t_ramp > 63) & (t_ramp < 65)
    # print("t:        ", t_ramp[boundary])
    # print("module 7: ", y_ramp[boundary, 6])
    # print("diff:     ", np.diff(y_ramp[boundary, 6]))

    return solns

def generate_signal(grad_steps, max_flow, start_comp, end_comp, start_step,
                    grad_interval, padding=0, align=None, axes=None, verbose=False, resolution=5000):

    y_coarse, t_coarse = mod_grad(
        grad_steps, grad_interval, start_step, max_flow, start_comp, end_comp,
        padding=padding, align=align, axes=axes, verbose=False
    )
    # ── interpolate signal portion to fine grid ───────────────────────────────
    t_fine = np.linspace(t_coarse[0], t_coarse[-1], resolution)
    y_fine = np.zeros((resolution, 10))
    for i in range(10):
        f = interp1d(t_coarse, y_coarse[:, i], bounds_error=False,
                     fill_value=(y_coarse[0, i], y_coarse[-1, i]))
        y_fine[:, i] = f(t_fine)

    # print(f"Fine output: {y_fine.shape}, {t_fine.shape}")  # (1080, 10), (1080,)

    if verbose:
        fig, ax = plt.subplots(1)
        for i in range(10):
            o=0
            ax.plot(t_coarse,  y_coarse[:, i],  '--', alpha=0.4, label=f"MOD{i} coarse")
            ax.plot(t_fine,  y_fine[:, i],         label=f"MOD{i} fine")
        plt.ylabel("Flow Rate [ul/s]")
        plt.xlabel("Time [s]")
        plt.legend(fontsize=7)
        plt.title("generate_signal — coarse vs fine")
        plt.show()

    return y_fine, t_fine

def my_model(flow_rate=None,interval=None,plate_mod=None,start_step=None,simple=True,with_physics=True):
    lib_params ={True: {'Front':85,'Back':66},False:{'Front':85,'Back':66}}
    if not simple:
        if plate_mod=='Front':
            jv=0.0357*flow_rate+49.5
        else:
            jv=54.1+0.0161*flow_rate
        if start_step<40:
            diff_fact = -0.696*interval+0.959
        elif 40<=start_step and start_step<80:
            diff_fact=0.5
        else:
            diff_fact=0.6
    else: 
        jv = lib_params[with_physics][plate_mod]
        # jv = 1
        diff_fact=0.6
    
    return jv,diff_fact

# Main Functions # 

def experiment_data(file_name): 
    comp_df = pd.read_csv("./Configure_files/compositions.csv",header=None)
    start_comp= np.array(comp_df.iloc[0,1:11].astype(float))
    end_comp= np.array(comp_df.iloc[0,11:].astype(float))
    lib_af={'1':'Front','2':'Back','4':'Front'}
    plate=lib_af[comp_df.iloc[0,0]]
    lib={'Back':1,'Front':0}
    plate_ind=lib[plate]
    df_enc= pd.read_csv(file_name, header=0)

    df_enc.drop(index=0, inplace=True)
    with open("./config.json", "r") as file:
        data = json.load(file)
    max_flow=data['max_total_speed']
    start_step=data['grad_start_step'][plate_ind]
    grad_interval_time=data['grad_interval_time'][plate_ind]/1000

    return (start_comp,end_comp,plate,df_enc,max_flow,start_step,grad_interval_time)

def encoder_composition_extraction(experiment,use_j0=True,use_dm=False,use_physics_mod=False):
    (start_comp,end_comp,plate,df_encoder,max_flow,start_step,grad_interval_time) = experiment
    
    my_jv,my_df = my_model(plate_mod=plate,simple=True,with_physics=True)
    junction_volume=my_jv
    diffusion_factor=my_df
    drop_time = 3.4381582075524673
    if not use_dm:
        droplets_made=[1]*total_num_drops
    try:
        derivatives,flowrates,start_time,time_seconds=process_data(syringe_radius,df_encoder,limit_range=True,use_high_pass=True,verbose=False)
        flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,flowrates,time_seconds,start_time,start_comp,verbose=False)
        # print("Use Jo is:",use_j0)
        if not use_j0:
            flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,flowrates,time_seconds,0,start_comp,platej0=None,use_set_j0=False,verbose=False)
        else:
            # print("Plate is:",plate)
            flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,flowrates,time_seconds,0,start_comp,platej0=plate,use_set_j0=True,verbose=False)

        y_pts,x_pts = set_volume_packets(droplets_made,flows,times,drop_time=drop_time,verbose=False)
        compositions = est_composition(times,flows,junction_volume,diffusion_factor,drop_time,flow_rate,start_comp,total_num_drops,x_pts,sequential=use_dm,verbose=False,plot_final=False)
        compositions = np.array(compositions)
        if compositions.ndim < 2 or compositions.shape[0] == 0 or len(compositions)<total_num_drops:
            print(f'{experiment} resulted in composition array of shape {compositions.shape}')
            # Add a flag for DiSCO to handle this case
            # continue
    except Exception as e:
        print(f"Trial failed on {experiment}: {e}")
        traceback.print_exc()

    return compositions

def deterministic_composition_extraction(experiment,use_j0=True,use_dm=False,use_physics_mod=False):
    (start_comp,end_comp,plate,df_encoder,max_flow,start_step,grad_interval_time) = experiment
    my_jv,my_df = my_model(plate_mod=plate,simple=True,with_physics=True)

    junction_volume=my_jv
    diffusion_factor=my_df
    drop_time = 3.4381582075524673

    # if counting<1:
    if not use_dm:
        droplets_made=[1]*total_num_drops
    try:
        # Make the predicted flow rate pattern
        comps,timing = mod_grad(grad_steps,grad_interval_time,start_step,max_flow,start_comp,end_comp,padding=0,axes=None,align=None)
        # Use it to predict the composition
        if not use_j0:
            flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,comps,timing,0,start_comp,verbose=False)
        else:
            flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,comps,timing,0,start_comp,platej0=plate,use_set_j0=True,verbose=False)

        y_pts,x_pts = set_volume_packets(droplets_made,flows,times,drop_time=drop_time,verbose=False)
        compositions = est_composition(times,flows,junction_volume,diffusion_factor,drop_time,flow_rate,start_comp,total_num_drops,x_pts,sequential=use_dm,verbose=False,plot_final=False)
        compositions = np.array(compositions)
        if compositions.ndim < 2 or compositions.shape[0] == 0 or len(compositions)<total_num_drops:
            print(f'{experiment} resulted in composition array of shape {compositions.shape}')
    except Exception as e:
        print(f"Deterministic prediciton Failed for {experiment}")
        traceback.print_exc()
    return compositions

def deterministic_physics_extraction(experiment,use_physics_mod=True,use_j0=True,use_sequential=False,use_dm=False):
    (start_comp,end_comp,plate,df_encoder,max_flow,start_step,grad_interval_time) = experiment
    # ─── Define Gradient Parameters ─────────────────────────────────────────────────────────────────────
    # junction_volume = fv(flow_rate, interval, lib[plate_name], start_step)
    # diffusion_factor = fd(flow_rate, interval, lib[plate_name], start_step)
    # # diffusion_factor = 0.2
    # me = fe(flow_rate, interval, lib[plate_name], start_step)
    # print(f"\nPredicted junction_volume:  {junction_volume:.2f}")
    # print(f"Predicted diffusion_factor: {diffusion_factor:.4f}")
    # print(f"Predicted max_error:        {me:.4f}")
    my_jv,my_df = my_model(plate_mod=plate,simple=True,with_physics=True)
    junction_volume=my_jv
    diffusion_factor=my_df
    drop_time = 3.4381582075524673
    if not use_dm:
        droplets_made=[1]*total_num_drops
    plt.close('all')
    # ── physics-based ramp response (new) ─────────────────────────────────────────
    y_gen, t_gen = generate_signal(
        grad_steps, max_flow, start_comp, end_comp, start_step, grad_interval_time,padding=0,verbose=False)
    if use_physics_mod:
        max_flow_uls = steps_to_flow(max_flow,'')
        # start_comp_x0 = start_comp - (start_step*(start_comp-end_comp)/grad_steps)
        start_comp_x0 = np.zeros_like(start_comp)
        xdot0    = np.expand_dims((y_gen[1] - y_gen[0]) / (t_gen[1] - t_gen[0]), axis=1)
        x0       = np.expand_dims(np.array(start_comp_x0)*max_flow_uls, axis=1)
        init_cond= np.concatenate((x0, xdot0), axis=1)
        physics_response = update_ramp_physics(init_cond, t_gen, y_gen,experiment)
        physics_array = np.stack(physics_response, axis=1)  # list of (1000,) → (1000, 10)
    # ── Predict Composition ─────────────────────────────────────────
    else: 
        physics_array = y_gen
    try:
        # comps,timing = mod_grad(grad_steps,grad_interval_time,start_step,max_flow,start_comp,end_comp,axes=None,align=None)
                # Use it to predict the composition
        if not use_j0:
            flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,physics_array,t_gen,0,start_comp,verbose=False)
        else:
            flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,physics_array,t_gen,0,start_comp,platej0=plate,use_set_j0=True,verbose=False)

        y_pts,x_pts = set_volume_packets(droplets_made,flows,times,drop_time=drop_time,verbose=False)
        compositions = est_composition(times,flows,junction_volume,diffusion_factor,drop_time,flow_rate,start_comp,total_num_drops,x_pts,verbose=False,plot_final=False)
        compositions = np.array(compositions)
        if compositions.ndim < 2 or compositions.shape[0] == 0 or len(compositions)<total_num_drops:
            print(f'{experiment} resulted in composition array of shape {compositions.shape}')
            # Add a flag for DiSCO to handle this case
            ############## You still ned to fix this bug where it can be too small ########
            # continue
    except Exception as e:
        print(f"Deterministic prediciton Failed for {experiment}")
        traceback.print_exc()
    return compositions

# ── Physics-based "actual composition" simulator (noise replacement) ──────────
# Drop-in replacement for the random ``add_composition_noise`` perturbation used
# by the synthetic ZoMBI-Hop runs. Instead of blurring a requested composition
# line with Gaussian noise, this pushes the requested start→end endpoints through
# the deterministic hardware physics (syringe ramp lag/overshoot + junction-volume
# diffusion mixing) and returns the compositions that would actually be printed.
#
# The physics acts on the 10 syringe modules and is agnostic to what chemistry
# each module carries, so a d-dimensional composition (d ≤ 10) is placed on the
# first d modules; the remaining modules stay at zero and are ignored on the way
# back out.

# Print parameters for the simulated hardware (mirrors optimize/debugging.py).
PHYSICS_PLATE              = "Front"
PHYSICS_MAX_FLOW           = 490
PHYSICS_START_STEP         = 0
PHYSICS_GRAD_INTERVAL_TIME = 500 / 1000   # seconds


def _resample_rows(arr: np.ndarray, n_target: int) -> np.ndarray:
    """Linearly resample ``arr`` (n, d) to (n_target, d) along the row axis."""
    n = arr.shape[0]
    if n == n_target:
        return arr
    src = np.linspace(0.0, 1.0, n)
    dst = np.linspace(0.0, 1.0, n_target)
    return np.column_stack([np.interp(dst, src, arr[:, j]) for j in range(arr.shape[1])])


def physics_simulate_line(
    x_left,
    x_right,
    *,
    num_points=None,
    plate=PHYSICS_PLATE,
    max_flow=PHYSICS_MAX_FLOW,
    start_step=PHYSICS_START_STEP,
    grad_interval_time=PHYSICS_GRAD_INTERVAL_TIME,
    device="cpu",
    dtype=None,
):
    """Simulate the compositions actually printed for a requested composition line.

    Parameters
    ----------
    x_left, x_right : (d,) array-like / torch tensor
        The requested line endpoints on the ``d``-simplex (d ≤ 10). These are the
        start and end compositions of the printed gradient.
    num_points : int, optional
        Number of returned droplet compositions. Defaults to the number the
        physics model produces (``total_num_drops``); if it differs the result is
        linearly resampled to ``num_points``.
    plate, max_flow, start_step, grad_interval_time :
        Hardware print parameters (defaults mirror optimize/debugging.py).
    device, dtype :
        If torch is available the result is returned as a torch tensor on this
        device/dtype; otherwise a numpy array is returned.

    Returns
    -------
    (num_points, d) tensor/array of simulated printed compositions, each row
    normalized to sum to 1.
    """
    def _to_np(v):
        arr = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
        return np.asarray(arr, dtype=float).ravel()

    left = _to_np(x_left)
    right = _to_np(x_right)
    d = left.shape[0]
    if d != right.shape[0]:
        raise ValueError(f"endpoint dims differ: {d} vs {right.shape[0]}")
    if d > 10:
        raise ValueError(f"physics model supports up to 10 modules, got d={d}")

    # Place the d-dim composition on the first d of the 10 syringe modules.
    start10 = np.zeros(10, dtype=float)
    end10 = np.zeros(10, dtype=float)
    start10[:d] = left
    end10[:d] = right

    experiment = [start10, end10, plate, None, max_flow, start_step, grad_interval_time]
    out = deterministic_physics_extraction(experiment)
    out = np.asarray(out, dtype=float)
    if out.ndim != 2 or out.shape[0] == 0:
        raise RuntimeError(f"physics extraction returned bad shape {out.shape}")

    comp = out[:, :d]
    if num_points is not None:
        comp = _resample_rows(comp, num_points)
    s = comp.sum(axis=1, keepdims=True)
    comp = comp / np.where(s <= 1e-12, 1.0, s)

    try:
        import torch as _torch
    except Exception:
        return comp
    return _torch.as_tensor(comp, device=device, dtype=(dtype or _torch.float64))


def error_prediction(encoder_compositions,determinisic_composition,physics_compostions,verbose=False):
    " This function takes all three composition predictions and outputs a true composition with assocaited error bounds"
    "Input sizes Nx10 , where N is usually total_num_droplets "
    # Set the physics_composition prediciton as the base 
    # Find the difference between all three predictions for each droplet  
    all_comps = np.stack((encoder_compositions,determinisic_composition,physics_compostions),axis=2)
    mins = np.min(all_comps,axis=2)
    maxs = np.max(all_comps,axis=2)
    if verbose:
        print(mins.shape)
        print(maxs.shape)

        plt.close('all')
        fig,ax=plt.subplots(1)
        xes=np.arange(0,physics_compostions.shape[0],1)
        for i in range(10):
            ax.scatter(xes,encoder_compositions[:,i],color="Orange")
            ax.scatter(xes,physics_compostions[:,i],color="blue")
            ax.scatter(xes,determinisic_composition[:,i],color="purple")
            ax.scatter(xes,mins[:,i],color="black")
            ax.scatter(xes,maxs[:,i],color="red")
        plt.show()
    return mins,maxs

def predict_composition(file_name):
    experiment = experiment_data(file_name)
    encoder_compositions = encoder_composition_extraction(experiment)
    determinisic_composition = deterministic_composition_extraction(experiment)
    physics_compostions=deterministic_physics_extraction(experiment)
    min_range,max_range = error_prediction(encoder_compositions,determinisic_composition,physics_compostions)
    return min_range,physics_compostions,max_range

def launch_physics_interactive_predictor(transporter=None,data=None,eds=None,start_comps_set=None,end_comps_set=None,overide_dm=False,use_physics=False,use_sequential=False,use_j0=False):
    
    display(HTML("""
    <style>
        .jp-OutputArea-output, .output_area, .widget-output {
            background: #0f1117 !important;
        }
        .widget-label { 
            color: #a0aec0 !important; 
            font-family: 'Courier New', monospace !important; 
            font-size: 12px !important; 
        }
        .widget-text input, .widget-floattext input, .widget-inttext input { 
            background: #1a1f2e !important; 
            color: #e2e8f0 !important; 
            border: 1px solid #2d3748 !important;
            border-radius: 4px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 12px !important;
            padding: 2px 6px !important;
            height: 24px !important;
            line-height: 24px !important;
        }
        .widget-floattext input:focus, .widget-inttext input:focus {
            border-color: #4fd1c5 !important;
            outline: none !important;
            box-shadow: 0 0 0 2px rgba(79, 209, 197, 0.2) !important;
        }
        .section-title {
            color: #4fd1c5;
            font-family: 'Courier New', monospace;
            font-size: 12px;
            letter-spacing: 3px;
            text-transform: uppercase;
            margin-bottom: 8px;
        }
        .widget-vbox, .widget-hbox {
            background: #0f1117 !important;
        }
        .widget-button { 
            color: #4fd1c5 !important; 
            border: 1px solid #4fd1c5 !important;
            border-radius: 4px !important;
            font-family: 'Courier New', monospace !important;
            font-size: 11px !important;
            padding: 2px 6px !important;
            height: 24px !important;
        }
        .widget-button:hover {
            background: #1a3a3a !important;
        }
        .widget-textarea textarea {
            font-family: 'Courier New', monospace !important;
            font-size: 16px !important;
            background: #1a1f2e !important;
            color: #a0aec0 !important;
            border: 1px solid #2d3748 !important;
            border-radius: 4px !important;
        }
    </style>
    """))

    def style_widget(w):
        w.style = {'description_width': '110px'}
        w.layout = widgets.Layout(width='260px', margin='1px', height='32px')
        return w

    def section(title, *widget_list):
        header = widgets.HTML(f'<div class="section-title">{title}</div>')
        return widgets.VBox(
            [header, widgets.VBox(list(widget_list))],
            layout=widgets.Layout(
                border='1px solid #2d3748',
                padding='7px 9px',
                margin='3px 0',
                background_color='#0f1117',
            )
        )

    plot_out  = widgets.Output()
    # print_out = widgets.Output(layout=widgets.Layout(
    #     border='1px solid #2d3748',
    #     padding='8px',
    #     margin='3px 0',
    #     min_height='40px',
    #     max_height='120px',
    #     overflow_y='auto',
    #     background_color='#0f1117',
    # ))
    print_out = widgets.Textarea(
    value='',
    layout=widgets.Layout(
        width='280px',
        height='150px',
        margin='3px 0',
    )
        )
    print_out.style = {'font_family': 'Courier New', 'font_size': '11px'}
    def run_interactive(junction_volume, drop_time, diffusion_factor,
                        grad_steps, total_num_drops, grad_interval,
                        start_step, max_flow):
        start_comps =start_comps_set
        end_comps =end_comps_set
        plate = transporter
        plot_out.clear_output(wait=True)
        # print_out.clear_output(wait=True)
        print_out.value = ''  # instead of print_out.clear_output()

        if eds is not None:
            (df_encoder, df_eds, start_comp_eds, droplets_made_eds, end_comp_eds,
             start_time_eds, max_flow_eds, start_step_eds, grad_interval_time_eds, plate_eds) = data[eds]
            droplets_made = droplets_made_eds
            start_comps = start_comp_eds
            end_comps = end_comp_eds
        else:
            droplets_made = [1] * total_num_drops

        try:
            grad_interval_time = grad_interval/1000

            y_gen, t_gen = generate_signal(
                grad_steps, max_flow, start_comps, end_comps, start_step, grad_interval_time,padding=300,verbose=True)
            max_flow_ulss = steps_to_flow(max_flow,'')
            # start_comp_x0 = start_comp_eds - (start_step*(start_comp_eds-end_comp_eds)/grad_steps)
            start_comp_x0 = np.zeros_like(start_comps)
            xdot0    = np.expand_dims((y_gen[1] - y_gen[0]) / (t_gen[1] - t_gen[0]), axis=1)
            x0       = np.expand_dims(np.array(start_comp_x0)*max_flow_ulss, axis=1)
            init_cond= np.concatenate((x0, xdot0), axis=1)
            if use_physics:
                physics_response = update_ramp_physics(init_cond, t_gen, y_gen,eds)
                physics_array = np.stack(physics_response, axis=1)  # list of (1000,) → (1000, 10)
            else:
                physics_array = y_gen
            if not use_j0:
                flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,physics_array,t_gen,0,start_comps,use_set_j0=False,verbose=False)
            elif eds==None:
                flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,physics_array,t_gen,0,start_comps,platej0=plate,use_set_j0=True,verbose=False)
            else: 
                flows,times,flow_rate,time_fft,flow_fft = norm_padd_flows(junction_volume,physics_array,t_gen,0,start_comps,platej0=plate_eds,use_set_j0=True,verbose=False)

            if overide_dm==True:
                droplets_made=[1]*total_num_drops
            y_pts,x_pts = set_volume_packets(droplets_made,flows,times,drop_time=drop_time,verbose=True)
            compositions = est_composition(times,flows,junction_volume,diffusion_factor,drop_time,flow_rate,start_comps,total_num_drops,x_pts,verbose=False,plot_final=True)
            print_out.value+=(f"JV {junction_volume}ul,FR {flow_rate}ul/s,IT {grad_interval_time}s,df {diffusion_factor} \n")

            compositions = np.array(compositions)

            if compositions.ndim < 2 or compositions.shape[0] == 0 or len(compositions) < total_num_drops:
                print_out.value+=f'Resulted in composition array of shape {compositions.shape}\n'

            colors = ['#4fd1c5', '#f6ad55', '#fc8181', '#68d391', '#76e4f7',
                      '#b794f4', '#fbb6ce', '#90cdf4', '#faf089', '#c6f6d5']
            if len(compositions) < total_num_drops:
                x_labels  = np.arange(0, len(compositions), 1)
            else:
                x_labels  = np.arange(0, total_num_drops, 1)
            comps_arr = np.array(compositions)

            with plot_out:
                plt.close('all')
                plt.style.use('dark_background')

                # ── Plot 1: Composition Prediction
                fig1, ax1 = plt.subplots(figsize=(10, 3.5))
                fig1.patch.set_facecolor('#0f1117')
                ax1.set_facecolor('#1a1f2e')

                for i, line in enumerate(comps_arr.T):
                    line = np.array(line).reshape(x_labels.shape)
                    if np.any(line > 0.01):
                        ax1.scatter(x_labels, line, label=f'Module {i}',
                                   color=colors[i % len(colors)], s=60, alpha=0.9,
                                   edgecolors='white', linewidths=0.3, zorder=5)
                        ax1.plot(x_labels, line, color=colors[i % len(colors)],
                                alpha=0.3, linewidth=1, zorder=4)

                if eds is not None:
                    if use_sequential:
                        measured_df  = df_eds.dropna(subset=['Sequential']).sort_values('Sequential')
                        drop_indices = measured_df['Sequential'].values.astype(int)
                    else: 
                        measured_df = df_eds
                        drop_indices = measured_df['Droplet #'].values.astype(int)
                    meas_Br = np.array(measured_df['Measured Br (at%)'], dtype=float)
                    meas_I  = np.array(measured_df['Measured I (at%)'],  dtype=float)
                    ax1.scatter(drop_indices, meas_Br, color='yellow', marker='o',
                               s=60, label='EDS Br', zorder=6)
                    ax1.scatter(drop_indices, meas_I,  color='blue',   marker='o',
                               s=60, label='EDS I',  zorder=6)

                ax1.grid(True, color='#2d3748', linewidth=0.5, linestyle='--', alpha=0.7)
                ax1.set_axisbelow(True)
                ax1.set_xlabel("Droplet #", color='#a0aec0', fontsize=11, labelpad=8)
                ax1.set_ylabel("Compositional Fraction", color='#a0aec0', fontsize=11, labelpad=8)
                ax1.tick_params(colors='#718096', labelsize=9)
                for spine in ax1.spines.values():
                    spine.set_edgecolor('#2d3748')
                if eds is not None:
                    ax1.set_title(
                    f"{eds}   |   jv={junction_volume:.1f}  ·  dt={drop_time:.2f}  ·  df={diffusion_factor:.2f}",
                    color='#e2e8f0', fontsize=11, pad=10, fontfamily='monospace'
                )
                else:
                    ax1.set_title(
                        f"Composition Prediction   |   jv={junction_volume:.1f}  ·  dt={drop_time:.2f}  ·  df={diffusion_factor:.2f}",
                        color='#e2e8f0', fontsize=11, pad=10, fontfamily='monospace'
                    )

                ax1.legend(bbox_to_anchor=(1.02, 1), loc='upper left',
                           framealpha=0.1, edgecolor='#2d3748',
                           labelcolor='#a0aec0', fontsize=9)
                plt.tight_layout()
                plt.show()
                plt.close(fig1)

                # ── Plot 2: Predicted vs Measured + Errors
                if eds is not None:
                    # print(f"Saved compositions to {save_path}")
                    if use_sequential:
                        measured_df  = df_eds.dropna(subset=['Sequential']).sort_values('Sequential')
                        drop_indices = measured_df['Sequential'].values.astype(int)
                    else: 
                        measured_df = df_eds
                        drop_indices = measured_df['Droplet #'].values.astype(int)
                    measured_Br = np.array(measured_df['Measured Br (at%)'])
                    measured_I = np.array(measured_df['Measured I (at%)'])
                    n = min(len(compositions), len(df_eds),len(measured_df))

                    
                    compositions = np.array(compositions[:n])
                    compositions = np.nan_to_num(compositions, nan=0.0)  # compositions shouldn't have NaN

                    br_cols = [0,1,2,3,4]
                    i_cols  = [5,6,7,8,9]
                    comb_br = np.sum(compositions[:, br_cols], axis=1)
                    comb_i  = np.sum(compositions[:, i_cols],  axis=1)


                    # Only compute error where measurements are not NaN
                    br_mask = ~np.isnan(measured_Br)
                    i_mask  = ~np.isnan(measured_I)

                    br_error = comb_br[br_mask] - measured_Br[br_mask]
                    i_error  = comb_i[i_mask]  - measured_I[i_mask]
                    # print_out.value+=f'{comb_br}\n'
                    # print_out.value+=f'{measured_Br}\n'
                    # print_out.value+=f'{br_mask}\n'
                    
                    fig, ax = plt.subplots(2)
                    ax[0].scatter(drop_indices, comb_br, color='cornsilk', label='Combined Br')
                    ax[0].scatter(drop_indices, comb_i,  color='lightblue',   label='Combined I')
                    ax[0].scatter(drop_indices[br_mask], measured_Br[br_mask], color='yellow',   label='Measured Br')
                    ax[0].scatter(drop_indices[i_mask],  measured_I[i_mask],   color='blue', label='Measured I')
                    ax[1].scatter(drop_indices[br_mask], br_error, color='red',    label='Br Error')
                    ax[1].scatter(drop_indices[i_mask],  i_error,  color='purple', label='I Error')
                    ax[0].legend()
                    ax[1].legend()
                    plt.grid('on')
                    # plt.savefig(f'./Deterministic/{set_name}_{experiment}.png')
                    plt.show()
                    plt.close(fig)

                    cost =  np.sum(np.abs(br_error))
                    # with print_out:
                    #     print("\033[97mMax Error Br:\033[0m",f"\033[97m{np.max(br_error)}\033[0m")
                    #     print("\033[97mMax Error I:\033[0m",f"\033[97m{np.max(i_error)}\033[0m")
                    #     if cost == 0.0:
                    #         print(f"  Zero cost for {experiment}: comb_br={comb_br[br_mask]}, measured={measured_Br[br_mask]}")
                    print_out.value+=f"Max Error Br:{np.max(br_error)}\n"
                    print_out.value+=f"Max Error I:{np.max(i_error)}\n"
                    print_out.value+=f"Effective Volume:{junction_volume*diffusion_factor}\n"
                    print_out.value+=f"Total Cost:{cost}\n"

                plt.style.use('default')

        except Exception as e:
            with print_out:
                print(f"Failed: {e}")
                traceback.print_exc()

    # Widgets
    my_jv,my_df = my_model(plate_mod=transporter,simple=True,with_physics=True)
    drop_time = 3.44
    w_jv = style_widget(widgets.FloatText(value=my_jv, description='Junction Vol:', step=1.0))
    w_dt = style_widget(widgets.FloatText(value=drop_time,  description='Drop Time:',    step=0.1))
    w_df = style_widget(widgets.FloatText(value=my_df,  description='Diffusion Fac:',step=0.01))
    w_gs = style_widget(widgets.IntText(  value=80,    description='Grad Steps:'))
    w_nd = style_widget(widgets.IntText(  value=24,    description='Num Drops:'))
    w_gi = style_widget(widgets.IntText(  value=800,   description='Grad Interval:'))
    w_ss = style_widget(widgets.IntText(  value=0,    description='Start Step:'))
    w_mf = style_widget(widgets.IntText(  value=1050,   description='Max Flow:'))
    
    defaults = {'jv': my_jv, 'dt': drop_time, 'df': my_df}

    def reset_defaults(b):
        w_jv.value = defaults['jv']
        w_dt.value = defaults['dt']
        w_df.value = defaults['df']

    reset_btn = widgets.Button(
        description='↺  Reset',
        layout=widgets.Layout(width='100px', height='24px', margin='1px'),
        style=dict(button_color='#1a1f2e', font_family='Courier New')
    )
    reset_btn.style.font_size = '11px'
    reset_btn.on_click(reset_defaults)

    # Left panel: controls + output
    left_panel = widgets.VBox([
        widgets.VBox([
            widgets.HTML('<div class="section-title">── Composition Model</div>'),
            widgets.VBox([w_jv, w_dt, w_df]),
            reset_btn,
        ], layout=widgets.Layout(
            border='1px solid #2d3748',
            padding='7px 9px',
            margin='3px 0',
            background_color='#0f1117',
        )),
        section('── Gradient Parameters', w_gs, w_nd, w_gi),
        section('── Motion Parameters', w_ss, w_mf),
        widgets.HTML('<div class="section-title" style="margin-top:8px;">── Output</div>'),
        print_out,
    ], layout=widgets.Layout(
        width='300px',
        min_width='300px',
        padding='4px',
        background_color='#0f1117',
    ))

    # Right panel: plots stacked vertically
    right_panel = widgets.VBox([
        plot_out
    ], layout=widgets.Layout(
        flex='1',
        padding='4px',
        background_color='#0f1117',
    ))

    # Main layout
    main_ui = widgets.HBox([
        left_panel,
        right_panel,
    ], layout=widgets.Layout(
        background_color='#0f1117',
        align_items='flex-start',
    ))

    out = widgets.interactive_output(run_interactive, {
        'junction_volume':  w_jv,
        'drop_time':        w_dt,
        'diffusion_factor': w_df,
        'grad_steps':       w_gs,
        'total_num_drops':  w_nd,
        'grad_interval':    w_gi,
        'start_step':       w_ss,
        'max_flow':         w_mf,
    })

    display(main_ui, out)

if __name__=="__main__":
    file_name='./AF_data_repo/encoder_test.csv'
    min_error,max_error,predicted_comp = predict_composition(file_name)
    print(predicted_comp.shape)

    plump=np.zeros((predicted_comp.shape[0],1))
    print(predicted_comp.shape[0])
    print(plump.shape)

    sql_pred = np.hstack((plump, predicted_comp))
    fig,ax=plt.subplots(1)
    for i in range(11):
        ax.plot(sql_pred[:,i])
    plt.show()

    print(sql_pred.shape)
    sql_pred.tolist()
    print(sql_pred)
    starting =  np.array([0.5,0.5,0,0,0,0,0,0,0,0])
    ending =    np.array([0,0,0,0,0,0,0,0.25,0.25,0.5])
    transporter='Back'
    experiment= experiment_data(file_name)
    compositions = deterministic_physics_extraction(experiment,use_physics_mod=True,use_j0=True,use_sequential=False,use_dm=False)
    print(compositions)
    # transporter='Front'
    # launch_physics_interactive_predictor(start_comps_set=starting,end_comps_set=ending,transporter=transporter,use_j0=True,use_physics=True)
