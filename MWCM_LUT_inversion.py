# -*- coding: utf-8 -*-
"""
Created on Thu Jun 11 19:36:25 2020

@author: Dipankar
"""

###Importing library
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
#from scipy.optimize import curve_fit
import pandas as pd
from sklearn.metrics import mean_absolute_error


##################################################################################################
##################################################################################################
##################################################################################################
#Load CAL data
#With headers
riceallhead = pd.read_excel('DSR_Rice_SAR_CAL_combine.xlsx',na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],skiprows=[0],header=None);

#Pandas dataframe to matrix conversion
cald1m=riceallhead.values


##################################################################################################
######
##PAI values
x1=np.float64(cald1m[:,11])#Col8==PAI
RH0=np.float64(cald1m[:,13])#Col16==HH; 17==HV; 18==VH; 19==VV
RV0=np.float64(cald1m[:,14])
##################################################################################################
##M-chi powers
Psm0=np.float64(cald1m[:,20])
Pdm0=np.float64(cald1m[:,21])
Pvm0=np.float64(cald1m[:,22])
##################################################################################################
##IS-Omega powers
Psi0=np.float64(cald1m[:,17])#Col6==Ps
Pdi0=np.float64(cald1m[:,16])#Col9==Pd
Pvi0=np.float64(cald1m[:,18])#Col5==Pv


##################################################################################################
##################################################################################################
##################################################################################################

# Read VAL data
riceallhead2 = pd.read_excel('DSR_Rice_SAR_VAL_combine.xlsx',na_values = ['no info', '.','None','#VALUE!', '#DIV/0!'],skiprows=[0],header=None);
cald1mv=riceallhead2.values
## RH-RV VAL data--------------------------------------------------------------
RHv=np.float64(cald1mv[:,13])#Col16==HH; 17==HV; 18==VH; 19==VV
RVv=np.float64(cald1mv[:,14])
## m-Chi VAL data--------------------------------------------------------------
Psmv=np.float64(cald1mv[:,20])
Pdmv=np.float64(cald1mv[:,21])
Pvmv=np.float64(cald1mv[:,22])
## iS_Omega VAL data--------------------------------------------------------------
Psiv=np.float64(cald1mv[:,17])#Col6==Ps
Pdiv=np.float64(cald1mv[:,16])#Col9==Pd
Pviv=np.float64(cald1mv[:,18])#Col5==Pv
## PAI VAL data--------------------------------------------------------------
x1v=np.float64(cald1mv[:,11])#Col8==PAI

valY=x1v
valX=np.column_stack((Psiv,Pdiv,Pviv))


##################################################################################################
##################################################################################################
##################################################################################################
## LUT search based inversion MWCM-iS-Omega

numrows = len(valX) 
laiout = np.zeros(numrows)

for index, row in riceallhead2.iterrows():
    #m = 10000
    min = None
    for index1, row1 in riceallhead.iterrows():
        
        #RMSE
        rmse=np.sqrt((((row[20]-row1[20])**2)+((row[21]-row1[21])**2)+
                   ((row[22]-row1[22])**2))/3)
#        
        #L1 estimate
#        rmse = (abs(row[4]-row1[0]) + abs(row[3]-row1[1]))
        
        #Bhattacharya distance
#        rmse = (-np.log(1 + (row[4]*row1[0])**0.5 - 0.5*(row[4] + row1[0]) +
#                (row[3]*row1[1])**0.5 - 0.5*(row[3] + row1[1])))
      
        if laiout[index] == 0 or rmse < min:
            min = rmse
            laiout[index] = row1[11]
            #smout[index] = row1[4]
        
    
y_out = laiout


##PAI estimation and error-------------------------------------------------------------
def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())
rmselai = rmse(np.array(x1v), np.array(y_out))
#Correlation coefficient 
corrr_value=np.corrcoef(np.array(x1v), np.array(y_out))
rrlai= corrr_value[0,1]
maelai=mean_absolute_error(x1v,y_out)
#Plotting
fig, ax = plt.subplots(figsize=(5, 5))    
plt.plot(x1v,y_out, 'go')
plt.xlim([0, 8])
plt.ylim([0, 8])
plt.xlabel("Observed PAI ($m^{2}~m^{-2}$)")
plt.ylabel("Estimated PAI ($m^{2}~m^{-2}$)")
#plt.title("PAI plot")
plt.plot([0, 8], [0, 8], 'k:')
plt.annotate('r = %.2f'%rrlai, xy=(0.5, 7.4))#round off upto 3decimals
plt.annotate('RMSE = %.3f'%rmselai, xy=(0.5, 6.8))
plt.annotate('MAE = %.3f'%maelai, xy=(0.5, 6.2))
plt.xticks(np.arange(0, 8+1, 2.0))
matplotlib.rcParams.update({'font.size': 24})
plt.savefig('MWCMiSOmega_PAI.png',bbox_inches="tight",dpi=100)
plt.show()
plt.close()








