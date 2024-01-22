#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[76]:


import sqlite3
import pandas as pd
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import linear_model
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score
import seaborn as sns
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error
class Moons:
    def __init__(self,connection ='data/JUPITER.db'):
        self.connection = connection
        self.connect = sqlite3.connect(self.connection)
        
    def dataFrame(self, query='SELECT * FROM moons'):
        self.data = pd.read_sql_query(query, self.connect)
        
    def show(self):
        print(self.data)
        
    def choose_moon(self):
        x = input('enter name of the moon you will like to view? ')
        return self.data[self.data['moon'] == x]
    def summary(self):
            return print(self.data.describe())
        
    def mass_raw(self):
        sns.stripplot(data=self.data, x="mass_kg", hue ='group' ,jitter=False, s=25, marker="D", linewidth=1)
        plt.title('Mass of moons raw data')
        plt.show()
        
    def mass_predicted(self):
        self.data['mag'] = self.data['mag'].fillna(self.data['mag'].mean())
        test = self.data.dropna(inplace=False).copy()
        train = self.data[self.data['mass_kg'].isnull()].copy()
        X = np.asarray(test.drop(['mass_kg', 'moon','group'], axis=1))
        Y = np.asarray(test['mass_kg'])
        x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=40)
        lr = LinearRegression()
        lr.fit(x_train, y_train)
        x_test= np.asarray(train.drop(['mass_kg', 'moon','group'], axis=1))
        mass_pred = lr.predict(x_test)
        train.loc[pd.isnull(train['mass_kg']), 'mass_kg'] = mass_pred
        mass_data = [test, train]
        predicted= pd.concat(mass_data)
        sns.stripplot(data=predicted, x="mass_kg", jitter=False, hue = 'group', s=25, marker="D", linewidth=1, alpha=0.8)
        plt.title('mass of moon /kg')
        plt.show()
        sns.scatterplot(data=predicted, x="mass_kg", y="distance_km", hue="group", marker= "+")
        plt.title('mass of moon /kg')
        plt.show()
        
    
        
    def heatmap(self):
        correlation = self.data.corr()
        sns.heatmap(correlation, annot = True, linewidth = 0.4, cmap = 'summer')
        plt.title('heatmap')
        plt.show()

    def plots_mag_vs_distance(self):
        self.data['mag'] = self.data['mag'].fillna(self.data['mag'].mean())
        variable_used = [ 'mag', 'distance_km']
        plot = sns.PairGrid(self.data, hue = 'group', vars=variable_used)
        plot.map_diag(sns.histplot, color = 'darkblue', alpha = 0.7, edgecolor = 'none', hue = None)
        plot.map_offdiag(sns.scatterplot, marker = '+')
        plot.add_legend()
        
    def plots_days_vs_ecc(self):
        variable_used = [ 'period_days', 'ecc']
        plot = sns.PairGrid(self.data, hue = 'group', vars=variable_used)
        plot.map_diag(sns.histplot, color= 'blueviolet' , hue = None ,edgecolor = 'none')
        plot.map_offdiag(sns.scatterplot, marker = 'd')
        plot.add_legend()
    
    def plots_deg_vs_ecc(self):
        variable_used = [ 'inclination_deg', 'ecc']
        plot = sns.PairGrid(self.data, hue = 'group', vars=variable_used)
        plot.map_diag(sns.histplot, color = 'peru', edgecolor = 'none', hue = None )
        plot.map_offdiag(sns.scatterplot, marker = 'v')
        plot.add_legend() 
    
    def plots_radius_vs_distance(self):
        self.data=self.data[self.data['radius_km'] <= 40]
        variable_used = [ 'distance_km', 'radius_km']
        plot = sns.PairGrid(self.data, hue = 'group',diag_sharey=False, vars=variable_used)
        plot.map_offdiag(sns.scatterplot, marker = 'v')
        plot.map_diag(sns.histplot, color ='rebeccapurple', alpha=0.7, hue = None, edgecolor = 'none')
        plot.add_legend()
    
        
    def circumference(self):
        moon = input('type name of moon')
        moon2 = input('type name of different moon')
        r= int(self.data.loc[self.data['moon'] == moon]['radius_km'])
        r2= int(self.data.loc[self.data['moon'] == moon2]['radius_km'])
        
        a= np.linspace( 0 , 2*np.pi , 180 ) 
        x = r * np.cos( a ) 
        y = r * np.sin( a ) 
        x1 = r2* np.cos( a ) 
        y1 = r2 * np.sin( a ) 

        figure, axes = plt.subplots( 1 ) 
 
        axes.plot( x, y, label = moon ) 
        axes.plot( x1, y1, label = moon2 ) 
    
        plt.title( 'Comparing circumferences of each Moon / km' ) 
        plt.legend()
        return plt.show() 
          
    def jupiter_mass(self):
        self.data['T_squared']=(self.data['period_days']*86400)**2
        self.data['A_cubed']=((self.data['distance_km'])*1000)**3
        a_cube = self.data['A_cubed']
        t_square = self.data['T_squared']
        X_train, X_test, y_train, y_test = train_test_split(a_cube, t_square)
        X_train = X_train.values.reshape([-1, 1])
        X_test = X_test.values.reshape([-1, 1])
        model = linear_model.LinearRegression(fit_intercept=True)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        G =(6.67)*(10**(-11))
        pi = np.pi
        constant = ((4 )*(pi**2) )
        Jupiter_mass = constant/(model.coef_[0]*G)
        print(f"unweighted model r2_score: {r2_score(y_test,y_pred)}")
        print(f"mass of Jupiter in kg:{Jupiter_mass}")
        


    
jupiter_moon_info = Moons()
jupiter_moon_info.dataFrame()


# In[77]:


jupiter_moon_info.show()


# the code above shows the dataset of Jupiter moons. This dataset provides the names, inclincation, degree, mass, ecc, group, radius and 
# distance from jupiter, plus the group

# In[78]:


jupiter_moon_info.choose_moon()


# the code above allows you to select the moon via input function. This shows the characteristics of chosen moon. an input
# function was selected to choose within this section of the class as it is simple to run.

# In[79]:


jupiter_moon_info.summary()


# The code above shows the general statistic summary of the ecc, periods, distance from Jupiter, radius,
# magnitude mass in kg and inclinication degree

# In[80]:


jupiter_moon_info.circumference()


# The code above allows the you to choose two moons and compare the circumference. this gives a visual representation
# of the size difference

# In[81]:


jupiter_moon_info.mass_predicted()


# Using linear regression, the mass of each moon was predicted and a scatter boxplot was produce and a graph plotting distance against 
# mass kg. the moons in the same group tend to cluster within the same mass region. Himalia however is the most spread in the range
# Kdeplot can be used to show more trends too. Each group almost form a line away from Jupiter, the closer the group is to jupiter
# the greater the range in mass

# In[ ]:





# In[82]:


jupiter_moon_info.mass_raw()


# this scatter boxplot shows the raw mass values for a few of jupiter's moons the range is similiar to the predicted mass range.
# the data from the other columns were used for the linear regression, as the magnitude column had missing values it was filled
# using the mean. This meant that some predicted masses maybe inaccurate. Other fillna methods could be suggested as well.

# In[83]:


jupiter_moon_info.heatmap()


# the code above shows a heatmap of the correlation between two factors. the mass correlation values are from the raw data. 
# a possible suggestion is that the correlation can be compared using the predicted mass. heatmap can be adjusted to use different
# themes for cmap. this method was chosen as it is a great way to visualise data

# In[84]:


jupiter_moon_info.jupiter_mass()


# Linear regression was again used to predict the mass of Jupiter. The values are similiar to the actual mass. the formula T^2
# =(4π^2/GM)a^3 was used to calculate the mass the actual mass of jupiter is 1.8987×10^27. the predicted values is quite similar at 
# around 1.926×10^27 kg the r2 value is close to 1 so the prediction is reliable

# In[85]:


jupiter_moon_info.plots_mag_vs_distance()


# The plots shows the relationships between groups, magnitude and distance. Distance and magnitude has a postive relationship
# both histograms are skewed left. Pasiphae has the greateast magnitude and distance and groups tend to cluster in the same area
# on the plot. The code is adjustable to edit to use different plots like kdeplot or lineplot. Scatterplots and Histograms were used
# as it is easy to interprated

# In[86]:


jupiter_moon_info.plots_deg_vs_ecc()


# the plot table code comparing the relationship between ecc, group and inclination degree is shown above. 
# .Ecc had a bell shape distribution and Inclination degree was bimodal for histagram.Ecc v inclination showed a weak postive trend and Inclination v ecc showed no trend
# groups form clusters with pasiphae having the greatest data spread between ecc and inclination degree.

# In[87]:


jupiter_moon_info.plots_radius_vs_distance()


# any radius greater than 40 km was removed from the data to create the plots to visualise the trends better. the scatter plots show
# no correlation between distance and Radius. The radius histograph was skewed right but the distance histagraph was skewed left.
# Groups data is rather spread out, with only Carme Ananke and Pasiphae forming distinct clusters at the bottom right of 
# radius vs distance graph

# In[88]:


jupiter_moon_info.plots_days_vs_ecc()


# the period days histogram was skewed left and the ecc histogram was bell-shaped. ecc vs period days show a postive correlation between
# . Moons form clusters based on their groups,

# In[ ]:





# In[ ]:




