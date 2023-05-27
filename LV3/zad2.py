import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data_C02_emission.csv')

def zad2a():
    data['CO2 Emissions (g/km)'].plot(kind='hist', bins=20)
    plt.xlabel('CO2 Emission (g/km)')
    plt.show()

def zad2b():
    data['Make']=pd.Categorical(data['Make'])
    data['Vehicle Class']=pd.Categorical(data['Vehicle Class'])
    data['Transmission']=pd.Categorical(data['Transmission'])
    data['Fuel Type']=pd.Categorical(data['Fuel Type'])
    data.plot.scatter(x='Fuel Consumption City (L/100km)', y='CO2 Emissions (g/km)', c='Fuel Type', cmap = 'viridis', s=20)
    plt.show()

def zad2c():
    data.groupby('Fuel Type').boxplot(column='Fuel Consumption Hwy (L/100km)')
    plt.show()

def zad2d():
    data.groupby('Fuel Type').agg({'Make':'count'}).rename(columns={'Make':'Car number'}).plot(kind="bar")
    plt.show()

def zad2e():
    data.groupby('Fuel Type').agg({'CO2 Emissions (g/km)':'mean'}).plot(kind='bar')
    plt.show()

zad2d()