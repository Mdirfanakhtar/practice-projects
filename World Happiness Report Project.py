#!/usr/bin/env python
# coding: utf-8

# # 1. World Happiness Report Project

# # For Analyzing Happiness around the Globe.

# In[2]:


get_ipython().system('pip install bubbly')


# # All library which is use

# In[22]:


# for some basic operations
import numpy as np 
import pandas as pd

# for visualizations
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

# for interactive visualizations
import plotly.offline as py
from plotly.offline import init_notebook_mode, iplot
import plotly.graph_objs as go
init_notebook_mode(connected = True)
from bubbly.bubbly import bubbleplot
import plotly.figure_factory as ff

import warnings
warnings.filterwarnings("ignore")


# In[9]:


import pandas as pd
data= pd.read_csv("happiness_score_dataset.csv")
data


# In[10]:


data.head()


# # Correlations Between the Data

# In[11]:


plt.rcParams['figure.figsize'] = (20, 15)
sns.heatmap(data.corr(), cmap = 'copper', annot = True)

plt.show()


# In the above Heat Map we can see that Happiness Score is very highly correlated with Economy, Health, and Family Satisfaction and somewhat related with Freedom also but has very low relation with Trust in Government in average case.

# # Correlations for Western Europe

# In[12]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data.loc[lambda data: data['Region'] == 'Western Europe']
sns.heatmap(d.corr(), cmap = 'Wistia', annot = True)

plt.show()


# The Heat Map particularly for Europe has one more thing to add apart from Family Satisfaction, Freedom, Economy, Generosity, It is also highly correlated with Trust in Government.
# 
# The European Region is the Happiest Region so far.

# # Correlations for Eastern Asia

# In[13]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data.loc[lambda data: data['Region'] == 'Eastern Asia']
sns.heatmap(d.corr(), cmap = 'Greys', annot = True)

plt.show()


# Here, The situation gets worsened as the Correlation is negative for many important factors such as Economy, Health, Trust in Government which makes the situation very critical. It has Positive correlations only with Freedom, Generosity and Famlily Satisfaction.

# # North America

# In[14]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data.loc[lambda data: data['Region'] == 'North America']
sns.heatmap(d.corr(), cmap = 'pink', annot = True)

plt.show()


# Everything is highly correlated to the Happiness in America. Amongst so many countries of the world. Being a very large country also America is still able to keep their people happy. America stands at position number 10 amongst the Happiness Rankings for the World.

# # Middle East and Northern Africa

# In[15]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data.loc[lambda data: data['Region'] == 'Middle East and Northern Africa']

sns.heatmap(d.corr(), cmap = 'rainbow', annot = True)

plt.show()


# The correlations are quite goood with almost all the important factors being highly correlated with Happiness. Family Satisfaction is the most important factor as it is the most important factor for happiness n this region.

# # Sub-Saharan Africa

# In[16]:


plt.rcParams['figure.figsize'] = (20, 15)
d = data.loc[lambda data: data['Region'] == 'Sub-Saharan Africa']
sns.heatmap(d.corr(), cmap = 'Blues', annot = True)

plt.show()


# The Situations are very bad for Sub-Saharan Region as it is the unhappiest region in the world. The correlations with Happiness Score are very low for features such as Generosity, Family Satisfaction, Freedom etc. Almost all of the features are having less than 0.5 correlation which is very bad.

# # Bubble Charts

# In[ ]:


figure = bubbleplot(dataset = data, x_column = 'Happiness Score', y_column = 'Generosity', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Generosity", title = 'Happiness vs Generosity vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# In[ ]:


figure = bubbleplot(dataset = data, x_column = 'Happiness Score', y_column = 'Trust (Government Corruption)', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Trust", title = 'Happiness vs Trust vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# In[ ]:


figure = bubbleplot(dataset = data, x_column = 'Happiness Score', y_column = 'Health (Life Expectancy)', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Health", title = 'Happiness vs Health vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# In[ ]:


figure = bubbleplot(dataset = data, x_column = 'Happiness Score', y_column = 'Family', 
    bubble_column = 'Country', size_column = 'Economy (GDP per Capita)', color_column = 'Region', 
    x_title = "Happiness Score", y_title = "Family", title = 'Happiness vs Family vs Economy',
    x_logscale = False, scale_bubble = 1, height = 650)

py.iplot(figure, config={'scrollzoom': True})


# Bubble plot to depict the relation between the Happiness Scores vs Family Satisfaction where size of the bubbles is represented by the Economy and the color of the bubbles is represented by the Different Regions of the World.
# 
# It is Quite Visible that as the Family Satisfaction ratings increases the Happiness Score increases. So, We can say that they have a direct relationship between them.
# 
# Also, European Countries and Austrelia are the Happiest Regions. After America.
# 
# There is not even a single country in American Region with low Happiness Index.
# 
# Asian and African countries suffer with some serious issues, that is why none of the Asian orr African Country stands at a good position in terms of Happiness Index.
# 
# Some Countries in Middle East are Happy while some are Unhappy.

# In[ ]:


data = (
  {"label": "Happiness", "sublabel":"score",
   "range": [5, 6, 8], "performance": [5.5, 6.5], "point": [7]},
  {"label": "Economy", "sublabel": "score", "range": [0, 1, 2],
   "performance": [1, 1.5], "sublabel":"score","point": [1.5]},
  {"label": "Family","sublabel":"score", "range": [0, 1, 2],
   "performance": [1, 1.5],"sublabel":"score", "point": [1.3]},
  {"label": "Freedom","sublabel":"score", "range": [0, 0.3, 0.6],
   "performance": [0.3, 0.4],"sublabel":"score", "point": [0.5]},
  {"label": "Trust", "sublabel":"score","range": [0, 0.2, 0.5],
   "performance": [0.3, 0.4], "point": [0.4]}
)

fig = ff.create_bullet(
    data, titles='label', subtitles='sublabel', markers='point',
    measures='performance', ranges='range', orientation='v',
)
py.iplot(fig, filename='bullet chart from dict')


# Bullet Chart to Represent the Range for some of the most Important Attributes given in the data. We have taken Happiness, Economy, Freedom, and Family. for analysis of their range.
# 
# If the values for the given attributes lie in the Dark Blue Region then it is in the critical region.
# 
# If the values for the given attributes lie in the light blue region then is is in good condition.
# 
# If the values for the given attributes lie above or near the diamond then is in the best state or condition.
# 
# White Regions are depicting the Maxima that could be achieved.

# In[ ]:


d = data['Region'].value_counts()

label_d = d.index
size_d = d.values


colors = ['aqua', 'gold', 'yellow', 'crimson', 'magenta']

trace = go.Pie(
         labels = label_d, values = size_d, marker = dict(colors = colors), name = 'DATA', hole = 0.3)

data = [trace]

layout1 = go.Layout(
           title = 'Regions')

fig = go.Figure(data = data, layout = layout1)
py.iplot(fig)


# The Above pie chart depicts the Number of Countries from each Region,
# 
# There are only two countries from North America(USA and Canada), and Austrelia(Austrelia, and New Zealand) Regions.
# 
# The highest number of countries are from Sub-Saharan and Central and Eastern Europe Regions with 40 and 29 countries respectively.

# In[ ]:


trace1 = [go.Choropleth(
               colorscale = 'Earth',
               locationmode = 'country names',
               locations = data['Country'],
               text = data['Country'], 
               z = data['Generosity'],
               )]

layout = dict(title = 'Generosity',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# Generosity is really low in Big Countries like Russia, China, and India but USA, Austrelia, and Canada has high ratings for generosity.
# 
# Africa and South America have very low Generosity Scores in general.

# # Top 10 Most Generous Countries

# In[ ]:


data[['Country', 'Generosity']].sort_values(by = 'Generosity', ascending = False).head(10)


# In[ ]:


trace1 = [go.Choropleth(
               colorscale = 'Cividis',
               locationmode = 'country names',
               locations = data['Country'],
               text = data['Country'], 
               z = data['Trust..Government.Corruption.'],
               )]

layout = dict(title = 'Trust in Governance',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# Looks like there is no trust in government all round the world, except Norway, sweden, and Finland.
# 
# Canada Saudi Arabia, Germany, United Kingdom, Somalia, Ireland, and Austrlia are also having good trust in the Governance of their Countries.

# # Top 10 Countries with Trust in Government

# In[ ]:


data[['Country', 'Trust..Government.Corruption.']].sort_values(by = 'Trust..Government.Corruption.', ascending = False).head(10)


# In[ ]:


trace1 = [go.Choropleth(
               colorscale = 'Portland',
               locationmode = 'country names',
               locations = data['Country'],
               text = data['Country'], 
               z = data['Family'],
               )]

layout = dict(title = 'Family Satisfaction Index',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# India and China have very low Family Satisfaction comparatively.
# 
# All over the world Family Satisfaction rate is really good.
# 
# Central African Republic has the lowest Family Satisfaction score in the world.

# # Top 10 Countries in Family Satisfaction

# In[ ]:


data[['Country', 'Family']].sort_values(by = 'Family', ascending = False).head(10)


# In[ ]:


trace1 = [go.Choropleth(
               colorscale = 'Viridis',
               locationmode = 'country names',
               locations = data['Country'],
               text = data['Country'], 
               z = data['Economy..GDP.per.Capita.'],
               )]

layout = dict(title = 'GDP in now',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# America, Canada, Austrelia, Saudi Arabia, European Countries are the Leaders in the Economy and GDP.
# 
# Smaller Countries like Norway, Qatar, Luxembourg are having the best GDP rate in the world.
# 
# Most of the African Countries are having very low GDP rate.
# 
# India, Pakistan, Myanmar are having very low GDP in the Asian Regions.

# # Top 10 Countries with Best Economy

# In[ ]:


data[['Country', 'Economy..GDP.per.Capita.']].sort_values(by = 'Economy..GDP.per.Capita.', ascending = False).head(10)


# In[ ]:


trace1 = [go.Choropleth(
               colorscale = 'Picnic',
               locationmode = 'country names',
               locations = data['Country'],
               text = data['Country'], 
               z = data['Freedom'],
               )]

layout = dict(title = 'Freedom Index',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# Looks like Canada, Austrelia and Europe are the best places on Earth to live. They have high scores throughout. In this case also They are the Winners.
# 
# Europe on Whole has a very high Freedom Index in comparison to other Coutries of the World.
# 
# African Countries such as Sudan and Angola are having the lowsest Freedom Index in the World.

# # Top 10 Most Freedom Oriented Countries

# In[ ]:


data[['Country', 'Freedom']].sort_values(by = 'Freedom', ascending = False).head(10)


# In[ ]:


trace1 = [go.Choropleth(
               colorscale = 'Electric',
               locationmode = 'country names',
               locations = data['Country'],
               text = data['Country'], 
               z = data['Happiness Rank'],
               )]

layout = dict(title = 'Happiness Rank',
                  geo = dict(
                      showframe = True,
                      showocean = True,
                      showlakes = True,
                      showcoastlines = True,
                      projection = dict(
                          type = 'hammer'
        )))


projections = [ "equirectangular", "mercator", "orthographic", "natural earth","kavrayskiy7", 
               "miller", "robinson", "eckert4", "azimuthal equal area","azimuthal equidistant", 
               "conic equal area", "conic conformal", "conic equidistant", "gnomonic", "stereographic", 
               "mollweide", "hammer", "transverse mercator", "albers usa", "winkel tripel" ]

buttons = [dict(args = ['geo.projection.type', y],
           label = y, method = 'relayout') for y in projections]
annot = list([ dict( x=0.1, y=0.8, text='Projection', yanchor='bottom', 
                    xref='paper', xanchor='right', showarrow=False )])


# Update Layout Object

layout[ 'updatemenus' ] = list([ dict( x=0.1, y=0.8, buttons=buttons, yanchor='top' )])
layout[ 'annotations' ] = annot


fig = go.Figure(data = trace1, layout = layout)
py.iplot(fig)


# # Top 10 Happiest Countries

# In[ ]:


data[['Country','Happiness.Rank']].head(10)

