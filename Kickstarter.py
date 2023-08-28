#!/usr/bin/env python
# coding: utf-8

# # Introduction to the dataset
# 

# ### Title : Why do Kickstarter campaigns fail?
# For this project, we used Kickstarter campaigns dataset from Kaggle: https://www.kaggle.com/datasets/thedevastator/most-kickstarter-campaigns-fail-here-s-why
# 
# This dataset contains data on 20,632 Kickstarter campaigns as of February 1st, 2017. Important attributes are described below: 
# - Project: a finite work with a clear goal that you’d like to bring to life (aka campaign)
# - Funding goal: amount of money that a creator needs to complete their project
# - Name: name of project on Kickstarter
# - Blurb: the short description displayed under the name of your project and on the browse page
# - Pledged and backers: amount of money that a project has raised and people that have supported it at the point of the API pull
# - State: successful, failed, cancelled, live or suspended.

# ## 0. Importing Libraries

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
from bokeh.plotting import figure, show
import numpy as np
from math import pi
import random
import seaborn as sns


# ## 1. Read CSV file and then load into a data frame

# In[2]:


#path= r"kickstarter_data_full.csv"
initdf= pd.read_csv('kickstarter_data_full.csv')
# initdf


# In[ ]:





# ## 2. Inferred Schema

# In[3]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------
# Schema information
initdf.info()


# #### Checking Columns with Null Values

# In[4]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

print([col for col in initdf.columns if initdf[col].isnull().any()])


# #### After running the above fucntion, we found out the columns with missing values to handle them.  

# #### Columns 'blurb', blurb_len', 'blurb_len_clean', 'location', 'name', 'country', 'is_starred', friends', 'permissions', 'is_backing'  have missing values which are required to be handled. 

# ## 3. Data Cleaning and Transformations

# In[5]:


#Checking rows for which specified column contains null
def nonnullcheck(df, col):
    return df[col][df[col].notna()]


# In[6]:


# function to check column values of null values 
def nullcheck(df, col):
    return df[col][df[col].isna()]

#insert the column name for which you wish to check
nullcheck(initdf,'blurb_len_clean') 


# 

# In[7]:


#Handling blurb
initdf["blurb"].fillna("Missing blurb", inplace=True)
initdf["blurb_len"].fillna(0, inplace=True)
initdf["blurb_len_clean"].fillna(0, inplace=True)


# In[8]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------
initdf[['blurb','blurb_len','blurb_len_clean']]


# In[9]:


# Checking null rows for location
nullcheck(initdf,'location')


# In[10]:


# HANDLING LOCATION NULL VALUES
# Replacing the Null values in location column with "No Location Specified"
initdf["location"].fillna("No location specified", inplace=True)


# In[11]:


# Chekcing 'name', 'location' and 'country' for rows which has unspecified location
initdf[['name', 'location', 'country']][initdf['location'] == "No location specified"]


# #### Checking null values in columns: 'is_starred', friends', 'permissions' &'is_backing'.

# In[12]:


#Checing the Non-null Values and evaluating their impact. 
nonnullcheck(initdf, 'is_backing')


# In[13]:


#Checing the Non-null Values and evaluating their impact. 
nonnullcheck(initdf, 'is_starred')


# In[14]:


#Checing the Non-null Values and evaluating their impact. 

nonnullcheck(initdf, 'friends')


# In[15]:


#Checing the Non-null Values and evaluating their impact. 

nonnullcheck(initdf, 'permissions')


# ### DROPPING UNNECESSARY COLUMNS
# 

# In[16]:


# Since the maximum values columns 'friends', 'is_starred', 'is_backing'& 'permissions' is either Null or inserted by the author which will also not affect out visualisations, we will drop these four columns.

initdf.drop(['friends', 'is_starred', 'is_backing', 'permissions'], inplace=True, axis=1)


# In[17]:


# To get the look of new Dataframe
initdf.info()


# In[18]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------
nullcheck(initdf, 'category')


# In[19]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

nonnullcheck(initdf, 'category')


# In[20]:


#For all the Null values in categories, we are grouping them and assigning them a value for a cleaner outlay of the dataset.

initdf[['name','category']][initdf['category']=="Uncategorized"]


# In[21]:


# Handling categories
initdf["category"].fillna("Uncategorized", inplace=True)


# In[22]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

#check if all null values have been handled
print([col for col in initdf.columns if initdf[col].isnull().any()])


# In[23]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------


nonnullcheck(initdf, 'name_len')


# In[24]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------


nullcheck(initdf, 'name_len')


# In[25]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------


initdf[['name','name_len','name_len_clean']][initdf['name_len']==0.0]


# In[26]:


# DROPPING INCOMPLETE DATA ROWS

initdf.drop(labels=[1411,6744,9239,11708,14805], axis=0, inplace=True)


# In[27]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------
# Checking for Null values in Dataframe

initdf.isnull().values.any()


# ## VALIDATION TEST CELLS 

# In[28]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

# initdf.info()


# In[29]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

# Checking count of categories by their states
# initdf.groupby("state")[["category"]].count()


# In[30]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

# Count the number of unique values
# initdf[["category","state"]].nunique()


# In[31]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

# List number of unique values
# initdf["state"].unique()


# In[32]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

# initdf["category"].unique()


# ### CREATING DATAFRAMES BY STATE
# 

# In[33]:


failed_df = initdf[initdf["state"] == "failed"].groupby(["category"])[['state']].count().reset_index()
successful_df = initdf[initdf["state"] == "successful"].groupby(["category"])[['state']].count().reset_index()
canceled_df = initdf[initdf["state"] == "canceled"].groupby(["category"])[['state']].count().reset_index()
live_df = initdf[initdf["state"] == "live"].groupby(["category"])[['state']].count().reset_index()
suspended_df = initdf[initdf["state"] == "suspended"].groupby(["category"])[['state']].count().reset_index()


# ### CREATING DEFAULT DICTIONARY FOR STATES

# In[34]:




categories_dict = {
    'Academic':0,
    'Places':0,
    'Uncategorized':0,
    'Blues':0,
    'Restaurants':0,
    'Webseries':0, 
    'Thrillers':0, 
    'Shorts':0, 
    'Web':0, 
    'Apps':0, 
    'Gadgets':0,
    'Hardware':0, 
    'Festivals':0, 
    'Plays':0, 
    'Musical':0, 
    'Flight':0, 
    'Spaces':0,
    'Immersive':0, 
    'Experimental':0, 
    'Comedy':0, 
    'Wearables':0, 
    'Sound':0,
    'Software':0, 
    'Robots':0, 
    'Makerspaces':0
}


# In[35]:


#    METHOD TO RETURN SERIES PER STATE FOR VISUALIZATION
def populating_state_series(df):

    shallow_copy = categories_dict.copy()
    for i in range(len(df)):
        if df['category'][i] in shallow_copy:
            shallow_copy[df['category'][i]] = df['state'][i]
#     print(shallow_copy)
    return list(shallow_copy.values())


# In[36]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------

# categories_dict


# In[37]:


# CALLING METHOD TO RETURN SERIES PER STATE FOR VISUALIZATION

categories = list(categories_dict.keys())
failed = populating_state_series(failed_df)
successful = populating_state_series(successful_df)
canceled = populating_state_series(canceled_df)
live = populating_state_series(live_df)
suspended = populating_state_series(suspended_df)


# In[38]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------


# print(categories)
# print(successful)
# print(failed)
# print(canceled)
# print(live)
# print(suspended)


# In[39]:


# VALIDATION TEST CELL -------------------------------------------------------------------------------------------------


# print(len(canceled), len(successful), len(live), len(suspended), len(failed))
# states = list(initdf["state"].unique())
# print(states)


# # 4. VISUALISATIONS

# ## 4.1 BAR CHART 

# In[40]:


from bokeh.plotting import figure, show
from bokeh.plotting import show, output_notebook, output_file
output_notebook()

categories = list(categories_dict.keys())
states = list(initdf["state"].unique())
colors = ["orange", "red", "green","blue", "silver"]

data = {'categories' : categories}
data['failed'] = populating_state_series(failed_df)
data['successful'] = populating_state_series(successful_df) 
data['canceled'] = populating_state_series(canceled_df)
data['live'] = populating_state_series(live_df)
data['suspended'] = populating_state_series(suspended_df)
 

# print(data)
p = figure(x_range=categories, height=500, title="States of Kicstarter by Categories",
           toolbar_location=None, tools='hover', tooltips="$name @categories: @$name")

p.vbar_stack(states, x='categories', width=0.9, color=colors, source=data,
             legend_label=states)

p.y_range.start = 0
p.x_range.range_padding = 0.1
p.xgrid.grid_line_color = None
p.axis.minor_tick_line_color = None
p.outline_line_color = None
p.legend.location = "top_left"
p.legend.orientation = "horizontal"
p.xaxis.major_label_orientation = "vertical"
p.xaxis.axis_label = 'Categories'
p.yaxis.axis_label = 'No.of Kickstarters'


show(p)


# ### 4.1.1 Inference from the Graph

# - We have total 25 Categories which contain 20627 kickstarters combined. 
# 
# 
# 
# - Top 5 famous categires with most kickstarters are:
#     1. Web 
#     2. Hardware 
#     3. Software 
#     4. Gadgets 
#     5. Uncategorized 
#     
#     
# - Top 5 Successful categories are:
#     1. Hardware 
#     2. Uncategorized
#     3. Plays
#     4. Gadgets
#     5. Musical
#     
#     
# - Top 5 Failed catergories are:
#     1. Web 
#     2. Software
#     3. Hardware 
#     4. Gadgets 
#     5. Uncategorized 
#   
# 

# In[41]:


# METHOD TO RETURN Categories PER STATE FOR VISUALIZATION

def populating_categories_dictionary(df):
    shallow_copy = categories_dict.copy()
    for i in range(len(df)):
        if df['category'][i] in shallow_copy:
            shallow_copy[df['category'][i]] = df['state'][i]
#             shallow_copy['academic'] = 20
#     print(shallow_copy)
    return shallow_copy


# ## 4.2 PIE CHART

# In[42]:


from math import pi
import pandas as pd
from bokeh.io import output_notebook, show
from bokeh.palettes import Category20c
import random
from bokeh.plotting import figure
from bokeh.transform import cumsum
from bokeh.models import LabelSet, ColumnDataSource

output_notebook()

pie_colors=['#039d72','#45BA7E','#de324c', '#f4895f', '#f8e16f',
            '#95cf92','#369acc','#9656a2','#B74E09','#61B22E',
            '#4B2DF7','#5EB999','#5DBDE7','#DD629B','#B2A6A3',
            '#C9212C','#E63DC4', '#A13C50','#4E4327','#76A9CA',
            '#DD7C03','#DD7C03','#80D077','#D84CE4', '#D67956']


total_df = initdf.groupby("category")["state"].count().reset_index()
x = populating_categories_dictionary(total_df)
data = pd.Series(x).reset_index(name='value').rename(columns={'index':'category'})
# print(data)

data['angle'] = data['value']/data['value'].sum() * 2*pi
data['color'] = ['#039d72','#45BA7E','#de324c', '#f4895f', '#f8e16f',
            '#95cf92','#369acc','#9656a2','#B74E09','#61B22E',
            '#4B2DF7','#5EB999','#5DBDE7','#DD629B','#B2A6A3',
            '#C9212C','#E63DC4', '#A13C50','#4E4327','#76A9CA',
            '#DD7C03','#DD7C03','#80D077','#D84CE4', '#D67956']

p = figure(plot_height=800, title="Pie Chart", toolbar_location=None,
           tools="hover", tooltips="@category: @value", x_range=(-0.8, 1.8))

p.wedge(x=0, y=1, radius=0.8,
        start_angle=cumsum('angle', include_zero=True), end_angle=cumsum('angle'),
        line_color="white", fill_color='color', legend='category', source=data)

data["value"] = data['value'].astype(str)
data["value"] = data["value"].str.pad(35, side = "left")
source = ColumnDataSource(data)

labels = LabelSet(x=0, y=1, text='value',
        angle=cumsum('angle', include_zero=True), source=source, render_mode='canvas')

p.add_layout(labels)

p.axis.axis_label=None
p.axis.visible=False
# p.grid.grid_line_color = None

show(p)


# ### 4.2.1Inference from the Pie Chart
# 

# - The above visualisation consist of a simple pie chart showing distribution of all the Kickstarters in various categories
# 
# 
# 
# - Pie chart suggests that most of the kickstarters are famous amongts the following categories:
#    1. WEB
#    2. HARDWARE
#    3. SOFTWARE
#    4. GADGETS
#    5. UNCATEGORIZED

# ##### CREATING A DATAFRAME FOR LINE CHART

# In[43]:


# Using Groupby, Checking count of kickstarter(name) by their launch year which have state as successful. 

initdf[initdf['state'] == 'successful'].groupby('launched_at_yr')['name'].count()


# In[44]:


# Using Groupby, Checking count of kickstarter(name) by their launch year which have state as Failed. 

initdf[initdf['state'] == 'failed'].groupby('launched_at_yr')['name'].count()


# In[45]:


# CALLING METHOD TO RETURN SERIES PER Launch Year  FOR VISUALIZATION

# This method will call all the kickstarters according to their launch year.
total_by_yr = initdf.groupby(['launched_at_yr'])[['name']].count().reset_index()

# This method will call all the sucessful kickstarters according to their launch year.
successful_by_yr = initdf[initdf['state'] == 'successful'].groupby('launched_at_yr')['name'].count().reset_index()

# This method will call all the failed kickstarters according to their launch year.
failed_by_yr = initdf[initdf['state'] == 'failed'].groupby('launched_at_yr')['name'].count().reset_index()


# In[46]:


# VALIDATION TEST CELL ------------------------------------------------------------------------------------------------
#list(successful_by_yr['name'])


# ## 4.3 LINE GRAPH

# In[47]:


from bokeh.plotting import figure, show
# prepare some data
x = list(total_by_yr['launched_at_yr']) #list of years
y1 = list(total_by_yr['name']) #count(name/anything) for total by year 
y2 = list(failed_by_yr['name']) #count(name/anything) for failed by year 
y3 = list(successful_by_yr['name']) #count(name/anything) for successful by year 

# create a new plot with a title and axis labels
p = figure(title="Total campaign vs Success/Failure rate", x_axis_label="Year", y_axis_label="Value", plot_width = 600, plot_height = 400)

# add multiple renderers
p.line(x, y1, legend_label="Total.", color="blue", line_width=2)
p.line(x, y2, legend_label="Failed", color="red", line_width=2)
p.line(x, y3, legend_label="Successful", color="green", line_width=2)
# show the results


show(p)


# ### 4.3.1 Inference from Line Graph

# - The graph shows the camparision between the Total campaigns, the total Failed and succesful campaign
# 
# - From the graph we can infer that, there are more number of failed campaings than successful ones. 
# 
# - Most of the campaigns from this dataset is from the year 2013 to year 2018.
# 

# ## 4.4 HEAT MAP

# In[48]:


pivot_table= initdf.pivot_table(index="category",columns="country",values="backers_count",aggfunc='mean')
color = plt.get_cmap('RdYlGn') 
color.set_bad('maroon')
sns.heatmap(pivot_table,cmap=color)
plt.title('Average Number of backers across Countries per category')
plt.xlabel('Country')
plt.ylabel('Category')
print("HeatMap")


# ### 4.4.1 Inference from Heat Map

# - The heat map shows that for certain categories like Gadgets, Hardware and Robots, there are a large number of backers in a single particular country such as China, Sweden and the Netherlands respectively for these categories. So launching these type of kickstarters in these countries will increase their success rate.
# 
# - There are a lot of bad values being seen in the heat map marked which are marked in maroon, which indicates that a huge number of kickstarters launched have 0 backers which contributes to them failing.
# 

# ## 4.5 HISTOGRAM

# In[49]:


sns.histplot(data=initdf,x="launched_at_month",hue="state",bins=12)
plt.title('Number of campaigns launched in each month')
plt.xlabel('Launch Month')
plt.ylabel('Number of Campaigns')


# ### 4.5.1 Inference from Histogram

# - We can infer from the histogram that, there is a high success rate for kickstarters launched from May to June. We can use this inference and increase the success rate of kickstartes by launching them in those particular months.
# 
# 
# - The failure rate seems to be high during December and January and it would be best to avoid those months for starting new projects.

# ## 4.6 BOXPLOT

# In[50]:


initdf['usd_goal'] = initdf['goal']*initdf['static_usd_rate'] # To convert currency of goal to US currency
boxdf = initdf[['category','SuccessfulBool', 'state', 'launch_to_deadline_days']]
boxdf.groupby(initdf.category).mean().reset_index()
failed = boxdf[boxdf['SuccessfulBool']==0][['launch_to_deadline_days','category' ,'state']]
success = boxdf[boxdf['SuccessfulBool']==1][['launch_to_deadline_days','category','state']]
x = failed['launch_to_deadline_days']
y = failed['category']
sns.set(rc={'figure.figsize':(15,6)})
sns.boxplot(x,y)
plt.title('Duration of campaign for all categories')
plt.xlabel('Length of campagin in days')
plt.ylabel('Categories')
plt.savefig('boxplot.png')


# ## 4.6.1 Inference from boxplot
# - It is recommended from the kickstarter webiste that for a campaign to have higher chances of success, its better to have the campaign duration of 30 days or less. 
# - From the visualization, it can be infered that most of the failed projects are run for more than 40-60 days. 
# - From the outliers, we can infer that most of the campaigns failed because their duration was too less to have enough time to gather enough funding or run for a long period 50 days and above.

# ## 4.7 Scatter Plot

# In[51]:


scatterdf = initdf[['usd_goal','category','usd_pledged','SuccessfulBool', 'state','backers_count']]
# filtering the failed projects 
failed = scatterdf[scatterdf['SuccessfulBool']==0][['usd_goal','backers_count','category' ,'usd_pledged','state']]
# filtering the successful projects
success = scatterdf[scatterdf['SuccessfulBool']==1][['usd_goal','backers_count','category' ,'usd_pledged','state']]
sns.set(font_scale=1.3)
#Extracting average values for all categories
avg1 = failed.groupby(initdf.category).mean().reset_index()
avg2 = success.groupby(initdf.category).mean().reset_index()
fig, ax = plt.subplots()
sns.scatterplot(avg1['usd_goal'],avg1['category'], size = avg1['backers_count'], color='r', label='failed')
sns.scatterplot(avg2['usd_goal'],avg2['category'], size = avg2['backers_count'], color='g', label='successful')
ax.set_xlim(1, 150000)
plt.title('Average project goal(Successful/Failed) per category')
plt.xlabel('Average goal in USD')
plt.ylabel('Categories')
plt.savefig('scatterplot.png')
plt.show()


# ## 4.7.1 Inference from Scatter Plot
# - Across all categories, the campaigns are most likely to be successful if their funding goal is below 20,000 dollars.
# - The median goal amount of successful projects is 6,000 dollars while failed projects is 1,15,171 dollars which is more than double the successful projects. 
# - This suggests that projects with a conservative goal are more likely to attract backers.
# - We can also see that failed projects across all categories have higher funding goals than successful ones. 
# - It can be infered that Gadgets, Hardware and wearables attract more backers.

# ##  5. Conclusion
# 

# ### 5.1 Insights from the visualizations
# Kickstarter data has a lot of valuable insights to offer. Findings from our analysis:
# 
# - The average success ratio of campaign on KickStarter is about 29% from 2008-2017.
# - Top 5 successful categories are Web, Hardware, Software, Gadgets and Uncategorized. 
# - For hardware, kickstarter is found to be most popular in countries like China, Sweden and Netherlands as these countries have more average number of backers.
# - The median goal amount for a successful project is found to be USD 6,000.
# - Success ratio of campaigns was on a increasing trend from 2013 to 2015. Post 2015 there is a gradual downfall in the success ratio of the campaigns.
# - Success rate is found to be higher in the period of May to June. 
# - The failure rate seems to be high during December and January.
# - Any kickstarter which is run for a duration longer than 30 days is mostly likely to fail across all categories.

# ### 5.2 Reasons of Failure
# 
# Crowdfunding has become increasing popular in the world as a form of alternative financing for new projects.
# 1. Market research failure
# 2. Product not been promising enough
# 3. Running out of cash
# 4. Over ambitious projects
# 5. Business model failure
# 6. Deadline extended forever

# ## 6. Scope for future analysis

# ### 6.1 Columns to be inserted for better visualizations
# 1. Marketing Budget
# 2. Backers Feedback
# 3. Duration of time it was live

# ### 6.2 Insights on future analysis
# 
# 1. Market Research: Good market research plays a crucial role in the Success or Failure of any Kickstarter. 
# 2. Branding of the Campaign: Often developers focus too much on creating a product they often spend less time on branding their product. The title is too lame, the blurb doesn't provide a detailed insight on what the campaign is about. 
# 3. Marketing the Product: Marketing does look like an additional cost to many developers and hence they overlook it, but it should rather be considered as a RETURN OF INVESTMENT. Since the kickstarter is dependent on the community to fund their project even before it has been finished, it is important to market it smartly, so that it reaches maximum audience and attracts potential backers.
# 4. Realistic Goals: Often with lack of market research and product knowledge, developers set goals which are too high to reach and hence the kickstarter often turns into a failure. 
# 5. Unprofessionalism: A biggest mistake amongst most of the kickstarter is not updating backers on the progress of the project. Not being in touch too often. And not completing within a deadline. 
# 6. Consumer Analysis: Having a sheer knowledge on which market has what percentage of consumers is crucial. Laying out surveys and collecting statistics on what a consumer wants will always give better results and kickstarters should consider that. 
# 7. Right Timing: Timing is everything, be it for asking for funding, launching a product or marketing it. One should know the best time as per the market for every scenario.

# In[ ]:




