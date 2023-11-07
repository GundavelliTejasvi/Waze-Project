
# # **Waze Project**
# **Course 2 - Get Started with Python**

# Welcome to the Waze Project!
# # **Course 2 End-of-course project: Inspect and analyze data**
# 
# In this activity, you will examine data provided and prepare it for analysis. This activity will help ensure the information is,
# 
# 1.   Ready to answer questions and yield insights
# 
# 2.   Ready for visualizations
# 
# 3.   Ready for future hypothesis testing and statistical methods
# <br/>
# 
# **The purpose** of this project is to investigate and understand the data provided.
# 
# **The goal** is to use a dataframe contructed within Python, perform a cursory inspection of the provided dataset, and inform team members of your findings.
# <br/>
# 
# *This activity has three parts:*
# 
# **Part 1:** Understand the situation
# * How can you best prepare to understand and organize the provided information?
# 
# **Part 2:** Understand the data
# 
# * Create a pandas dataframe for data learning, future exploratory data analysis (EDA), and statistical activities
# 
# * Compile summary information about the data to inform next steps
# 
# **Part 3:** Understand the variables
# 
# * Use insights from your examination of the summary data to guide deeper investigation into variables

# # **Identify data types and compile summary information**

# # **PACE stages**
# ## **PACE: Plan**
# ### **Task 1. Understand the situation**
# 
# *   How can you best prepare to understand and organize the provided driver data?
# ### **Task 2a. Imports and data loading**
# In[1]:


# Import packages for data manipulation
import pandas as pd
import numpy as np


# Then, load the dataset into a dataframe. Creating a dataframe will help you conduct data manipulation, exploratory data analysis (EDA), and statistical activities.

# In[2]:


# Load dataset into dataframe
df = pd.read_csv('waze_dataset.csv')


# ### **Task 2b. Summary information**
# *Consider the following questions:*
# 
# 1. When reviewing the `df.head()` output, are there any variables that have missing values?
# None of the variables in the first 10 observations have missing values. Note that this does not imply the whole dataset does not have any missing values
# 2. When reviewing the `df.info()` output, what are the data types? How many rows and columns do you have?
# The variables `label` and `device` are of type `object`; `total_sessions`, `driven_km_drives`, and `duration_minutes_drives` are of type `float64`; the rest of the variables are of type `int64`. There are 14,999 rows and 13 columns.
# 3. Does the dataset have any missing values?
#The dataset has 700 missing values in the `label` column.
# In[3]:


df.head(10)


# In[4]:


df.info()

# ### **Task 2c. Null values and summary statistics**
# 
# Compare the summary statistics of the 700 rows that are missing labels with summary statistics of the rows that are not missing any values.
# 
# **Question:** Is there a discernible difference between the two populations?
#Comparing summary statistics of the observations with missing retention labels with those that aren't missing any values reveals nothing remarkable. The means and standard deviations are fairly consistent between the two groups.
# In[5]:


# Isolate rows with null values
null_df = df[df['label'].isnull()]
# Display summary stats of rows with null values
null_df.describe()


# In[6]:


# Isolate rows without null values
not_null_df = df[~df['label'].isnull()]
# Display summary stats of rows without null values
not_null_df.describe()


# ### **Task 2d. Null values - device counts**
# 
# Next, check the two populations with respect to the `device` variable.
# 
# **Question:** How many iPhone users had null values and how many Android users had null values?
#Of the 700 rows with null values, 447 were iPhone users and 253 were Android users.

# In[7]:


# Get count of null values by device
null_df['device'].value_counts()

# Now, of the rows with null values, calculate the percentage with each device&mdash;Android and iPhone. You can do this directly with the [`value_counts()`](https://pandas.pydata.org/docs/reference/api/pandas.Series.value_counts.html) function.

# In[8]:


# Calculate % of iPhone nulls and Android nulls
null_df['device'].value_counts(normalize=True)


# How does this compare to the device ratio in the full dataset?

# In[9]:


# Calculate % of iPhone users and Android users in full dataset
df['device'].value_counts(normalize=True)


# The percentage of missing values by each device is consistent with their representation in the data overall.
# 
# There is nothing to suggest a non-random cause of the missing data.

# Examine the counts and percentages of users who churned vs. those who were retained. How many of each group are represented in the data?

# In[10]:


# Calculate counts of churned vs. retained
print(df['label'].value_counts())
print()
print(df['label'].value_counts(normalize=True))


# This dataset contains 82% retained users and 18% churned users.
# 
# Next, compare the medians of each variable for churned and retained users. The reason for calculating the median and not the mean is that you don't want outliers to unduly affect the portrayal of a typical user. Notice, for example, that the maximum value in the `driven_km_drives` column is 21,183 km. That's more than half the circumference of the earth!

# In[11]:


# Calculate median values of all columns for churned and retained users
df.groupby('label').median(numeric_only=True)


# This offers an interesting snapshot of the two groups, churned vs. retained:
# 
# Users who churned averaged ~3 more drives in the last month than retained users, but retained users used the app on over twice as many days as churned users in the same time period.
# 
# The median churned user drove ~200 more kilometers and 2.5 more hours during the last month than the median retained user.
# 
# It seems that churned users had more drives in fewer days, and their trips were farther and longer in duration. Perhaps this is suggestive of a user profile. Continue exploring!
# 
# 

# Calculate the median kilometers per drive in the last month for both retained and churned users.

# In[12]:


# Group data by `label` and calculate the medians
medians_by_label = df.groupby('label').median(numeric_only=True)
print('Median kilometers per drive:')
# Divide the median distance by median number of drives
medians_by_label['driven_km_drives'] / medians_by_label['drives']


# The median user from both groups drove ~73 km/drive. How many kilometers per driving day was this?

# In[13]:


# Divide the median distance by median number of driving days
print('Median kilometers per driving day:')
medians_by_label['driven_km_drives'] / medians_by_label['driving_days']


# Now, calculate the median number of drives per driving day for each group.

# In[14]:


# Divide the median number of drives by median number of driving days
print('Median drives per driving day:')
medians_by_label['drives'] / medians_by_label['driving_days']


# The median user who churned drove 608 kilometers each day they drove last month, which is almost 250% the per-drive-day distance of retained users. The median churned user had a similarly disproporionate number of drives per drive day compared to retained users.
# 
# It is clear from these figures that, regardless of whether a user churned or not, the users represented in this data are serious drivers! It would probably be safe to assume that this data does not represent typical drivers at large. Perhaps the data&mdash;and in particular the sample of churned users&mdash;contains a high proportion of long-haul truckers.
# 
# In consideration of how much these users drive, it would be worthwhile to recommend to Waze that they gather more data on these super-drivers. It's possible that the reason for their driving so much is also the reason why the Waze app does not meet their specific set of needs, which may differ from the needs of a more typical driver, such as a commuter.

# Finally, examine whether there is an imbalance in how many users churned by device type.
# 
# Begin by getting the overall counts of each device type for each group, churned and retained.

# In[15]:


# For each label, calculate the number of Android users and iPhone users
df.groupby(['label', 'device']).size()


# Now, within each group, churned and retained, calculate what percent was Android and what percent was iPhone.

# In[16]:


# For each label, calculate the percentage of Android users and iPhone users
df.groupby('label')['device'].value_counts(normalize=True)


# The ratio of iPhone users and Android users is consistent between the churned group and the retained group, and those ratios are both consistent with the ratio found in the overall dataset.

# ### **Task 3. Conclusion**
# 
# **Questions:**
# 
# 1. Did the data contain any missing values? How many, and which variables were affected? Was there a pattern to the missing data?
# 
# > *The dataset has 700 missing values in the `label` column. There was no obvious pattern to the missing values.*
# 
# 2. What is a benefit of using the median value of a sample instead of the mean?
# 
# > *Mean is subject to the influence of outliers, while the median represents the middle value of the distribution regardless of any outlying values.*
# 
# 3. Did your investigation give rise to further questions that you would like to explore or ask the Waze team about?
# 
# > *Yes. For example, the median user who churned drove 608 kilometers each day they drove last month, which is almost 250% the per-drive-day distance of retained users. It would be helpful to know how this data was collected and if it represents a non-random sample of users.*
# 
# 4. What percentage of the users in the dataset were Android users and what percentage were iPhone users?
# 
# > *Android users comprised approximately 36% of the sample, while iPhone users made up about 64%*
# 
# 5. What were some distinguishing characteristics of users who churned vs. users who were retained?
# 
# > *Generally, users who churned drove farther and longer in fewer days than retained users. They also used the app about half as many times as retained users over the same period.*
# 
# 6. Was there an appreciable difference in churn rate between iPhone users vs. Android users?
# 
# > *No. The churn rate for both iPhone and Android users was within one percentage point of each other. There is nothing suggestive of churn being correlated with device.*
