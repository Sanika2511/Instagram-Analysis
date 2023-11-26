#!/usr/bin/env python
# coding: utf-8

# Let's start by importing the libraries that we will be using.

# In[19]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator


# In[20]:


from sklearn.model_selection import train_test_split
from sklearn.linear_model import PassiveAggressiveRegressor


# In[21]:


data = pd.read_csv("Instagram Data.csv", encoding = 'latin1')
data.head()


# In[22]:


#Before starting everything, let’s have a look at whether this dataset contains any null values or not:
data.isnull().sum()


# In[23]:


data.info()


# # Analyzing Instagram Reach

# Now let’s start with analyzing the reach of my Instagram posts. I will first have a look at the distribution of impressions I have received from home:
# 

# In[24]:


plt.figure(figsize=(10, 8))
plt.style.use('fivethirtyeight')
plt.title("Distribution of Impressions From Home")
sns.distplot(data['From Home'])
plt.show()


# The impressions I get from the home section on Instagram shows how much my posts reach my followers. Looking at the impressions from home, I can say it’s hard to reach all my followers daily.
# 

# Now let’s have a look at the distribution of the impressions I received from hashtags

# In[25]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Hashtags")
sns.distplot(data['From Hashtags'])
plt.show()


# Hashtags are tools we use to categorize our posts on Instagram so that we can reach more people based on the kind of content we are creating. Looking at hashtag impressions shows that not all posts can be reached using hashtags, but many new users can be reached from hashtags.   

# Now let’s have a look at the distribution of impressions I have received from the explore section of Instagram:
# 

# In[26]:


plt.figure(figsize=(10, 8))
plt.title("Distribution of Impressions From Explore")
sns.distplot(data['From Explore'])
plt.show()


# The explore section of Instagram is the recommendation system of Instagram. It recommends posts to the users based on their preferences and interests. By looking at the impressions I have received from the explore section, I can say that Instagram does not recommend our posts much to the users. Some posts have received a good reach from the explore section, but it’s still very low compared to the reach I receive from hashtags.
# 

# Now let’s have a look at the percentage of impressions I get from various sources on Instagram:

# In[27]:


home = data["From Home"].sum()
hashtags = data["From Hashtags"].sum()
explore = data["From Explore"].sum()
other = data["From Other"].sum()

labels = ['From Home','From Hashtags','From Explore','Other']
values = [home, hashtags, explore, other]

fig = px.pie(data, values=values, names=labels, 
             title='Impressions on Instagram Posts From Various Sources', hole=0.5)
fig.show()


# So the above donut plot shows that almost 44 per cent of the reach is from my followers, 33.6 per cent is from hashtags, 19.2 per cent is from the explore section, and 3.05 per cent is from other sources.

# # #Analyzing Content

# Now let’s analyze the content of my Instagram posts. The dataset has two columns, namely caption and hashtags, which will help us understand the kind of content I post on Instagram.
# 
# Let’s create a wordcloud of the caption column to look at the most used words in the caption of my Instagram posts:

# In[28]:


text = " ".join(i for i in data.Caption)
stopwords = set(STOPWORDS)

wordcloud = WordCloud(stopwords=stopwords, background_color="white").generate(text)

plt.style.use('classic')
plt.figure(figsize=(12, 10))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()


# # #Analyzing Relationships
# 

# Now let’s analyze relationships to find the most important factors of our Instagram reach. It will also help us in understanding how the Instagram algorithm works.
# Let’s have a look at the relationship between the number of likes and the number of impressions on my Instagram posts

# In[29]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Likes", size="Likes", trendline="ols", 
                    title = "Relationship Between Likes and Impressions")
figure.show()


# There is a linear relationship between the number of likes and the reach I got on Instagram.

# Now let’s see the relationship between the number of comments and the number of impressions on my Instagram posts:

# In[30]:


figure = px.scatter(data_frame = data, x="Impressions",
                    y="Comments", size="Comments", trendline="ols", 
                    title = "Relationship Between Comments and Total Impressions")
figure.show()


# It looks like the number of comments we get on a post doesn’t affect its reach.

# Now let’s have a look at the relationship between the number of shares and the number of impressions:

# In[31]:


figure = px.scatter(data_frame = data, x="Impressions",
                   y="Shares", size="Shares", trendline="ols", 
                   title = "Relationship Between Shares and Total Impressions")
figure.show()


# A more number of shares will result in a higher reach, but shares don’t affect the reach of a post as much as likes do.

# Now let’s have a look at the relationship between the number of saves and the number of impressions:

# In[32]:


figure = px.scatter(data_frame = data, x="Impressions",
                   y="Saves", size="Saves", trendline="ols", 
                   title = "Relationship Between Post Saves and Total Impressions")
figure.show()


# There is a linear relationship between the number of times my post is saved and the reach of my Instagram post.

# Now let’s have a look at the correlation of all the columns with the Impressions column:

# In[33]:


correlation = data.corr()
print(correlation["Impressions"].sort_values(ascending=False))


# So we can say that more likes and saves will help you get more reach on Instagram. The higher number of shares will also help you get more reach, but a low number of shares will not affect your reach either.

# # Analyzing Conversion Rate 

# In Instagram, conversation rate means how many followers you are getting from the number of profile visits from a post. The formula that you can use to calculate conversion rate is (Follows/Profile Visits) * 100. Now let’s have a look at the conversation rate of my Instagram account:
# 

# In[34]:


conversion_rate = (data["Follows"].sum() / data["Profile Visits"].sum()) * 100
print(conversion_rate)


# So the conversation rate of my Instagram account is 41% which sounds like a very good conversation rate. 
# Let’s have a look at the relationship between the total profile visits and the number of followers gained from all profile visits:

# In[35]:


figure = px.scatter(data_frame = data, x="Profile Visits",
                    y="Follows", size="Follows", trendline="ols", 
                    title = "Relationship Between Profile Visits and Followers Gained")
figure.show()


# The relationship between profile visits and followers gained is also linear.

# # Instagram Reach Prediction Model

# Now in this section, I will train a machine learning model to predict the reach of an Instagram post. Let’s split the data into training and test sets before training the model:
# 

# In[36]:


x = np.array(data[['Likes', 'Saves', 'Comments', 'Shares', 
                   'Profile Visits', 'Follows']])
y = np.array(data["Impressions"])
xtrain, xtest, ytrain, ytest = train_test_split(x, y, 
                                                test_size=0.2, 
                                                random_state=42)


# Now here’s is how we can train a machine learning model to predict the reach of an Instagram post using Python:

# In[38]:


model = PassiveAggressiveRegressor()
model.fit(xtrain, ytrain)
model.score(xtest, ytest)


# This gives you a measure of the model's performance on the test data. Now let’s predict the reach of an Instagram post by giving inputs to the machine learning model:

# Features = [['Likes','Saves', 'Comments', 'Shares', 'Profile Visits', 'Follows']]

# In[40]:


features = np.array([[282.0, 233.0, 4.0, 9.0, 165.0, 54.0]])
model.predict(features)


# # Summary

# The predicted reach for my Instagram project, using the provided features (Likes, Saves, Comments, Shares, Profile Visits, Follows), is approximately 12007.8  
# So this is how you can analyze and predict the reach of Instagram posts with machine learning using Python. If a content 
# creator wants to do well on Instagram in a long run, they have to look at the data of their Instagram reach. That is where the 
# use of Data Science in social media comes in. 
