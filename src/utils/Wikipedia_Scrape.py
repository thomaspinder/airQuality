#!/usr/bin/env python
# coding: utf-8

# # County GVA Scrape

# In[3]:


from bs4 import BeautifulSoup
import requests
import pandas as pd
import re
import sys
sys.path.append('../')


# ## Obtain Wikipedia source HTML

# In[4]:


url = 'https://en.wikipedia.org/wiki/List_of_ceremonial_counties_in_England_by_gross_value_added'
html = requests.get(url)
soup = BeautifulSoup(html.content)


# ## Extract table values

# In[5]:


ranks = []
counties = []
values = []
gva_table = soup.find('table', {'class': 'wikitable'})
for row in gva_table.find_all('tr'):
    try:
        ranks.append(row.findAll('td')[0].text)
        counties.append(row.findAll('td')[1].text)
        values.append(row.findAll('td')[2].text)
    except:
        pass

values = [re.search(r'\d+(\.\d{1,2})?', value)[0].strip() for value in values]


# ## Consolidate results
# 

# In[6]:


county_gva = pd.DataFrame({
    'rank': ranks,
    'county': counties,
    'value (£billions)': values
})


# ## Write to csv

# In[8]:


county_gva.to_csv('../data/county_gva.csv')

