#Part 1

from calendar import calendar
from contextlib import closing
import requests
import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
from pandas_datareader import data
from datetime import datetime, timedelta
from pandas.tseries.holiday import USFederalHolidayCalendar
import os
import re
import json
import certifi

resp = requests.get('http://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
soup = BeautifulSoup(resp.text, 'html.parser')
table = soup.find('table', {'class': 'wikitable sortable'})

company_tickers = []
companies = []
sectors = []
subsectors = []

for row in table.findAll('tr')[1:]:
    
    company_ticker = row.findAll('td')[0].text
    company_ticker = company_ticker.strip('\n')
    company_tickers.append(company_ticker)

    company = row.findAll('td')[1].text
    companies.append(company)

    sector = row.findAll('td')[3].text
    sectors.append(sector)

    subsector = row.findAll('td')[4].text
    subsectors.append(subsector)

df = pd.DataFrame({
                "company_ticker":company_tickers,
                "company":companies,
                "sector":sectors,
                "subsector":subsectors
                    })

information_technology_df = df[df['sector'] == 'Information Technology'].reset_index(drop=True)
information_technology_df.head()

information_technology_df_summary = information_technology_df[['subsector']].groupby('subsector').size() \
    .to_frame('company_count') \
    .sort_values('company_count', ascending = False) \
    .reset_index()

information_technology_df_summary = information_technology_df_summary.append(information_technology_df_summary.sum(numeric_only=True), ignore_index=True)
information_technology_df_summary.loc[(len(information_technology_df_summary)-1),'subsector'] = 'Total # of information technology companies'
information_technology_df_summary


print(information_technology_df_summary)

df = pd.DataFrame()

# Document with apikey
creds_dir = os.path.expanduser("~/downloads/financialmodelingprep.txt")

with open(creds_dir, encoding='utf-16') as text:
    apikey = text.read().strip('\n')

# Create lists that will be looped through to pull desired transcripts

# Every information technology company in the s&p 500
company_ticker_list = information_technology_df['company_ticker']

# From years 2017, 2018, 2019, 2020 and 2021
year_list = ['2017','2018','2019','2020','2021']

# For all quaters 
quarter_list = [1,2,3,4] #[1,2,3,4]

# Pull desired transcripts using lists defined above
# There will be 1 row in the dataframe for every company + year + quarter combination

df = pd.DataFrame()

# Get the offset businessday
def offset_date(start, offset):
    return start + pd.offsets.CustomBusinessDay(n=offset, calendar=USFederalHolidayCalendar())

# NYSE & NASDAQ closing time
CLOSING_TIME = "16:00:00"

for company in company_ticker_list:
    for year in year_list:
        for quarter in quarter_list:
            # pull down transcript
            transcript = requests.get(f'https://financialmodelingprep.com/api/v3/earning_call_transcript/{company}?quarter={quarter}&year={year}&apikey={apikey}').json()
            try:
                transcript_text = transcript[0]['content'].split(' ')
                date_time = transcript[0]['date'].split(' ')
                date = date_time[0]
                time = date_time[1]
                adjusted_date = date

                if time >= CLOSING_TIME:
                    adjusted_date = offset_date(datetime.strptime(date, '%Y-%m-%d'), 1)
                    adjusted_date = adjusted_date.strftime('%Y-%m-%d')

            except:
                transcript_text = ""
                date = np.nan
                time = np.nan
                adjusted_date = np.nan
            finally:
                # store the objects in a list
                lst = [[company, year, quarter, date, adjusted_date, time, transcript_text]]
                # append list to dataframe
                df = df.append(lst)

# Dataframe formatting
df.columns = ['company_ticker', 'year', 'quarter', 'date', 'adjusted_date', 'time', 'transcript_text']
df.reset_index(drop=True, inplace = True)
df.head()

# Closing price dataframe and idx dataframe
df_closing_prices = pd.DataFrame()
df_closing_idx = pd.DataFrame()

#Import index data (SPCLEAN3)
df_idx = pd.read_csv('SPClean3.csv',sep=',')
idx_dict = {}
for ind in df_idx.index:
    idx_dict[df_idx['Date'][ind]] = (df_idx['Close'][ind])

def offset_date(start, offset):
    return start + pd.offsets.CustomBusinessDay(n=offset, calendar=USFederalHolidayCalendar())


# Looping over the rows in the dataframe
for ind in df.index:
    company = df['company_ticker'][ind]
    date = df['adjusted_date'][ind]
    closing_prices = [np.nan] * 161
    idx_prices = [np.nan] * 161

    # If the date isn't nan check the closing prices
    if not pd.isna(date):
        # Formating the dates to datetime to make it easier to add and subract number of days, then making them strings again

        date_today = datetime.strptime(date, '%Y-%m-%d')
        five_days_after = offset_date(date_today, 5)

        # Getting the values from the API
        transcript = requests.get(f'https://financialmodelingprep.com/api/v3/historical-price-full/{company}?apikey={apikey}').json()
        transcript = transcript['historical']

        # Looping over all of the dates from the request above
        # Could be improved by just checking an interval
        # In that case, change the api request to include 'from' and 'to' arguments.
        for i in range(len(transcript)):
            day = transcript[i]
            day_date = datetime.strptime(day['date'], '%Y-%m-%d')

            # Checking for the sought-after dates, if found add the value to the correct column in the dataframe
            if day_date == five_days_after:
                idx = 0
                for j in range(i, i+161):
                    try:
                        day_ = transcript[j]
                        idx_prices[idx] = idx_dict[day_['date']]
                        closing_prices[idx] = day_['close']
                    except:
                        continue
                    idx += 1    
                break

    df_closing_prices = df_closing_prices.append([closing_prices])
    df_closing_idx = df_closing_idx.append([idx_prices])
    
# Event day will be day #5
column_names = []
for i in range(161):
    column_names.append("day_" + str(i))


df_closing_prices.columns = column_names
df_closing_prices.reset_index(drop=True, inplace=True)

df_closing_idx.columns = column_names
df_closing_idx.reset_index(drop=True, inplace=True)

# Print dataframes to csv files
df.to_csv('data_earningcalls.csv')
df_closing_prices.to_csv('data_stock_closing.csv')
df_closing_idx.to_csv('data_SP500_index.csv')
