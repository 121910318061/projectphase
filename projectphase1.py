#!/usr/bin/env python
# coding: utf-8

# In[ ]:





# In[1]:


import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
import yfinance as yf


# In[4]:


# import package
import pandas_datareader.data as web
# set start and end dates 
start = datetime.datetime(2018, 1, 20) 
end = datetime.datetime(2022,10,12)
# extract the closing price data
ultratech_df = web.DataReader(['HDFC.NS'], 'yahoo', start = start, end = end)['Close']
ultratech_df.columns = {'Close Price'}
ultratech_df.head(10)


# In[5]:


ultratech_df['Close Price'].plot(figsize = (15, 8))
plt.grid()
plt.ylabel("Price in Rupees")
plt.show()


# In[6]:


# create 20 days simple moving average column
ultratech_df['20_SMA'] = ultratech_df['Close Price'].rolling(window = 20, min_periods = 1).mean()
# create 50 days simple moving average column
ultratech_df['50_SMA'] = ultratech_df['Close Price'].rolling(window = 50, min_periods = 1).mean()
# display first few rows
ultratech_df.head()


# In[7]:


ultratech_df['Signal'] = 0.0
ultratech_df['Signal'] = np.where(ultratech_df['20_SMA'] > ultratech_df['50_SMA'], 1.0, 0.0)


# In[8]:


ultratech_df['Position'] = ultratech_df['Signal'].diff()
# display first few rows
ultratech_df.head()


# In[9]:


plt.figure(figsize = (20,10))
# plot close price, short-term and long-term moving averages 
ultratech_df['Close Price'].plot(color = 'k', label= 'Close Price') 
ultratech_df['20_SMA'].plot(color = 'b',label = '20-day SMA') 
ultratech_df['50_SMA'].plot(color = 'g', label = '50-day SMA')
# plot ‘buy’ signals
plt.plot(ultratech_df[ultratech_df['Position'] == 1].index, 
         ultratech_df['20_SMA'][ultratech_df['Position'] == 1], 
         '^', markersize = 15, color = 'g', label = 'buy')
# plot ‘sell’ signals
plt.plot(ultratech_df[ultratech_df['Position'] == -1].index, 
         ultratech_df['20_SMA'][ultratech_df['Position'] == -1], 
         'v', markersize = 15, color = 'r', label = 'sell')
plt.ylabel('Price in Rupees', fontsize = 15 )
plt.xlabel('Date', fontsize = 15 )
plt.title('ULTRACEMCO', fontsize = 20)
plt.legend()
plt.grid()
plt.show()


# In[11]:


# set start and end dates
start = datetime.datetime(2018, 1, 1)
end = datetime.datetime(2022, 10, 12)
# extract the daily closing price data
ultratech_df = web.DataReader(['HDFC.NS'], 'yahoo', start = start, end = end)['Close']
ultratech_df.columns = {'Close Price'}
# Create 20 days exponential moving average column
ultratech_df['20_EMA'] = ultratech_df['Close Price'].ewm(span = 20, adjust = False).mean()
# Create 50 days exponential moving average column
ultratech_df['50_EMA'] = ultratech_df['Close Price'].ewm(span = 50, adjust = False).mean()
# create a new column 'Signal' such that if 20-day EMA is greater   # than 50-day EMA then set Signal as 1 else 0
  
ultratech_df['Signal'] = 0.0  
ultratech_df['Signal'] = np.where(ultratech_df['20_EMA'] > ultratech_df['50_EMA'], 1.0, 0.0)


# create a new column 'Position' which is a day-to-day difference of # the 'Signal' column
ultratech_df['Position'] = ultratech_df['Signal'].diff()
plt.figure(figsize = (20,10))

# plot close price, short-term and long-term moving averages 
ultratech_df['Close Price'].plot(color = 'k', lw = 1, label = 'Close Price')  
ultratech_df['20_EMA'].plot(color = 'b', lw = 1, label = '20-day EMA') 
ultratech_df['50_EMA'].plot(color = 'g', lw = 1, label = '50-day EMA')


# plot ‘buy’ and 'sell' signals
plt.plot(ultratech_df[ultratech_df['Position'] == 1].index, 
         ultratech_df['20_EMA'][ultratech_df['Position'] == 1], 
         '^', markersize = 15, color = 'g', label = 'buy')
plt.plot(ultratech_df[ultratech_df['Position'] == -1].index, 
         ultratech_df['20_EMA'][ultratech_df['Position'] == -1], 
         'v', markersize = 15, color = 'r', label = 'sell')
plt.ylabel('Price in Rupees', fontsize = 15 )
plt.xlabel('Date', fontsize = 15 )
plt.title('ULTRACEMCO - EMA Crossover', fontsize = 20)
plt.legend()
plt.grid()
plt.show()


# In[12]:


get_ipython().run_line_magic('matplotlib', 'inline')
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import datetime
from tabulate import tabulate
import warnings
warnings.filterwarnings('ignore')
import pandas_datareader.data as web

def MovingAverageCrossStrategy(stock_symbol = 'ULTRACEMCO.NS', start_date = '2018-01-01', 
                               short_window = 20, long_window = 50, moving_avg = 'SMA', display_table = True):
    '''
    The function takes the stock symbol, time-duration of analysis, 
    look-back periods and the moving-average type(SMA or EMA) as input 
    and returns the respective MA Crossover chart along with the buy/sell signals for the given period.
    '''
    # stock_symbol - (str)stock ticker as on Yahoo finance. Eg: 'ULTRACEMCO.NS' 
    # start_date - (str)start analysis from this date (format: 'YYYY-MM-DD') Eg: '2018-01-01'
    # end_date - (str)end analysis on this date (format: 'YYYY-MM-DD') Eg: '2020-01-01'
    # short_window - (int)lookback period for short-term moving average. Eg: 5, 10, 20 
    # long_window - (int)lookback period for long-term moving average. Eg: 50, 100, 200
    # moving_avg - (str)the type of moving average to use ('SMA' or 'EMA')
    # display_table - (bool)whether to display the date and price table at buy/sell positions(True/False)

    # import the closing price data of the stock for the aforementioned period of time in Pandas dataframe
    start = datetime.datetime(*map(int, start_date.split('-')))
    end= datetime.datetime.now()
#     end = datetime.datetime(*map(int, end_date.split('-'))) 
    stock_df = web.DataReader(stock_symbol, 'yahoo', start = start, end = end)['Close']
    stock_df = pd.DataFrame(stock_df) # convert Series object to dataframe 
    stock_df.columns = {'Close Price'} # assign new colun name
    stock_df.dropna(axis = 0, inplace = True) # remove any null rows 
                        
    # column names for long and short moving average columns
    short_window_col = str(short_window) + '_' + moving_avg
    long_window_col = str(long_window) + '_' + moving_avg  
  
    if moving_avg == 'SMA':
        # Create a short simple moving average column
        stock_df[short_window_col] = stock_df['Close Price'].rolling(window = short_window, min_periods = 1).mean()

        # Create a long simple moving average column
        stock_df[long_window_col] = stock_df['Close Price'].rolling(window = long_window, min_periods = 1).mean()

    elif moving_avg == 'EMA':
        # Create short exponential moving average column
        stock_df[short_window_col] = stock_df['Close Price'].ewm(span = short_window, adjust = False).mean()

        # Create a long exponential moving average column
        stock_df[long_window_col] = stock_df['Close Price'].ewm(span = long_window, adjust = False).mean()

    # create a new column 'Signal' such that if faster moving average is greater than slower moving average 
    # then set Signal as 1 else 0.
    stock_df['Signal'] = 0.0  
    stock_df['Signal'] = np.where(stock_df[short_window_col] > stock_df[long_window_col], 1.0, 0.0) 

    # create a new column 'Position' which is a day-to-day difference of the 'Signal' column. 
    stock_df['Position'] = stock_df['Signal'].diff()

    # plot close price, short-term and long-term moving averages
    plt.figure(figsize = (20,10))
    plt.tick_params(axis = 'both', labelsize = 14)
    stock_df['Close Price'].plot(color = 'k', lw = 1, label = 'Close Price')  
    stock_df[short_window_col].plot(color = 'b', lw = 1, label = short_window_col)
    stock_df[long_window_col].plot(color = 'g', lw = 1, label = long_window_col) 

    # plot 'buy' signals
    plt.plot(stock_df[stock_df['Position'] == 1].index, 
            stock_df[short_window_col][stock_df['Position'] == 1], 
            '^', markersize = 15, color = 'g', alpha = 0.7, label = 'buy')

    # plot 'sell' signals
    plt.plot(stock_df[stock_df['Position'] == -1].index, 
            stock_df[short_window_col][stock_df['Position'] == -1], 
            'v', markersize = 15, color = 'r', alpha = 0.7, label = 'sell')
    plt.ylabel('Price in ₹', fontsize = 16 )
    plt.xlabel('Date', fontsize = 16 )
    plt.title(str(stock_symbol) + ' - ' + str(moving_avg) + ' Crossover', fontsize = 20)
    plt.legend()
    plt.grid()
    plt.show()
    
    
    
    if display_table == True:
        df_pos = stock_df[(stock_df['Position'] == 1) | (stock_df['Position'] == -1)]
        df_pos['Position'] = df_pos['Position'].apply(lambda x: 'Buy' if x == 1 else 'Sell')
        print(tabulate(df_pos, headers = 'keys', tablefmt = 'psql'))


# In[13]:


MovingAverageCrossStrategy('INFY.NS', '2020-08-31', 50, 200, 'SMA', display_table = True)


# In[14]:


MovingAverageCrossStrategy('AKASH.NS', '2016-08-31',  50, 200, 'SMA', display_table = True)


# In[16]:


MovingAverageCrossStrategy('ASIANPAINT.NS', '2016-08-31', 50, 200, 'SMA', display_table = True)


# In[18]:


MovingAverageCrossStrategy('AKASH.NS', '2016-08-31', 50, 200, 'EMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('AXISBANK.NS', '2016-08-31', '2022-08-31', 50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('INDIGO.NS', '2016-08-31', '2022-08-31', 50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('ADANIPOWER.NS', '2016-08-31', '2022-08-31', 50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('JETAIRWAYS.NS', '2016-08-31', '2022-09-07', 50, 200, 'SMA', display_table = True)


# In[19]:


MovingAverageCrossStrategy('SCI.NS', '2019-08-31', 50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('HDFCBANK.NS', '2010-08-31', '2022-09-07', 20, 100, 'EMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('BOEING', '2010-08-31', 20, 100, 'EMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('UBER', '2010-08-31', '2022-09-07', 20, 100, 'EMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('UBER', '2016-08-31',  50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('BOEING.NS', '2016-08-31',  50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('UBER', '2016-08-31',  50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('HDFC.NS', '2020-08-31',  50, 200, 'SMA', display_table = True)


# In[ ]:


MovingAverageCrossStrategy('ADANIPOWER.NS', '2016-08-31',  50, 200, 'SMA', display_table = True)


# In[21]:


MovingAverageCrossStrategy('HDFC.NS', '2016-08-31',50, 200, 'EMA', display_table = True)


# In[20]:


MovingAverageCrossStrategy('INFOSYS.NS', '2020-08-31', 50, 200, 'SMA', display_table = True)


# In[ ]:




