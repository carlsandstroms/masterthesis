#Remove missing values 

import pandas as pd

#Import the scraped data / Dowloaded in this case 
data_T =pd.read_csv("/~/downloads/data_earningcalls.csv")
data_S=pd.read_csv("/~/downloads/data_stock_closing.csv")
data_I =pd.read_csv("/~/downloads/data_SP500_index.csv")

#Concat all data to remove nulls:
(pd.concat([data_T, data_S, data_I], axis=1)
  .to_csv('all_data.csv', index=False, na_rep='N/A')
)

(pd.concat([data_S, data_I], axis=1)
  .to_csv('Stock_Index.csv', index=False, na_rep='N/A')
)


#Drop observations with nulls 
test_for_null = pd.read_csv('all_data.csv')
SI_data =      pd.read_csv('Stock_Index.csv')

data_T= data_T[test_for_null.isna().any(axis=1)==False]
data_SI= SI_data[test_for_null.isna().any(axis=1)==False]


#Rename the count to a tag so we can check for no probelms:

data_T.rename(columns={"Unnamed: 0": "Tag"}, inplace=True)
data_SI.rename(columns={"Unnamed: 0": "Tag"}, inplace=True)

#Resave the data to CSV and Excel, 

#Transcripts
data_T.to_csv('data_T.csv')

#Stock closing price and index
data_SI.to_excel("data_SI.xlsx",sheet_name="Testing", index=False )

#Check length
print(len(data_T))
print(len(data_SI))
