#Download required tools for FinBert

tokenizer = AutoTokenizer.from_pretrained("ProsusAI/finbert")
model = AutoModelForSequenceClassification.from_pretrained("ProsusAI/finbert")

#Creat new columns fro teh scores

data = pd.read_csv("data_TT.csv")
data["positive"] = ""
data["negative"] = ""
data["neutral"] = ""

#Calculate FinBert scores

for i in range (0,len(data)):
    ec=data.loc[i,'transcript_text']
    inputs = tokenizer(ec, padding = True, truncation = True, return_tensors='pt')
    outputs = model(**inputs)
    predictions = torch.nn.functional.softmax(outputs.logits, dim=-1)
    data.loc[i,'positive']= predictions[:, 0].tolist()
    data.loc[i,'negative']= predictions[:, 1].tolist()
    data.loc[i,'neutral']= predictions[:, 2].tolist()  

#Remove some columns 
#data.rename(columns={"Unnamed: 0": "Index"}, inplace=True)
data.drop('Unnamed: 0.1', inplace=True, axis=1)
data.drop('Unnamed: 0', inplace=True, axis=1)
data.columns

####
#   Part Four 
data_T_F =pd.read_csv("/Users/adamrudolfsson/Downloads/True_False.csv",sep=';')

data["True_False"] = ""
data['True_False'] =data_T_F['True_False']

data.to_csv('data_TTN.csv')

csv.writer( )
