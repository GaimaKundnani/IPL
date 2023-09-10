import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#Bowling
df=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Bowling Stats/BOWLING STATS - IPL_2016.csv")
df2=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Bowling Stats/BOWLING STATS - IPL_2017.csv")
df3=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Bowling Stats/BOWLING STATS - IPL_2018.csv")
df4=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Bowling Stats/BOWLING STATS - IPL_2019.csv")
df5=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Bowling Stats/BOWLING STATS - IPL_2020.csv")
df6=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Bowling Stats/BOWLING STATS - IPL_2021.csv")
df7=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Bowling Stats/BOWLING STATS - IPL_2022.csv")
df15=pd.concat([df,df2,df3,df4,df5,df6,df7], axis=0, ignore_index=False)
#Batting
df8=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Batting Stats/BATTING STATS - IPL_2016.csv")
df9=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Batting Stats/BATTING STATS - IPL_2017.csv")
df10=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Batting Stats/BATTING STATS - IPL_2018.csv")
df11=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Batting Stats/BATTING STATS - IPL_2019.csv")
df12=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Batting Stats/BATTING STATS - IPL_2020.csv")
df13=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Batting Stats/BATTING STATS - IPL_2021.csv")
df14=pd.read_csv(r"/Users/Garima/Desktop/Python/Project-4/IPL Player Stats/Batting Stats/BATTING STATS - IPL_2022 2.csv")
df16=pd.concat([df8,df9,df10,df11,df12,df13,df14], axis=0, ignore_index=False)
print("Bowling:")
print(df15)
print("Batting:")
print(df16)

#Print the first few rows
print("First rows of Bowling:")
print(df15.head())
print("First rows of Batting:")
print(df16.head())

#Check the dimensions of the dataset
print("Formating of Bowlers dataset:")
print(df15.describe())
print(df15.info())
print("Formatting of Batting dataset:")
print(df16.describe())
print(df16.info())

#Identify the different variables/columns available in the dataset
print("The different variables/columns available in the bowling dataset are")
print(df15.columns)
print("The different variables/columns available in the batting dataset are")
print(df16.columns)

#Handle missing values appropriately
print("Missing values in Bowling:")
print(df15.isnull())
print("Missing values in Batting:")
print(df16.isnull())

#Data types if required
print("Clean Batting Data:")
df16["Avg"]=df16["Avg"].astype(str)
df16["Avg"]=df16['Avg'].str.replace("-", "0")
df16["Avg"]=df16['Avg'].str.strip("")
df16["Avg"]=df16['Avg'].astype('float64')
df16["HS"] = df16["HS"].str.replace("*", "")
print(df16.drop(["POS"], axis=1, inplace=True))
print(df16)
print("Clean Bowling Data:")
print(df15.drop(['POS'], axis=1, inplace=True))
print(df15.drop(['BBI'], axis=1, inplace=True))
print(df15)

#Perform Analysis on the dataset
print("Batting:",df16.corr()['Runs'].sort_values(ascending=False))
print("Bowlers:",df15.corr()['Wkts'].sort_values(ascending=False))
# Key performance metrics that are relevant for each player position
print("Batting Strike Rate:",np.multiply(np.divide(df16[["Runs"]],df16[["BF"]]),100))
print("Bowling Strike Rate:",np.divide(df15[["Inns"]],df15[["Wkts"]]))
print("Batting Average:",np.divide(df16[["Runs"]],df16[["NO"]]))
print("Bowling Economy Rate:",np.divide(df15[["Runs"]],df15[["Wkts"]]))

#"Great" batsmen in terms of T20 Average and Strike Rate
print("'Great' batsmen in terms of T20 Average and Strike Rate")
print(df16[(df16['Avg'] >= 35) & (df16['SR'] >= 130)])
# "Exceptional" batsmen in terms of T20 Average and Strike Rate
print("'Exceptional' batsmen in terms of T20 Average and Strike Rate")
print(df16[(df16['Avg'] >= 40) & (df16['SR'] >= 140)])

#"Great" bowlers in terms of T20 Average and Strike Rate
print("'Great' bowlers in terms of T20 Average and Strike Rate")
print(df15[(df15['Avg'] >= 35) & (df15['SR'] >= 130)])
# "Exceptional" bowlers in terms of T20 Average and Strike Rate
print("'Exceptional' bowlers in terms of T20 Average and Strike Rate")
print(df15[(df15['Avg'] >= 40) & (df15['SR'] >= 140)])
#Total 50s scored by a single player
d=df16.sort_values(by="50",ascending=False)[:20].copy()
d["custom_label"]=d["Player"]+"\n"+d["50"].astype(str)
plt.pie(d["50"],labels=d["custom_label"],radius=1.4, labeldistance=1.1,pctdistance=0.8)
plt.title("Total 50s scored by a single player")
plt.show()
#Total 100s 
e=df16.sort_values(by="100",ascending=False)[:20].copy()
e["custom_label"]=e["Player"]+"\n"+e["100"].astype(str)
plt.pie(d["100"],labels=e["custom_label"],radius=1.4, labeldistance=1.1,pctdistance=0.8)
plt.title("Total 100s scored by a single player")
plt.show() 

#Histograms for Batsman
plt.hist(df16[["Runs"]])
plt.title("Runs Scored by Batsman")
plt.show()

plt.hist(df16[["SR"]])
plt.title("Strike Rate of  Batsman")
plt.show()

plt.hist(df16[["50"]])
plt.title("No of the times 50 scored by Batsman")
plt.show()

plt.hist(df16[["100"]])
plt.title("No of the times 100 scored by Batsman")
plt.show()

plt.hist(df16[["4s"]])
plt.title("Total Fours Scored by Batsman")
plt.show()

plt.hist(df16[["6s"]])
plt.title("Total Sixes Scored by Batsman")
plt.show()

#Histgram Bowlers
plt.hist(df15[["Wkts"]])
plt.title("Total Wickets taken Bowlers")
plt.show()

plt.hist(df15[["SR"]])
plt.title("Strike Rate Of Bowlers")
plt.show()

plt.hist(df15[["Econ"]])
plt.title(" Bowlers Economy")
plt.show()

plt.hist(df15[["4w"]])
plt.title("4 wickets haul of Bowlers")
plt.show()

plt.hist(df15[["5w"]])
plt.title("5 wickets haul of Bowlers")
plt.show()

#Batsman
df_top_25_batsmen = df16.sort_values(by='Runs', ascending=False)[:25].copy()
df_top_25_batsmen["Runs_In_Boundaries"] =  (df_top_25_batsmen["4s"] * 4) + (df_top_25_batsmen["6s"] * 6)
df_top_25_batsmen["Boundary_Percentage"] = round((df_top_25_batsmen["Runs_In_Boundaries"] / df_top_25_batsmen["Runs"])*100,2)
df_top_25_batsmen["Balls_Per_Innings"] = round(df_top_25_batsmen["BF"] / df_top_25_batsmen["Inns"], 0)
print("The Ideal Percent of total runs scored in boundaries for an aspiring player to aim for is: " + str(df_top_25_batsmen["Boundary_Percentage"].mean()) + "%.")
print("The Ideal Strike Rate a T20 Batsman should aim for is: " + str(round(df_top_25_batsmen["SR"].mean(),2)) + ".")
print("The Ideal Average a T20 Batsman should aim for is: " + str(round(df_top_25_batsmen["Avg"].mean(),2)) + ".")
print("The Average Balls Faced per Innings Played for a T20 Batsman is: " + str(df_top_25_batsmen["Balls_Per_Innings"].mean()) + ".")
print()
#Bowlers
df_top_25_bowlers = df15.sort_values(by="Wkts", ascending=False)[:25].copy()
df_top_25_bowlers["Overs_Per_Inning"] = round(df_top_25_bowlers["Ov"] / df_top_25_bowlers["Inns"], 0)
print("A top T20 Bowler only allows: " + str(df_top_25_bowlers["Econ"].mean()) + " runs per over.")
print("The Number of Balls it takes for a top T20 Bowler to take a wicket is: " + str(round(df_top_25_bowlers["SR"].mean(),0)) + ".")
print("The Amount of Runs allowed per each wicket taken for a top T20 Bowler is: " + str(round(df_top_25_bowlers["Avg"].mean(),2)) + ".")
print("The Average Overs Bowled per Innings Played for a top T20 Bowler is: " + str(round(df_top_25_bowlers["Overs_Per_Inning"].mean(),1)) + ".")
