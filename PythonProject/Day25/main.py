import csv
import pandas


'''
data=pandas.read_csv("weather_data.csv")
print(data)
print(data["temp"])

temp_list = data["temp"].to_list()


avg_temp = sum(temp_list)/len(temp_list)
max_temp = data["temp"].max()
print(avg_temp)
print(max_temp)
'''

data = pandas.read_csv("2018_Central_Park_Squirrel_Census_-_Squirrel_Data.csv")
grey_squirrels = data[data["Primary Fur Color"] == "Gray"]
grey_squirrels_count = len(grey_squirrels)

print(grey_squirrels)
print(grey_squirrels_count)

data_dict = {
    "Fur Color" : ["Gray","Cinnamon","Black"],
    "Count" : [grey_squirrels_count,red_squirrel_count,black_Squirrel_count]

}