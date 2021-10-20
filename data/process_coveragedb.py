# -*- coding: utf-8 -*-
# @Author: Sebastian B. Mohr
# @Date:   2021-10-20 11:31:05
# @Last Modified by:   Sebastian Mohr
# @Last Modified time: 2021-10-20 13:12:48
import pandas as pd


# Before running you need to download and extract the inputDB.zip file
# from: https://osf.io/9dsfk/
df = pd.read_csv("inputDB.csv", header=1)


# Country strings and short versions
countries = [
    ["Italy", "IT"],
    ["Slovakia", "SK"],
    ["Slovenia", "SI"],
    ["Spain", "ES"],
    ["Sweden", "SE"],
    ["Switzerland", "CH"],
    # ["Netherlands", "NL"]
]


for country, short in countries:
    temp = df[df["Country"] == country]
    temp = temp[temp["Measure"] == "Cases"]
    temp = temp[temp["Metric"] == "Count"]
    temp = temp[temp["Region"] == "All"]
    temp["Date"] = pd.to_datetime(temp["Date"], format="%d.%m.%Y")
    temp = temp.drop(
        columns=["AgeInt", "Short", "Code", "Region", "Country", "Metric", "Measure"]
    )

    # Summ all age groups if there is no total column
    if "TOT" in temp["Age"].unique():
        temp = temp[temp["Age"] == "TOT"]
        temp = temp.reset_index()
        temp = temp.drop(columns=["index", "Age"])
    else:
        temp = temp.groupby(["Date", "Sex"]).sum()
        temp = temp.reset_index()

    temp = temp.set_index("Date")

    # Reindex date and fill missing values
    idx = pd.date_range(temp.index.min(), temp.index.max())

    save = []
    for sex in ["m", "f"]:
        t = temp[temp["Sex"] == sex].drop(columns=["Sex"])
        t = t.reindex(idx, method="ffill")
        t = t.diff()
        t["gender"] = ["male" if sex == "m" else "female"] * len(t)
        save.append(t)

    save = pd.concat(save)
    save = save.reset_index()
    save = save.rename(columns={"Value": "cases", "index": "date"})

    save.to_csv(f"./case_data_gender/{short}.csv", index=False)
