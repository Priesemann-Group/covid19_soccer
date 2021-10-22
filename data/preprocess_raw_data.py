"""
Converts every raw case file to one streamlined format

date|gender|age_group|cases|deaths

date: %Y-%m-%d 
gender: male,female
age_group: str, total
cases,deaths: int
"""
import numpy as np
import pandas as pd
from datetime import datetime
import json

raw_data_path = "./case_data_gender_raw/"
processed_data_path = "./case_data_gender/"
data_begin = datetime(2021, 5, 1)
data_end = datetime(2021, 8, 1)


def Germany():

    de = pd.read_csv(raw_data_path + "DE.csv", parse_dates=["Meldedatum"])
    de.drop(
        columns=[
            "FID",
            "IdBundesland",
            "Bundesland",
            "Landkreis",
            "IdLandkreis",
            "Datenstand",
            "Refdatum",
            "IstErkrankungsbeginn",
            "Altersgruppe2",
            "NeuerFall",
            "NeuerTodesfall",
            "NeuGenesen",
            "AnzahlGenesen",
        ],
        inplace=True,
    )
    de.rename(
        columns={
            "Meldedatum": "date",
            "Geschlecht": "gender",
            "Altersgruppe": "age_group",
            "AnzahlFall": "cases",
            "AnzahlTodesfall": "deaths",
        },
        inplace=True,
    )
    de = de.groupby(by=["date", "gender", "age_group"]).sum().reset_index()
    de.gender.replace(to_replace={"M": "male", "W": "female"}, inplace=True)
    de.age_group.replace(
        to_replace={
            "A00-A04": "00-04",
            "A05-A14": "05-14",
            "A15-A34": "15-34",
            "A35-A59": "35-59",
            "A60-A79": "60-79",
            "A80+": "80+",
        },
        inplace=True,
    )
    # calculate the missing totals
    tmp = de.groupby(by=["date", "gender"]).sum().reset_index()
    tmp.insert(2, "age_group", "total")
    de = de.append(tmp)
    tmp = de.groupby(by=["date", "age_group"]).sum().reset_index()
    tmp.insert(1, "gender", "total")
    de = de.append(tmp)
    # drop entries with unknown gender/age, they are included in the totals
    de = de[(de.gender != "unbekannt") & (de.age_group != "unbekannt")]

    de.sort_values(["date", "gender", "age_group"], axis=0, inplace=True)
    de = de[(de.date >= data_begin) & (de.date < data_end)]
    de.to_csv(path_or_buf=processed_data_path + "DE.csv", index=None)


### France
def France():
    fr = pd.read_csv(raw_data_path + "FR.csv", parse_dates=["jour"], sep=";")
    fr.drop(columns=["fra", "pop_f", "pop_h", "pop"], inplace=True)
    fr.rename(columns={"jour": "date", "cl_age90": "age_group"}, inplace=True)
    fr.age_group.replace(
        to_replace={
            9: "00-09",
            19: "10-19",
            29: "20-29",
            39: "30-39",
            49: "40-49",
            59: "50-59",
            69: "60-69",
            79: "70-79",
            89: "80-89",
            90: "90+",
            0: "total",
        },
        inplace=True,
    )
    # male, female and total cases are in the columns P_h, P_f and P
    # splitting this information in gender and cases columns
    fr_f = fr.copy()
    fr_f.insert(1, "gender", ["female"] * len(fr_f.index))
    fr_f.insert(len(fr_f.columns), "cases", fr_f["P_f"])
    fr_m = fr.copy()
    fr_m.insert(1, "gender", ["male"] * len(fr_m.index))
    fr_m.insert(len(fr_m.columns), "cases", fr_m["P_h"])
    fr.insert(1, "gender", ["total"] * len(fr_m.index))
    fr.insert(len(fr.columns), "cases", fr["P"])
    fr = fr.append(fr_f).append(fr_m)
    fr.insert(len(fr.columns), "deaths", [np.nan] * len(fr.index))
    fr.drop(columns=["P_f", "P_h", "P"], inplace=True)

    fr.sort_values(["date", "gender", "age_group"], axis=0, inplace=True)
    fr = fr[(fr.date >= data_begin) & (fr.date < data_end)]
    fr.to_csv(path_or_buf=processed_data_path + "FR.csv", index=None)


### United Kingdom
def GB_ENG():
    gb_json = json.load(open(raw_data_path + "GB-ENG.json"))
    gb = pd.DataFrame(columns=["date", "gender", "age_group", "cases"])
    for data in gb_json["data"]:
        date = datetime.strptime(data["date"], "%Y-%m-%d")
        if date >= data_begin and date < data_end:
            for line in data["femaleCases"]:
                gb.loc[len(gb.index)] = [date, "female", line["age"], line["value"]]
            for line in data["maleCases"]:
                gb.loc[len(gb.index)] = [date, "male", line["age"], line["value"]]

    gb.replace(
        to_replace={
            "0_to_4": "00-04",
            "5_to_9": "05-09",
            "10_to_14": "10-14",
            "15_to_19": "15-19",
            "20_to_24": "20-24",
            "25_to_29": "25-29",
            "30_to_34": "30-34",
            "35_to_39": "35-39",
            "40_to_44": "40-44",
            "45_to_49": "45-49",
            "50_to_54": "50-54",
            "55_to_59": "55-59",
            "60_to_64": "60-64",
            "65_to_69": "65-69",
            "70_to_74": "70-74",
            "75_to_79": "75-79",
            "80_to_84": "80-84",
            "85_to_89": "85-89",
            "90+": "90+",
        },
        inplace=True,
    )
    # calculate the missing totals
    tmp = gb.groupby(by=["date", "gender"]).sum().reset_index()
    tmp.insert(2, "age_group", "total")
    gb = gb.append(tmp)
    tmp = gb.groupby(by=["date", "age_group"]).sum().reset_index()
    tmp.insert(1, "gender", "total")
    gb = gb.append(tmp)

    # Calculate daily change in the infections
    gb = gb.set_index(["date", "gender", "age_group"])
    gb = gb.sort_index()  # is important otherwise the diff does not work
    gb = gb.groupby(level=[1, 2]).diff(1)
    gb.insert(len(gb.columns), "deaths", [np.nan] * len(gb.index))

    gb.to_csv(path_or_buf=processed_data_path + "GB-ENG.csv",)


### Scotland
def Scotland():
    sct = pd.read_csv(raw_data_path + "GB-SCT.csv", parse_dates=["Date"])
    sct.drop(
        columns=[
            "Country",
            "SexQF",
            "AgeGroupQF",
            "CumulativePositive",
            "CrudeRatePositive",
            "CumulativeDeaths",
            "CrudeRateDeaths",
            "CumulativeNegative",
            "CrudeRateNegative",
        ],
        inplace=True,
    )
    sct.rename(
        columns={
            "Date": "date",
            "Sex": "gender",
            "AgeGroup": "age_group",
            "DailyPositive": "cases",
            "DailyDeaths": "deaths",
        },
        inplace=True,
    )
    # drop the additional age groups 0-59 and 60+
    sct = sct[sct.age_group.isin(["0 to 59", "60+"]) == False]
    sct.replace(
        to_replace={
            "Female": "female",
            "Male": "male",
            "Total": "total",
            "0 to 14": "00-14",
            "15 to 19": "15-19",
            "20 to 24": "20-24",
            "25 to 44": "25-44",
            "45 to 64": "45-64",
            "65 to 74": "65-74",
            "75 to 84": "75-84",
            "85plus": "85+",
        },
        inplace=True,
    )
    sct = sct[(sct.date >= data_begin) & (sct.date < data_end)]
    sct.to_csv(path_or_buf=processed_data_path + "GB-SCT.csv", index=None)


def Portugal():
    pt = pd.read_csv(raw_data_path + "PT.csv")
    pt["data"] = pd.to_datetime(pt["data"], format="%d-%m-%Y")
    pt.rename(
        columns={
            "data": "date",
            "confirmados_f": "femaleCases",
            "confirmados_m": "maleCases",
            "obitos_f": "femaleDeaths",
            "obitos_m": "maleDeaths",
            "confirmados": "totalCases",
            "obitos": "totalDeaths",
        },
        inplace=True,
    )

    for key in [
        "femaleCases",
        "maleCases",
        "femaleDeaths",
        "maleDeaths",
        "totalCases",
        "totalDeaths",
    ]:
        pt[key] = pt[key].diff()
    df = pd.DataFrame(columns=["date", "gender", "age_group", "cases", "deaths"])
    for i, line in pt.iterrows():
        df.loc[len(df.index)] = [
            line["date"],
            "female",
            "total",
            line["femaleCases"],
            line["femaleDeaths"],
        ]
        df.loc[len(df.index)] = [
            line["date"],
            "male",
            "total",
            line["maleCases"],
            line["maleDeaths"],
        ]
        df.loc[len(df.index)] = [
            line["date"],
            "total",
            "total",
            line["totalCases"],
            line["totalDeaths"],
        ]
    df.to_csv(path_or_buf=processed_data_path + "PT.csv", index=False)


def Austria():
    at = pd.read_csv(raw_data_path + "AT.csv", sep=";")
    at["Time"] = pd.to_datetime(at["Time"], format="%d.%m.%Y 00:00:00")
    at = at.drop(
        columns=[
            "Altersgruppe",
            "Bundesland",
            "AnzEinwohner",
            "BundeslandID",
            "AltersgruppeID",
        ]
    )
    at = at.groupby(["Time", "Geschlecht"]).sum()
    at = at.sort_values(["Geschlecht", "Time"])
    at = at.groupby(["Geschlecht"]).diff()
    at = at.reset_index()

    at["Geschlecht"] = at["Geschlecht"].replace(
        to_replace={"M": "male", "W": "female"},
    )
    at = at.rename(
        columns={
            "Anzahl": "cases",
            "Time": "date",
            "Geschlecht": "gender",
            "AnzahlTot": "deaths",
            "AnzahlGeheilt": "recovered",
        }
    )
    at.to_csv(path_or_buf=processed_data_path + "AT.csv", index=False)


def Belgium():
    df = pd.read_csv(raw_data_path + "BE.csv")
    df["DATE"] = pd.to_datetime(df["DATE"], format="%Y-%m-%d")
    df.drop(columns=["PROVINCE", "REGION", "AGEGROUP"])
    df = df.groupby(["DATE", "SEX"]).sum()
    df = df.sort_values(["SEX", "DATE"])
    df = df.reset_index()
    df["SEX"] = df["SEX"].replace(to_replace={"M": "male", "F": "female"},)
    df = df.rename(columns={"CASES": "cases", "DATE": "date", "SEX": "gender",})

    df.to_csv(path_or_buf=processed_data_path + "BE.csv", index=False)


def Czech():
    df = pd.read_csv(raw_data_path + "CZ.csv")
    df["datum"] = pd.to_datetime(df["datum"], format="%Y-%m-%d")
    df = df.rename(columns={"datum": "date", "pohlavi": "gender", "vek": "age"})
    df["cases"] = 1
    df = df.drop(
        columns=[
            "age",
            "kraj_nuts_kod",
            "okres_lau_kod",
            "nakaza_v_zahranici",
            "nakaza_zeme_csu_kod",
        ]
    )
    df = df.groupby(["date", "gender"]).count()
    df = df.reset_index()
    df["gender"] = df["gender"].replace(to_replace={"M": "male", "Z": "female"},)
    df = df.sort_values(["gender", "date"])

    df.to_csv(path_or_buf=processed_data_path + "CZ.csv", index=False)


def Netherlands():
    df = pd.read_csv(raw_data_path + "NL.csv", sep=";")

    # Select onset of symptoms only
    df = df[df["Date_statistics_type"] == "DOO"]
    df["Date_statistics"] = pd.to_datetime(df["Date_statistics"], format="%Y-%m-%d")

    df = df.drop(
        columns=[
            "Date_file",
            "Date_statistics_type",
            "Agegroup",
            "Province",
            "Hospital_admission",
            "Deceased",
            "Week_of_death",
            "Municipal_health_service",
        ]
    )
    df = df.rename(columns={"Date_statistics": "date", "Sex": "gender"})
    df["cases"] = 1
    mi = min(df["date"])
    ma = max(df["date"])
    df = df.groupby(["date", "gender"]).count()
    idx = pd.date_range(mi, ma)
    df = df.reindex(
        pd.MultiIndex.from_product([idx, ["Female", "Male"]], names=["date", "gender"]),
        fill_value=0,
    )
    df = df.reset_index()
    df["gender"] = df["gender"].replace(
        to_replace={"Male": "male", "Female": "female"},
    )
    df.to_csv(path_or_buf=processed_data_path + "NL.csv", index=False)
