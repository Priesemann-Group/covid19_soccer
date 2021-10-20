import requests

urls = {
    "DE.csv": "https://www.arcgis.com/sharing/rest/content/items/f10774f1c63e40168479a1feb6c7ca74/data",
    "FR.csv": "https://www.data.gouv.fr/fr/datasets/r/57d44bd6-c9fd-424f-9a72-7834454f9e3c",
    "GB-ENG.json": "https://api.coronavirus.data.gov.uk/v1/data?filters=areaType=nation;areaName=england&structure={%22date%22:%22date%22,%22name%22:%22areaName%22,%22code%22:%22areaCode%22,%22maleCases%22:%22maleCases%22,%22femaleCases%22:%22femaleCases%22}&format=%22csv%22",
    ### SCT url is day specific, not autoupdating
    "GB-SCT.csv": "https://www.opendata.nhs.scot/dataset/b318bddf-a4dc-4262-971f-0ba329e09b87/resource/9393bd66-5012-4f01-9bc5-e7a10accacf4/download/trend_agesex_20210817.csv",
    "PT.csv": "https://raw.githubusercontent.com/dssg-pt/covid19pt-data/master/data.csv",
    "AT.csv": "https://covid19-dashboard.ages.at/data/CovidFaelle_Altersgruppe.csv",
    "BE.csv": "https://epistat.sciensano.be/Data/COVID19BE_CASES_AGESEX.csv",
    "CZ.csv": "https://onemocneni-aktualne.mzcr.cz/api/v2/covid-19/osoby.csv",
    "DK.zip": "https://files.ssi.dk/covid19/overvagning/dashboard/overvaagningsdata-dashboard-covid19-18102021-77fq",
}


for key in urls.keys():
    r = requests.get(urls[key])
    with open("./case_data_gender_raw/" + key, "wb") as f:
        f.write(r.content)
