import pandas as pd
import numpy as np
import glob
import os

from datetime import datetime, timedelta
import covid19_inference.data_retrieval as cov19_data

import logging

log = logging.getLogger(__name__)


class Dataloader:
    """
    Easy dataloader class to load our different data files into
    one easy object with properties. This object should not use any
    model parameters such as data_begin, sim_length...
    """

    def __init__(
        self,
        countries=None,
        data_begin=datetime(2021, 6, 1),
        data_end=datetime(2021, 8, 15),
        sim_begin=datetime(2021, 5, 16),
        data_folder=f"{os.path.dirname(__file__)}/../data/",
        offset_games=0,
        offset_data=0,
    ):
        """
        Parameters
        ----------
        countries : list, optional
            List of countries teams we want to use for the analysis.
            Tries to lookup the name. Throws error if name is not found
            in lookup dictionary. If none are give, takes all from the
            lookup database.

        """
        self.data_folder = data_folder

        self.data_begin = data_begin
        self.data_end = data_end
        self.sim_begin = sim_begin
        self.offset_games = offset_games
        self.offset_data = timedelta(days=offset_data)

        # Load country lookup
        self.lookup = pd.read_csv(os.path.join(self.data_folder, "countries.csv"))

        # Default arguments
        if countries is None:
            countries = list(self.lookup["country"].values)
            for rem in [
                "Liechtenstein",
                "Kosovo",
                "Iceland",
                "Malta",
                "Monaco",
                "San Marino",
                # "Azerbaijan",  # because no temperature information
                # "Latvia",  # because no temperature information
                # "Ukraine",  # because no temperature information
                "Andorra",
            ]:
                countries.remove(rem)

        # Check if country exists in database
        for c in countries:
            if not self.lookup.isin([c]).any().any():
                raise ValueError(f"Country '{c}' not in lookup table!")

        # Get aĺl entries from the country/iso2 column corresponding
        # to the picked countries, uses setter below
        self.countries = countries
        self.countries_iso2 = countries

        """
        From here onwards we download/load data.
        """
        # Load timetable
        self.timetable = pd.read_csv(
            os.path.join(self.data_folder, "em_game_data.csv"),
            header=2,
            skipinitialspace=True,
        )
        self.timetable["date"] = pd.to_datetime(
            self.timetable["date"], format="%Y-%m-%d"
        )
        # Shift the date of the games if offset is greater 0
        if offset_games != 0:
            self.timetable["date"] = self.timetable["date"] + timedelta(
                days=offset_games
            )

        jhu = cov19_data.JHU(True)
        jhu.download_all_available_data(force_local=True)
        self._cases = pd.DataFrame()
        self._deaths = pd.DataFrame()
        for c in self.countries:
            self._cases[str(c)] = jhu.get_new(country=c)
            self._deaths[str(c)] = jhu.get_total_confirmed_deaths_recovered(country=c)[
                "deaths"
            ]

        # Load stringency data
        self._stringencyPHSM = None
        self._stringencyOxCGRT = None

        # Load wheather data
        self._load_wheather()

    def _load_wheather(self):
        # Validate weather data
        self._wheather = {}
        for c in self.countries_iso2:
            df = pd.read_csv(
                os.path.join(self.data_folder, "weather_data", f"{c}_weather_data.csv")
            )
            df["Date time"] = pd.to_datetime(df["Date time"], format="%m/%d/%Y")
            df = df.set_index("Date time")

            # Check if time dimension does match up
            if df.shape[0] < self.new_cases_obs.shape[0]:
                log.warning(
                    f"Possible missing values in weather data for [{c}]! \n"
                    f"\tWeather data has length {df.shape[0]} should have "
                    f"{self.new_cases_obs.shape[0]}!"
                )
            self._wheather[c] = df

    def _load_PHSM(self):
        self._stringencyPHSM = []
        temp = pd.read_csv(
            os.path.join(self.data_folder, "Severity index - 2022-03-31.csv")
        )
        temp["Date"] = pd.to_datetime(temp["Date"], format="%d-%b-%y")
        temp = temp.set_index(["Country", "Date"])
        for c in self.countries:
            if c in ["England", "Scotland"]:
                c = "United Kingdom"
            if c == "Czechia":
                c = "Czech Republic"
            self._stringencyPHSM.append(
                temp.loc[c, :]["PHSM SI"].resample("D").fillna(0)
            )

    def _load_OxCGRT(self):
        self._stringencyOxCGRT = []
        ox = cov19_data.OxCGRT(True)
        for c in self.countries:
            if c == "England":
                c = "United Kingdom"
                region = "England"
                temp_data = ox.data[ox.data["country"] == c]
                temp_data = temp_data[temp_data["RegionName"] == region][
                    "StringencyIndex"
                ]
            elif c == "Scotland":
                c = "United Kingdom"
                region = "Scotland"
                temp_data = ox.data[ox.data["country"] == c]
                temp_data = temp_data[temp_data["RegionName"] == region][
                    "StringencyIndex"
                ]
            else:
                if c == "Czechia":
                    c = "Czech Republic"
                elif c == "Slovakia":
                    c = "Slovak Republic"
                temp_data = ox.data[ox.data["country"] == c]["StringencyIndex"]
            self._stringencyOxCGRT.append(temp_data)

    @property
    def countries(self):
        """
        Returns array of used countries, same order as for every other data tensor.

        shape : (country)
        """
        return self._countries

    @countries.setter
    def countries(self, countries):
        self._countries = self.lookup["country"][
            self.lookup.isin(countries).any(axis=1)
        ].to_numpy()

    @property
    def countries_iso2(self):
        """
        Returns arry of used countries, same order as for every other data tensor.

        shape : (country)
        """
        return self._countries_iso2

    @countries_iso2.setter
    def countries_iso2(self, countries):
        self._countries_iso2 = self.lookup["iso2"][
            self.lookup.isin(countries).any(axis=1)
        ].to_numpy()

    @property
    def population(self):
        """
        Returns population per country

        shape : (country)
        """
        population = []
        for c in self.countries:
            pop = self.lookup["population"][self.lookup.isin([c]).any(axis=1)].values[0]
            population.append(pop)
        return np.array(population)

    @property
    def date_of_games(self):
        """
        Returns array of all game dates as datetime objects. Sometime called t_g
        in the model.

        shape : (game)
        """
        return self.timetable["date"]

    @property
    def game2phase(self):
        """
        Returns game to phase matrix.

        shape : (game, phase)
        """

        ind, phases = pd.factorize(self.timetable["phase"])

        ret = []
        for p in range(len(np.unique(ind))):
            ret_g = []
            for g in range(len(ind)):
                if ind[g] == p:
                    ret_g.append(1)
                else:
                    ret_g.append(0)
            ret.append(ret_g)

        return np.array(ret).T

    @property
    def new_cases_obs(self):
        """
        Returns daily confirmed cases per country (dataframe with datetime index).

        shape : (time, country)
        """

        return self._cases[
            self.data_begin + self.offset_data : self.data_end + self.offset_data
        ]

    @property
    def total_deaths(self):
        """
        Returns total confirmed deaths per country (dataframe with datetime index).

        shape : (time, country)
        """

        return self._deaths[
            self.data_begin + self.offset_data : self.data_end + self.offset_data
        ]

    @property
    def new_cases_obs_last_year(self):
        """
        Returns daily confirmed cases per country (dataframe with datetime index).

        shape : (time, country)
        """

        return self._cases[
            self.data_begin - timedelta(weeks=51) : self.data_end - timedelta(weeks=51)
        ]

    @property
    def alpha_prior(self):
        """
        The alpha prior matrix encodes the prior expectation of the effect of a game on the reproduction number.

        shape : (country, game)
        """

        # Easy implementation of tensor:
        #    - 0 if game is played by different team
        #    - 1 if game is played by own team

        temp = []
        for g, game in self.timetable.iterrows():
            temp_g = []
            for c, country in enumerate(self.countries_iso2):
                if game["team1"] == country or game["team2"] == country:
                    temp_g.append(1.0)
                else:
                    temp_g.append(0)
            temp.append(temp_g)

        if hasattr(self, "_alpha_prior"):
            return self._alpha_prior

        return np.array(temp).T

    @alpha_prior.setter
    def alpha_prior(self, value):
        self._alpha_prior = value

    @property
    def weighted_alpha_prior(self):
        """
        The alpha prior matrix encodes the prior expectation of the effect of a game on the reproduction number.

        shape : (country, game)
        """

        # Easy implementation of tensor:
        #    - 0 if game is played by different team
        #    - 1 if game is played by own team

        temp = []
        for g, game in self.timetable.iterrows():
            temp_g = []
            for c, country in enumerate(self.countries_iso2):
                if game["team1"] == country or game["team2"] == country:
                    if game["team1"] == "GB-ENG" and game["team2"] == "GB-SCT":
                        temp_g.append(2)
                    elif game["phase"] == "FI":
                        temp_g.append(1.5)
                    elif game["phase"] == "HF":
                        temp_g.append(1.3)
                    elif game["phase"] == "VF":
                        temp_g.append(1.2)
                    else:
                        temp_g.append(1.0)
                else:
                    temp_g.append(0)
            temp.append(temp_g)

        return np.array(temp).T

    @property
    def beta_prior(self):
        """
        The beta prior matrix encodes where a game took place and eventually could
        encode whether effective hygiene concepts were implemented during the game.

        shape : (country, game)
        """

        # Easy implementation of tensor:
        #    - 0 if game is not played in local stadium
        #    - 1 if game is played in local stadium

        locations = pd.read_csv(
            os.path.join(self.data_folder, "em_locations.csv"), header=7
        )

        temp = []
        for g, game in self.timetable.iterrows():
            temp_g = []
            location_id = locations["name"] == game["location"]
            if np.any(location_id):
                country_of_stadium = locations["country"][location_id].values[0]
                for c, country in enumerate(self.countries_iso2):
                    # country can also be a region, therefore take only 3 first letters
                    if (
                        country_of_stadium in country[:3]
                        or country == country_of_stadium
                    ):
                        temp_g.append(1)
                    else:
                        temp_g.append(0)
            else:
                temp_g.append(0)
            temp.append(temp_g)

        return np.array(temp).T

    @property
    def stadium_size(self):
        """
        The stadium size per country, zero if the country does not have a stadium.
        Sometimes also called S_c in the manuscript and the model.

        shape : (country)
        """
        locations = pd.read_csv(
            os.path.join(self.data_folder, "em_locations.csv"), header=7
        )
        S_c = []
        for c, country in enumerate(self.countries_iso2):
            capacity = locations[locations["country"] == country][
                "capacity"
            ]  # returns pd series
            if capacity.empty:
                S_c.append(0)
            else:
                S_c.append(capacity.values[0])
        return np.array(S_c)

    @property
    def temperature(self):
        """
        Returns temperature of each country per day.


        shape : (time, country)
        """
        temp = []
        len_data = (self.data_end - self.sim_begin).days + 1
        for c in self.countries_iso2:
            df = pd.read_csv(
                os.path.join(self.data_folder, "weather_data", f"{c}_weather_data.csv")
            )
            df["Date time"] = pd.to_datetime(df["Date time"], format="%m/%d/%Y")
            df = df.set_index("Date time")
            df = df["Temperature"][self.sim_begin : self.data_end]
            last_day_in_data = df.index[-1]
            num_days_missing_at_end = (self.data_end - df.index[-1]).days
            if num_days_missing_at_end > 0:
                add_temps = np.array([df[-1] for _ in range(num_days_missing_at_end)])
                log.info(
                    f"Appending {num_days_missing_at_end} days of constant temperature for country {c}"
                )
            else:
                add_temps = []
            temp_for_list = np.concatenate([df.to_numpy(), add_temps])
            temp.append(temp_for_list)

            if not len(temp_for_list) == len_data:
                missing_days = set(
                    pd.to_datetime(
                        [
                            self.data_begin + timedelta(days=i) + self.offset_data
                            for i in range(len_data - num_days_missing_at_end)
                        ]
                    )
                ) - set(df.index)
                log.warning(
                    f"Temperature missing for {c}: ",
                    *[f" {m.strftime('%Y-%m-%d')}" for m in missing_days],
                )

        # Convert to celsius... IDK why would a wheather service use
        # fahrenheit. Seems very unscientific
        temp_c = (np.array(temp).T - 32) / 1.8
        assert temp_c.shape[0] == len_data
        return temp_c

    @property
    def temperature_last_year(self):
        """
        Returns the temperature of each country per day for last year.
        """
        temp = []
        for c in self.countries_iso2:
            df = pd.read_csv(
                os.path.join(
                    self.data_folder, "weather_data", f"{c}_2020-06-01to2020-07-01.csv",
                )
            )
            df["Date time"] = pd.to_datetime(df["Date time"], format="%m/%d/%Y")
            df = df.set_index("Date time")
            temp.append(
                df["Temperature"][
                    self.sim_begin
                    - timedelta(weeks=51) : self.data_end
                    - timedelta(weeks=51)
                ].to_numpy()
            )

        # Convert to celsius... IDK why would a wheather service use
        # fahrenheit. Seems very unscientific
        temp_c = (np.array(temp) - 32) / 1.8

        return temp_c.T

    @property
    def stringencyPHSM(self):
        """
        Returns the stringency index for each country
        """
        if self._stringencyPHSM is None:
            self._load_PHSM()
        ret = []
        for stringency in self._stringencyPHSM:
            ret.append(
                stringency.loc[
                    self.data_begin
                    + self.offset_data : self.data_end
                    + self.offset_data
                ]
            )

        return ret

    @property
    def stringencyOxCGRT(self):
        """
        Returns the stringency index for each country
        """
        if self._stringencyOxCGRT is None:
            self._load_OxCGRT()

        ret = []
        for stringency in self._stringencyOxCGRT:
            ret.append(
                stringency.loc[
                    self.data_begin
                    + self.offset_data : self.data_end
                    + self.offset_data
                ]
            )

        return ret

    @property
    def nRegions(self):
        """
        Number of countries/regions
        """
        return len(self.countries_iso2)

    @property
    def nPhases(self):
        """['Scotland']

        Number of phases in the tournament.
        """
        return self.game2phase.shape[1]

    @property
    def nGames(self):
        """
        Number of phases in the tournament.
        """
        return self.game2phase.shape[0]

    # For nice print in console
    def __repr__(self):
        return f"Dataloader for the countries: {self.countries}"

    def __str__(self):
        return f"Dataloader for the countries: {self.countries}"


class Dataloader_gender(Dataloader):
    """
    Easy dataloader class to load our different data files into
    one easy object with properties. This object should not use any
    model parameters such as data_begin, sim_length...
    """

    def __init__(
        self,
        countries=None,
        data_begin=datetime(2021, 6, 1),
        data_end=datetime(2021, 8, 15),
        sim_begin=datetime(2021, 5, 16),
        data_folder=f"{os.path.dirname(__file__)}/../data/",
        offset_games=0,
        offset_data=0,
        old_timetable=False,
    ):
        """
        Parameters
        ----------
        countries : list, optional
            List of countries teams we want to use for the analysis.
            Tries to lookup the name. Throws error if name is not found
            in lookup dictionary. If none are give, takes all from the
            lookup database.

        """
        self.data_folder = data_folder

        self.data_begin = data_begin
        self.data_end = data_end
        self.sim_begin = sim_begin
        self.offset_games = offset_games
        self.offset_data = timedelta(days=offset_data)

        # Load country lookup
        self.lookup = pd.read_csv(os.path.join(self.data_folder, "countries.csv"))

        # Default arguments
        if countries is None:
            countries = list(
                self.lookup[self.lookup["gender_data"] == "True"]["country"].values
            )

        # Check if country exists in database
        for c in countries:
            if not self.lookup.isin([c]).any().any():
                raise ValueError(f"Country '{c}' not in lookup table!")

        # Get aĺl entries from the country/iso2 column corresponding
        # to the picked countries
        self.countries = countries
        self.countries_iso2 = countries

        """
        From here onwards we download/load data.
        """
        # Load timetable
        self.timetable = pd.read_csv(
            os.path.join(self.data_folder, "em_game_data.csv"),
            header=2,
            skipinitialspace=True,
        )
        if old_timetable:
            self.timetable = self.timetable.iloc[:-7]

        self.timetable["date"] = pd.to_datetime(
            self.timetable["date"], format="%Y-%m-%d"
        )
        if offset_games != 0:
            self.timetable["date"] = self.timetable["date"] + timedelta(
                days=offset_games
            )

        # Load case data from folder "data/case_data_gender"
        self.__load_cases()

        # Load wheather data TODO:fix SCT
        # self._load_wheather()

        # Load stringency data
        self._stringencyPHSM = None
        self._stringencyOxCGRT = None

    def __load_cases(self):

        # Get list of all files
        all_files = glob.glob(self.data_folder + "case_data_gender/*")
        available_countries = [
            os.path.splitext(file)[0].split("/")[-1] for file in all_files
        ]

        # select selected countries from available ones
        iso2 = [
            self.lookup[self.lookup["country"] == c]["iso2"].values[0]
            for c in self.countries
        ]
        intersection = list(filter(lambda x: x in iso2, available_countries))

        # Use setter to update countries
        self.countries = intersection
        self.countries_iso2 = intersection

        # Load preprocessed data
        self._cases = pd.DataFrame()
        self._deaths = pd.DataFrame()
        for i, c in enumerate(self.countries_iso2):
            # select file
            file = self.data_folder + "case_data_gender/" + c + ".csv"
            # load csv
            df = pd.read_csv(file, parse_dates=["date"])
            # set multindex
            if "age_group" not in df.columns:
                df["age_group"] = "total"
            df = df.set_index(["date", "gender", "age_group"])

            # add to dataframe
            self._cases[str(c)] = df["cases"]
            if "deaths" in df.columns:
                self._deaths[str(c)] = df["deaths"]

        # sort index otherwise we cant slice
        self._cases = self._cases.sort_index(level="date")
        self._deaths = self._deaths.sort_index(level="date")

    @property
    def new_cases_obs(self):
        """
        Returns daily confirmed cases per country (dataframe with datetime index).

        shape : (time, gender, country)
        """
        temp = pd.DataFrame()
        temp_m = self._cases.loc[
            self.data_begin + self.offset_data : self.data_end + self.offset_data,
            "male",
            "total",
        ]
        temp_f = self._cases.loc[
            self.data_begin + self.offset_data : self.data_end + self.offset_data,
            "female",
            "total",
        ]

        # Reshape to (date, gender) and male at the front
        return np.stack((temp_m.to_numpy(), temp_f.to_numpy()), axis=1)

    @property
    def population(self):
        """
        Returns population per country

        shape : (country)
        """
        population = []
        for c in self.countries:
            pop = [
                self.lookup["population_male"][
                    self.lookup.isin([c]).any(axis=1)
                ].values[0],
                self.lookup["population_female"][
                    self.lookup.isin([c]).any(axis=1)
                ].values[0],
            ]
            population.append(pop)
        return np.array(population).T

    @property
    def nGenders(self):
        return 2


# Short example/test case
if __name__ == "__main__":
    countries = ["Scotland"]

    dl = Dataloader_gender(countries, offset_games=4 * 7)
    da = Dataloader_gender()

    import matplotlib.pyplot as plt
    import pandas as pd

    plt.plot(
        np.stack([pd.date_range(da.data_begin, da.data_end)] * 15).T,
        da.new_cases_obs[:, 0, :] / da.new_cases_obs[:, 1, :],
    )
