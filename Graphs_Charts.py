import os
import numpy as np
import pandas as pd
import scipy.stats
from sklearn.linear_model import LinearRegression as lr
import datetime as dt
import matplotlib.pyplot as plt


class GraphsCharts:
    def __init__(self, data):
        self.df = data
        self.locales = ['Outdoors', 'Aboveground\n Platform', 'Underground\n Platform', 'Train Car']
        self.output_path = r"Outputs"
        self.outcal = np.nan
        self.outraw = np.nan

        self.tcarcal = np.nan
        self.tcarraw = np.nan
        self.platcal = np.nan
        self.platraw = np.nan
        self.outplatcal = np.nan
        self.outplatraw = np.nan

        self.outair = np.nan
        self.tcarair = np.nan
        self.platair = np.nan
        self.outplatair = np.nan
        self.outrawair = np.nan
        self.tcarrawair = np.nan
        self.platrawair = np.nan
        self.outplatrawair = np.nan

        self.outstd = np.nan
        self.tcarstd = np.nan
        self.platstd = np.nan
        self.outplatstd = np.nan
        self.outrawstd = np.nan
        self.tcarrawstd = np.nan
        self.platrawstd = np.nan
        self.outplatrawstd = np.nan

        self.outair_err = np.nan
        self.tcarair_err = np.nan
        self.platair_err = np.nan
        self.outplatair_err = np.nan
        self.outrawair_err = np.nan
        self.tcarrawair_err = np.nan
        self.platrawair_err = np.nan
        self.outplatrawair_err = np.nan

# self.df = pd.read_csv('data/Masterlist.csv', header=0, parse_dates=['Date_Time_NYC'])
    def value_gen(self):
        # Fields: [Date_Time_NYC, Temp, Humidity_%, PM2.5, PM2.5_calib, PA_part, Temtop, Neph, Location,
        #         Train_Line, Train_num, Loc_code, Initials]

        # master lists:
        calmaster = self.df[['PM2.5_calib', 'Loc_code']]
        calmaster = calmaster.loc[calmaster['PM2.5_calib'].notnull()]

        rawmaster = self.df[['PM2.5', 'Loc_code']]
        rawmaster = rawmaster.loc[rawmaster['PM2.5'].notnull()]

        # segregate by loc code and raw/calibrated
        self.outcal = calmaster['PM2.5_calib'].loc[calmaster['Loc_code'] == 'O']
        self.outraw = rawmaster['PM2.5'].loc[rawmaster['Loc_code'] == 'O']

        self.tcarcal = calmaster['PM2.5_calib'].loc[calmaster['Loc_code'] == 'T']
        self.tcarraw = rawmaster['PM2.5'].loc[rawmaster['Loc_code'] == 'T']

        self.platcal = calmaster['PM2.5_calib'].loc[calmaster['Loc_code'] == 'PU']
        self.platraw = rawmaster['PM2.5'].loc[rawmaster['Loc_code'] == 'PU']

        self.outplatcal = calmaster['PM2.5_calib'].loc[calmaster['Loc_code'] == 'PA']
        self.outplatraw = rawmaster['PM2.5'].loc[rawmaster['Loc_code'] == 'PA']

        # means
        self.outair = np.mean(self.outcal)
        self.tcarair = np.mean(self.tcarcal)
        self.platair = np.mean(self.platcal)
        self.outplatair = np.mean(self.outplatcal)

        self.outrawair = np.mean(self.outraw)
        self.tcarrawair = np.mean(self.tcarraw)
        self.platrawair = np.mean(self.platraw)
        self.outplatrawair = np.mean(self.outplatraw)

        # stdev
        self.outstd = np.std(self.outcal)
        self.tcarstd = np.std(self.tcarcal)
        self.platstd = np.std(self.platcal)
        self.outplatstd = np.std(self.outplatcal)

        self.outrawstd = np.std(self.outraw)
        self.tcarrawstd = np.std(self.tcarraw)
        self.platrawstd = np.std(self.platraw)
        self.outplatrawstd = np.std(self.outplatraw)

        # standard errors
        def stderr(data):
            return np.std(data) / data.size

        self.outair_err = stderr(self.outcal)
        self.tcarair_err = stderr(self.tcarcal)
        self.platair_err = stderr(self.platcal)
        self.outplatair_err = stderr(self.outplatcal)

        self.outrawair_err = stderr(self.outraw)
        self.tcarrawair_err = stderr(self.tcarraw)
        self.platrawair_err = stderr(self.platraw)
        self.outplatrawair_err = stderr(self.outplatraw)

    def run_boxplots(self):
        # Location Comparison: Boxplot_calibrated
        fig, ax = plt.subplots()
        cal_box = [self.outcal, self.outplatcal, self.platcal, self.tcarcal]
        ax.boxplot(cal_box, labels=self.locales)
        ax.set_ylim(bottom=0)
        ax.set(title='Calibrated Data', ylabel='PM2.5 Concentration (µg/m^3)')

        filename = os.path.join(self.output_path, "Boxplots_Calibrated.png")
        fig.savefig(filename)

        # Location Comparison: Boxplot_raw
        raw_box = [self.outraw, self.outplatraw, self.platraw, self.tcarraw]
        fig, ax = plt.subplots()
        ax.boxplot(raw_box, labels=self.locales)
        ax.set_ylim(bottom=0)
        ax.set(title='Raw Data', ylabel='PM2.5 Concentration (µg/m^3)')

        filename = os.path.join(self.output_path, "Boxplots_Raw.png")
        fig.savefig(filename)

    def run_barplots(self):
        # Location Comparison: barplot (raw data)
        fig, ax = plt.subplots()
        vals_raw = [self.outrawair, self.outplatrawair, self.platrawair, self.tcarrawair]
        err_raw = [self.outrawair_err, self.outplatrawair_err, self.platrawair_err, self.tcarrawair_err]
        std_raw = [self.outrawstd, self.outplatrawstd, self.platrawstd, self.tcarrawstd]

        vals_cal = [self.outair, self.outplatair, self.platair, self.tcarair]
        err_cal = [self.outair_err, self.outplatair_err, self.platair_err, self.tcarair_err]
        std_cal = [self.outstd, self.outplatstd, self.platstd, self.tcarstd]

        colors = ['cornflowerblue', 'lightsteelblue', 'darkgoldenrod', 'silver']

        ax.bar(self.locales, vals_raw, color=colors)
        ax.errorbar(self.locales, vals_raw, yerr=std_raw, fmt="o", linewidth=1, color='dimgrey')
        ax.set_ylim(bottom=0)
        ax.set(title='Raw PA Data',
               ylabel='Avg PM2.5 Concentration (µg/m^3)')
        filename = os.path.join(self.output_path, "Barplot_Raw.png")
        fig.savefig(filename)

        # Location Comparison:barplot calibrated
        colors = ['cornflowerblue', 'lightsteelblue', 'darkgoldenrod', 'silver']
        fig, ax = plt.subplots()
        ax.bar(self.locales, vals_cal, color=colors)
        ax.errorbar(self.locales, vals_cal, yerr=std_cal, fmt="o", linewidth=1, color='dimgrey')

        ax.set(title='Calibrated PA Data',
               ylabel='Avg PM2.5 Concentration (µg/m^3)')
        filename = os.path.join(self.output_path, "Barplot_Calibrated.png")
        fig.savefig(filename)

        # Location Comparison: barplot figures & labels:
        colors = ['cornflowerblue', 'lightsteelblue', 'darkgoldenrod', 'silver']
        fig, ax = plt.subplots()
        ax.bar(self.locales, vals_cal, color=colors)
        ax.errorbar(self.locales, vals_cal, yerr=err_cal, fmt="o", color="black")

        ax.axhline(35, color='black', linewidth=2.5, linestyle='dashed')
        ax.text(2, 41.4, "EPA standard: 35 µg/m^3", bbox=dict(facecolor='white', alpha=1), **dict(size=11, color='black'))

        ax.text(-0.34, vals_cal[0] + 6, str(round(vals_cal[0], 2)) + " ± " + str(round(err_cal[0], 2)),
                bbox=dict(facecolor='white', alpha=1), **dict(size=11))
        ax.text(1 - 0.4, vals_cal[1] + 6, str(round(vals_cal[1], 2)) + " ± " + str(round(err_cal[1], 2)),
                bbox=dict(facecolor='white', alpha=1), **dict(size=11))
        ax.text(2 - 0.43, vals_cal[2] + 6, str(round(vals_cal[2], 2)) + " ± " + str(round(err_cal[2], 2)),
                bbox=dict(facecolor='white', alpha=1), **dict(size=11))
        ax.text(3 - 0.43, vals_cal[3] + 6, str(round(vals_cal[3], 2)) + " ± " + str(round(err_cal[3], 2)),
                bbox=dict(facecolor='white', alpha=1), **dict(size=11))
        ax.set(title='Average PM2.5 Levels Recorded vs. EPA Standard',
               ylabel='Avg PM2.5 Concentration (µg/m^3)')
        ax.set_ylim(top=150)
        filename = os.path.join(self.output_path, "Barplot_Calibrated_Labeled.png")
        fig.savefig(filename)

        # Trainline Comparison
        trains = np.sort(self.df['Train_Line'].loc[self.df['Loc_code'].isin(['T'])].unique())

        train_means = {}
        for s in trains:
            train_means[s] = np.mean(self.df['PM2.5_calib'].loc[self.df['Train_Line'] == s])

        train_std = {}
        for t in trains:
            train_std[t] = np.std(self.df['PM2.5_calib'].loc[self.df['Train_Line'] == t])

        fig, ax = plt.subplots()
        train_means['All'] = np.mean(self.df['PM2.5_calib'].loc[self.df['Loc_code'] == 'T'])
        train_std['All'] = np.std(self.df['PM2.5_calib'].loc[self.df['Loc_code'] == 'T'])
        colors = ['red', 'red', 'red', 'purple', 'tab:blue', 'orange', 'tab:blue', 'orange',
                  'tab:blue', 'orange', 'darkgrey', 'gold', 'tab:blue', 'gold', 'black']

        ax.bar(*zip(*train_means.items()), color=colors)
        ax.errorbar(*zip(*train_means.items()), list(train_std.values()), fmt="o", linewidth=1, color='dimgrey')
        # ax.axhline(12, linewidth = 2.5, linestyle = 'dashed',color = 'dimgrey')
        ax.axhline(35, linewidth=2.5, linestyle='dashed', color='dimgrey')

        # ax.text(0-0.8,22,"12 µg/m^3",bbox=dict(facecolor='snow', alpha=0.9),**dict(size=9,color='black'))
        ax.text(0 - 0.8, 20, "35 µg/m^3", bbox=dict(facecolor='snow', alpha=0.9), **dict(size=9, color='black'))

        # ax.set_xticklabels([])
        # ax.set_yticklabels([])
        ax.set_ylim(bottom=0, top=250)
        ax.set(title="Air Quality Between Train Lines (Means & Standard Deviation)",
               xlabel="Train Line", ylabel="Mean PM2.5 Concentration (µg/m^3)")
        filename = os.path.join(self.output_path, "TrainLine_Comparison.png")
        fig.savefig(filename)

    def run_histograms(self):
        stations = self.df['Location'].loc[self.df['Loc_code'].isin(['PU', 'PA'])].unique()

        station_means = {}
        for s in stations:
            station_means[s] = np.mean(self.df['PM2.5_calib'].loc[self.df['Location'] == s])

        station_std = {}
        for t in stations:
            station_std[t] = np.std(self.df['PM2.5_calib'].loc[self.df['Location'] == t])
        # self.df['Location'].loc[self.df['Loc_code'].isin(['PU','PA'])].value_counts()

        # Example: 116th st station (1 line)
        fig, ax = plt.subplots()
        ax.hist(self.df['PM2.5_calib'].loc[self.df['Location'] == "116th St - Columbia University"], color='red', bins=35)
        ax.axvline(35, color='black', linestyle='dashed')
        ax.text(37, 6, "35 µg/m^3", bbox=dict(facecolor='white', alpha=0.8), **dict(size=11, color='black'))
        # ax.set_xlim(right = 400)
        ax.set(title='PM2.5 Samples for 116th St - Columbia University',
               xlabel='PM2.5 Concentration (µg/m^3)', ylabel='No. Samples')
        filename = os.path.join(self.output_path, "116th_St_histogram.png")
        fig.savefig(filename)

        # Histogram: All train lines
        fig, ax = plt.subplots()
        ax.hist(self.df['PM2.5_calib'].loc[self.df['Loc_code'].isin(['PU', 'T', 'PA'])], color='darkgray', bins=40)
        ax.axvline(35, color='red', linestyle='dashed')
        # ax.set_xticklabels([])
        # ax.set_yticklabels([])

        ax.text(14, 90, "35 µg/m^3", bbox=dict(facecolor='white', alpha=0.8), **dict(size=11, color='red'))
        ax.set_xlim(left=0)
        ax.set(title='PM2.5 Samples for All Platforms and Trains',
               xlabel='PM2.5 Concentration (µg/m^3)', ylabel='No. Samples')
        filename = os.path.join(self.output_path, "All_Stations_histogram.png")
        fig.savefig(filename)
