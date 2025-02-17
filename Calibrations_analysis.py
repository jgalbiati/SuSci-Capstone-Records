
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression as lr
import datetime as dt
import matplotlib.pyplot as plt
plt.style.use('seaborn-whitegrid')


class AQCalibrator:
    def __init__(self):
        self.df = None

    def load_data(self):
        """Loads and cleans raw data"""
        def utc_delocalize(frame, field, timestart, timeend):
            """Function to remove timezone awareness from utc datetime fields (for easier manipulation), and to trim
            data to start and end timestamps:"""
            frame[field] = pd.to_datetime(frame[field]).dt.tz_localize(None)
            frame = frame.loc[(frame[field] >= timestart) & (frame[field] < timeend)]
            frame = frame.set_index(frame[field])
            return frame

        def null_remover(frame, fields):
            for field in fields:
                frame = frame.loc[frame[field].notnull()]
            return frame

        # designate timespan we want
        start = dt.datetime.strptime('2020-07-10', '%Y-%m-%d')
        end = dt.datetime.strptime('2021-01-27', '%Y-%m-%d')
        
        # Read in the NY DEC data, combine date/time columns into one datetime field for easier analysis
        refdata = pd.read_csv('data/nydec.csv', header=1, parse_dates=[['Date', 'Time']])
        refdata = refdata.rename(columns={"PM25C_ug/m3LC": "PM2.5_ref"})
        
        # Localize refdata to UTC (then remove timezone awareness). refdata's datetime field was captured in Eastern
        #   time, but the field is not timezone-aware.
        # First set timezone awareness, then convert to UTC:
        refdata['Date_Time'] = refdata['Date_Time'].dt.tz_localize('US/Eastern', ambiguous='NaT')
        refdata['Date_Time'] = refdata['Date_Time'].dt.tz_convert('UTC')
        refdata = utc_delocalize(refdata, 'Date_Time', start, end)
        
        # read in purpleair datasets: hour averages (for best comparison to refdata data)
        sens_a = pd.read_csv('data/school_4_A_hr.csv', header=0)
        sens_a = sens_a.rename(columns={"Temperature_F": "Temp", "PM2.5_ATM_ug/m3": "PM2.5A"})
        sens_a = utc_delocalize(sens_a, 'created_at', start, end)
        
        sens_b = pd.read_csv('data/school_4_B_hr.csv', header=0)
        sens_b = sens_b.rename(columns={"PM2.5_ATM_ug/m3": "PM2.5B"})
        sens_b = utc_delocalize(sens_b, 'created_at', start, end)
        
        # Get all info onto a single dataframe
        df = pd.concat([sens_a[['Temp', 'Humidity_%', 'PM2.5A']], sens_b['PM2.5B'], refdata['PM2.5_ref']], axis=1)
        df['PM2.5_test'] = df[['PM2.5A', 'PM2.5B']].mean(axis=1)
        
        # Null values will cause problems in statswork later on, so eliminate them now.
        df_nulls = ['PM2.5_ref', 'PM2.5_test', 'Temp', 'Humidity_%']
        self.df = null_remover(df, df_nulls)

    def calibrate_data(self):
        """Defines and runs calibration functions"""
        def reg_out(x):
            """Calibration Function for all Non-Train/Platform data"""

            # State-verified data used to calibrate non-subway control figures
            # trained model: nydec = x0 + x1[temp] + x2[humidity] + x3[PA data]
            x_training = np.array(self.df[['Temp', 'Humidity_%', 'PM2.5_test']])
            y = np.array(self.df['PM2.5_ref'])
            reg = lr(fit_intercept=False).fit(x_training, y)
            return np.dot(x, reg.coef_)

        # CALIBRATION FOR UNDERGROUND PLATFORM & TRAIN DATA
        def reg_plat(x):
            """Calibration Function for underground platform/train data"""

            # Mass calculations from air filter collections used to train the calibration model
            filt = pd.read_csv('data/Gravcal.csv', header=0)

            x_training = np.column_stack([np.zeros(len(filt['PA PM2.5 avg'])), filt['PA PM2.5 avg']])
            y = np.array(filt['Grav(average)'])
            calibs = lr(fit_intercept=False).fit(x_training, y)
            return calibs.coef_[1] * x
        
        # Calibration Scheme C
        # CALIBRATION FOR ABOVEGROUND PLATFORMS AND TRAINS
        def reg_sir(x):
            """Calibration function for aboveground platform/train data"""
            return x * (30.9 / 36.2)

        df = self.df
        df['PM2.5_calib'] = np.nan

        def calibrate(frame):
            """Runs the calibration functions, based on codes for where data was collected"""
            codes_nydec = ['O', 'I', 'F']
            codes_filter = ['PU', 'T', 'E']
            codes_sir = ['PA']

            # nydec calibration for non-subway data
            x_nydec = frame[['Temp', 'Humidity_%', 'PM2.5']].loc[frame['Loc_code'].isin(codes_nydec)].copy()
            frame.loc[frame['Loc_code'].isin(codes_nydec), 'PM2.5_calib'] = reg_out(x_nydec)

            # filter calibration for underground trains/platforms
            x_filter = frame['PM2.5'].loc[frame['Loc_code'].isin(codes_filter)].copy()
            frame.loc[frame['Loc_code'].isin(codes_filter), 'PM2.5_calib'] = reg_plat(x_filter)

            # Calibration for aboveground plats
            x_sir = frame['PM2.5'].loc[frame['Loc_code'].isin(codes_sir)].copy()
            frame.loc[frame['Loc_code'].isin(codes_sir), 'PM2.5_calib'] = reg_sir(x_sir)

            # Calibration for trains traveling aboveground
            x_sir2 = frame['PM2.5'].loc[(frame['Loc_code'] == 'T') & (frame['Outside? (Y/N)'] == 'Y')].copy()
            frame.loc[(frame['Loc_code'] == 'T') & (frame['Outside? (Y/N)'] == 'Y'), 'PM2.5_calib'] = reg_sir(x_sir2)

            return frame

        # Read in collected data, run calibrations, and write to the Class
        dfsub = pd.read_csv('data/Master_compiled_BB.csv', header=0, parse_dates=[['Date', 'Time_NYC']]).rename(
            columns={'Temperature (F)': 'Temp', 'Humidity (%)': 'Humidity_%', 'PA PM2.5 (um/m3)': 'PM2.5'})
        
        dfsub = dfsub.loc[dfsub['PM2.5'].notnull() & dfsub['Temp'].notnull() & dfsub['Humidity_%'].notnull() &
                          dfsub['Location'].notnull()]
        dfsub = calibrate(dfsub)

        self.df = dfsub
