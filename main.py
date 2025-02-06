import os
import pandas as pd
import Calibrations_analysis
if __name__ == "__main__":
    obj = Calibrations_analysis()
    obj.load_data()
    obj.calibrate_data()

