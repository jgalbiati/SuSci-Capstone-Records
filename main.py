import os
import pandas as pd
from Calibrations_analysis import AQCalibrator
from Graphs_Charts import GraphsCharts
if __name__ == "__main__":
    calibs = AQCalibrator()
    calibs.load_data()
    calibs.calibrate_data()

    grapher = GraphsCharts(data=calibs.df)
    grapher.value_gen()
    grapher.run_boxplots()
    grapher.run_barplots()
    grapher.run_histograms()

