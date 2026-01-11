from src.analysis import region_failure_clusters, data_drift_over_time, detect_anomalies
from src.visualization import plot_data_drift, plot_region_anomalies


def main():
    print("Running region failure cluster analysis...")
    region_data = detect_anomalies()
    print(region_data.head())

    print("\nRunning data drift analysis...")
    drift_data = data_drift_over_time()

    print("\nPlotting visualizations...")
    plot_data_drift(drift_data)
    plot_region_anomalies(region_data)


if __name__ == "__main__":
    main()

