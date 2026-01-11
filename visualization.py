import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style="whitegrid")


def plot_data_drift(drift_data):
    plt.figure(figsize=(12, 6))
    plt.plot(drift_data["date"], drift_data["total_enrolment"], label="Total Enrolment")
    plt.plot(drift_data["date"], drift_data["total_updates"], label="Total Updates")
    plt.plot(drift_data["date"], drift_data["update_pressure"], label="Update Pressure")
    plt.xlabel("Date")
    plt.ylabel("Count / Ratio")
    plt.title("Data Drift Over Time")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_region_anomalies(region_data):
    plt.figure(figsize=(14, 8))
    sns.barplot(
        data=region_data,
        x="update_ratio",
        y="state",
        hue="anomaly",
        dodge=False,
        palette={"Normal": "green", "Anomaly": "red"},
    )
    plt.xlabel("Update Ratio (Failure Proxy)")
    plt.ylabel("State")
    plt.title("Region Failure Clusters with Anomalies")
    plt.tight_layout()
    plt.show()
