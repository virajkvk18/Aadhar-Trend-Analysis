import pandas as pd
from sklearn.ensemble import IsolationForest
import numpy as np
from src.preprocess import load_all_data, clean_data


def region_failure_clusters():
    """
    Builds region-level failure clusters using update pressure
    as a proxy for enrolment quality issues
    """

    # Load and clean data
    enrolment, biometric, demographic = load_all_data()
    enrolment, biometric, demographic = clean_data(
        enrolment, biometric, demographic
    )

    # ---- ENROLMENT TOTAL ----
    enrolment["total_enrolment"] = (
        enrolment["age_0_5"]
        + enrolment["age_5_17"]
        + enrolment["age_18_greater"]
    )

    enrolment_state = (
        enrolment
        .groupby("state")["total_enrolment"]
        .sum()
        .reset_index()
    )

    # ---- BIOMETRIC UPDATES TOTAL ----
    biometric["total_bio_updates"] = (
        biometric["bio_age_5_17"]
        + biometric["bio_age_17_"]
    )

    biometric_state = (
        biometric
        .groupby("state")["total_bio_updates"]
        .sum()
        .reset_index()
    )

    # ---- DEMOGRAPHIC UPDATES TOTAL ----
    demographic["total_demo_updates"] = (
        demographic["demo_age_5_17"]
        + demographic["demo_age_17_"]
    )

    demographic_state = (
        demographic
        .groupby("state")["total_demo_updates"]
        .sum()
        .reset_index()
    )

    # ---- MERGE ALL ----
    region_data = (
        enrolment_state
        .merge(biometric_state, on="state", how="left")
        .merge(demographic_state, on="state", how="left")
    )

    region_data.fillna(0, inplace=True)

    # ---- UPDATE RATIO (FAILURE PROXY) ----
    region_data["update_ratio"] = (
        region_data["total_bio_updates"]
        + region_data["total_demo_updates"]
    ) / region_data["total_enrolment"]

    return region_data
def data_drift_over_time():
    """
    Analyzes enrolment vs update drift over time
    """

    enrolment, biometric, demographic = load_all_data()
    enrolment, biometric, demographic = clean_data(
        enrolment, biometric, demographic
    )

    # Convert date column
    enrolment["date"] = pd.to_datetime(enrolment["date"], dayfirst=True)
    biometric["date"] = pd.to_datetime(biometric["date"], dayfirst=True)
    demographic["date"] = pd.to_datetime(demographic["date"], dayfirst=True)


    # Total enrolment per date
    enrolment["total_enrolment"] = (
        enrolment["age_0_5"]
        + enrolment["age_5_17"]
        + enrolment["age_18_greater"]
    )

    enrolment_time = (
        enrolment
        .groupby("date")["total_enrolment"]
        .sum()
        .reset_index()
    )

    # Total updates per date
    biometric["total_bio_updates"] = (
        biometric["bio_age_5_17"]
        + biometric["bio_age_17_"]
    )

    demographic["total_demo_updates"] = (
        demographic["demo_age_5_17"]
        + demographic["demo_age_17_"]
    )

    updates_time = (
        biometric.groupby("date")["total_bio_updates"].sum()
        + demographic.groupby("date")["total_demo_updates"].sum()
    ).reset_index(name="total_updates")

    # Merge both
    drift_data = enrolment_time.merge(
        updates_time, on="date", how="left"
    )

    drift_data.fillna(0, inplace=True)

    drift_data["update_pressure"] = (
        drift_data["total_updates"] / drift_data["total_enrolment"]
    )

    return drift_data
def detect_anomalies():
    """
    Detect anomalous states based on update_ratio using Isolation Forest
    """

    region_data = region_failure_clusters()

    # Prepare data for model
    X = region_data[["update_ratio"]].values

    # Initialize Isolation Forest
    clf = IsolationForest(contamination=0.1, random_state=42)
    clf.fit(X)

    # Predict anomalies (-1: anomaly, 1: normal)
    preds = clf.predict(X)

    region_data["anomaly"] = preds
    region_data["anomaly"] = region_data["anomaly"].map({1: "Normal", -1: "Anomaly"})

    return region_data.sort_values(by="update_ratio", ascending=False)
