import os
import pandas as pd

# Get project root directory (Aadhar-Trend-Analysis)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR = os.path.join(BASE_DIR, "data")


def load_csvs_from_folder(folder_path):
    all_files = [
        os.path.join(folder_path, file)
        for file in os.listdir(folder_path)
        if file.endswith(".csv")
    ]

    dataframes = []
    for file in all_files:
        df = pd.read_csv(file)
        dataframes.append(df)

    combined_df = pd.concat(dataframes, ignore_index=True)
    return combined_df


def load_all_data():
    enrolment = load_csvs_from_folder(os.path.join(DATA_DIR, "enrolment"))
    biometric = load_csvs_from_folder(os.path.join(DATA_DIR, "biometric"))
    demographic = load_csvs_from_folder(os.path.join(DATA_DIR, "demographic"))

    return enrolment, biometric, demographic


def clean_data(enrolment, biometric, demographic):
    for df in [enrolment, biometric, demographic]:
        df.columns = df.columns.str.lower().str.strip()
        df.fillna(0, inplace=True)

    return enrolment, biometric, demographic
