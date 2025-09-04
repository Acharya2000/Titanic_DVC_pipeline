from dataingestion import DataIngestion
from datapreprocessing import Preprocessing

def run_data_ingestion():
    # Suppose Titanic dataset is downloaded at datasets/titanic.csv
    ingestion = DataIngestion(source_path=r"C:\Users\SUMAN\Downloads\Titanic-Dataset.csv")
    raw_path = ingestion.save_raw()
    print(f"Raw data stored at: {raw_path}")

def save_preprocessing_data():
    load_data=Preprocessing(r"C:\Users\SUMAN\Desktop\Titanic_DVC_pipeline\data\raw\train.csv",0.24,42)
    load_data.save_preprocessed_data()


if __name__ == "__main__":
    run_data_ingestion()
    save_preprocessing_data()

