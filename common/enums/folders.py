import configparser
from enum import Enum

class Folders(Enum):
    OUTPUT_RF_FOLDER = "./output/sliding_rf_models"
    OUTPUT_SECURITIES_REPORTS_FOLDER = "./output/securities_reports"

    @staticmethod
    def load_from_config(config_path="./configs/commands_mgr.ini"):
        config = configparser.ConfigParser()
        config.read(config_path)
        if "FOLDERS" in config:
            for key, value in config["FOLDERS"].items():
                if key.upper() in Folders.__members__:
                    Folders.__members__[key.upper()]._value_ = value
