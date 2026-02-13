import configparser

class MLSettingsLoader():

    def __init__(self):
        pass


    def load_settings(self,config_file_path):
        try:
            config = configparser.ConfigParser()
            config.read(config_file_path)
            config_settings={}


            config_settings["hist_data_conn_str"]= config['DB']['HIST_DATA_CONN_STR']
            config_settings["ml_reports_conn_str"] = config['DB']['ML_REPORTS_CONN_STR']
            config_settings["fund_mgmt_dashboard_cs"] = config['DB']['FUND_MGMT_DASHBOARD_CS']

            config_settings["NEO4J_URI"] = config['GRAPH_DB']['NEO4J_URI']
            config_settings["NEO4J_USER"] = config['GRAPH_DB']['NEO4J_USER']
            config_settings["NEO4J_PASS"] = config['GRAPH_DB']['NEO4J_PASS']

            config_settings["QDRANT_SERVER"] = config['QDRANT']['QDRANT_SERVER']
            config_settings["QDRANT_PORT"] = config['QDRANT']['QDRANT_PORT']

            config_settings["LANGFUSE_SECRET_KEY"] = config['LANGFUSE']['LANGFUSE_SECRET_KEY']
            config_settings["LANGFUSE_PUBLIC_KEY"] = config['LANGFUSE']['LANGFUSE_PUBLIC_KEY']
            config_settings["LANGFUSE_BASE_URL"] = config['LANGFUSE']['LANGFUSE_BASE_URL']

            config_settings["OBSERVABILITY_PROXY_URL"] = config['MCP_OBSERVABILITY']['OBSERVABILITY_PROXY_URL']


            config_settings["classification_map_key"] = config['SETTINGS']['CLASSIFICATION_MAP_KEY']
            config_settings["IB_PROD_WS"] = config['SETTINGS']['IB_PROD_WS']
            config_settings["PRIMARY_PROD_WS"] = config['SETTINGS']['PRIMARY_PROD_WS']
            config_settings["IB_DEV_WS"] = config['SETTINGS']['IB_DEV_WS']
            config_settings["FRED_API_KEY"] = config['SETTINGS']['FRED_API_KEY']
            config_settings["TRADING_VIEW_USER"] = config['SETTINGS']['TRADING_VIEW_USER']
            config_settings["TRADING_VIEW_PWD"] = config['SETTINGS']['TRADING_VIEW_PWD']
            config_settings["BCRA_API_KEY"] = config['SETTINGS']['BCRA_API_KEY']
            config_settings["OUTPUT_RF_FOLDER"] = config['FOLDERS']['OUTPUT_RF_FOLDER']
            config_settings["OUTPUT_SECURITIES_REPORTS_FOLDER"] = config['FOLDERS']['OUTPUT_SECURITIES_REPORTS_FOLDER']
            config_settings["UI_PORT"] = config['UI']['UI_PORT']
            config_settings["MLFLOW_TRACKING_URI"] = config['ML_FLOW']['MLFLOW_TRACKING_URI']
            config_settings["LOKI_URL"] = config['GRAFANA']['LOKI_URL']
            config_settings["GRAFANA_ON"] = config['GRAFANA']['GRAFANA_ON']=="True"

            config_settings["RAG_OUTPUT_FOLDER"] = config['EMBEDDINGS']['RAG_OUTPUT_FOLDER']


            return  config_settings



        except Exception as e:
            raise Exception("Critical error reading config file {}:{}".format(config_file_path,str(e)))