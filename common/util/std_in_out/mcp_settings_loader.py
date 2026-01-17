import configparser

class MCPSettingsLoader():

    def __init__(self):
        pass


    def load_settings(self,config_file_path):
        try:
            config = configparser.ConfigParser()
            config.read(config_file_path)
            config_settings={}


            config_settings["MCP_REPORTS_URI"]= config['MCP']['MCP_REPORTS_URI']
            config_settings["MCP_INGEST_URI"] = config['MCP']['MCP_INGEST_URI']

            return  config_settings



        except Exception as e:
            raise Exception("Critical error reading config file {}:{}".format(config_file_path,str(e)))