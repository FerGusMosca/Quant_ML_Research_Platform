import configparser


class Configuration:
    def __init__(self, configFile):
        config = configparser.ConfigParser()
        config.read(configFile)

        self.SingletonAssembly = config['DEFAULT']['SINGLETON_ASSEMBLY']
        self.SingletonConfigFile = config['DEFAULT']['SINGLETON_CONFIG_FILE']
        self.ModuleDirection = config['DEFAULT']['MODULE_DIRECTION']


