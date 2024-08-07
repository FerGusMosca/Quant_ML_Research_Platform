import importlib
class BaseCommunicationModule:
    def __init__(self):

        self.InvokingModule = None
        self.Configuration = None
        self.ModuleConfigFile = None

    def DoLog(self, msg, message_type):

        if self.InvokingModule is not None:
            self.InvokingModule.DoLog(msg, message_type)
        else:
            print(msg)


    def InitializeSingletonModule(self, module, configFile):
        """ Initialize Generic Order Router with specific module name from configuration file.

        """
        module_name, class_name = module.rsplit(".", 1)
        module_class = getattr(importlib.import_module(module_name), class_name)

        if module_class is not None:
            module =  module_class.instance()
            module.Initialize(self,configFile)
            return module
        else:
            raise Exception("Could not instantiate module {}".format(module))

    def InitializeModule(self, module, configFile):
        """ Initialize Generic Order Router with specific module name from configuration file.

        """
        module_name, class_name = module.rsplit(".", 1)
        module_class = getattr(importlib.import_module(module_name), class_name)

        if module_class is not None:
            module = module_class()
            state = module.Initialize(self, configFile)

            if not state.Success:
                raise state.Exception

            return module
        else:
            raise Exception("Could not instantiate module {}".format(module))
