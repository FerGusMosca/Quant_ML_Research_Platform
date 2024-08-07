from sources.framework.util.singleton.common.util.singleton import *
from sources.framework.common.interfaces.icommunication_module import ICommunicationModule
from sources.framework.common.abstract.base_communication_module import *
from sources.framework.common.logger.message_type import *
from sources.framework.util.singleton.common.configuration.configuration import *
from sources.framework.common.dto.cm_state import *
import importlib


class SingletonModuleHandler(BaseCommunicationModule):
    def __init__(self):
        self.Configuration=None
        self.SingletonHandler=None
        self.Initialized=False

    def ProcessMessage(self, wrapper):
        try:
            return self.SingletonHandler.ProcessMessage (wrapper)
        except Exception as e:
            self.DoLog("@{}:Error @ProcessMessage. Error={} ".format(self.Configuration.Name,e), MessageType.ERROR)
            return CMState.BuildFailure(self, Exception=e)

    def LoadConfig(self):
        self.Configuration = Configuration(self.ModuleConfigFile)
        return True


    def InitializeSingletonClass(self,InvokingModule):
        module_name, class_name = self.Configuration.SingletonAssembly.rsplit(".", 1)
        singleton_module_class = getattr(importlib.import_module(module_name), class_name)

        if singleton_module_class is not None:
            self.SingletonHandler = singleton_module_class.instance()
            state = self.SingletonHandler.Initialize( InvokingModule,self.Configuration.SingletonConfigFile)

            if not state.Success:
                raise state.Exception

            if self.Configuration.ModuleDirection== "O":
                self.SingletonHandler.SetOutgoingModule(InvokingModule)
            elif  self.Configuration.ModuleDirection == "I":
                self.SingletonHandler.SetIncomingModule(InvokingModule)
            else:
                raise Exception("Unknown module direction {0}".format(self.Configuration.ModuleDirection),MessageType.ERROR)
        else:
            raise Exception("Could not instantiate module {}".format(self.Configuration.SingletonConfigFile))


    def Initialize(self, pInvokingModule, pConfigFile):

        self.ModuleConfigFile = pConfigFile

        try:
            if not self.Initialized:
                self.Initialized=True
                self.InvokingModule = pInvokingModule
                if self.LoadConfig():

                    self.InitializeSingletonClass(pInvokingModule)

                    self.DoLog("SingletonModuleHandler Successfully initializer", MessageType.INFO)

                    return CMState.BuildSuccess(self)


                else:
                    self.DoLog("Error initializing config file for SingletonModuleHandler", MessageType.ERROR)
                    return CMState.BuildFailure(self, Exception=e)

        except Exception as e:
            self.DoLog("Error Loading Singleton Modules Handler module:{}".format(str(e)), MessageType.ERROR)
            return CMState.BuildFailure(self, Exception=e)