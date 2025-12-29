from logging.handlers import TimedRotatingFileHandler
import configparser
import logging
import os

from common.util.logging.loki_logger import LokiLogger
from common.util.std_in_out.ml_settings_loader import MLSettingsLoader
from framework.common.logger.message_type import MessageType


class Logger:

    def __init__(self):
        self.logger = logging.getLogger("ml_research")
        self.config = configparser.ConfigParser()
        self.config.read("configs/logger.ini")

        self.level = int(self.config['DEFAULT']['level'])
        self.log_dir = self.config['DEFAULT']['log_dir']
        self.when_to_rotate = self.config['DEFAULT']['when_to_rotate']
        self.backup_count = int(self.config['DEFAULT']['backup_count'])
        self.log_file_name = self.config['DEFAULT']['log_file_name']

        self._observers = []

        loader = MLSettingsLoader()
        config_settings = loader.load_settings("./configs/commands_mgr.ini")
        self.loki_url = config_settings.get("LOKI_URL")
        self.grafana_on=config_settings.get("GRAFANA_ON")

        # --- NEW: Loki logger ---
        self.loki = LokiLogger(
            loki_url=self.loki_url,
            app_name="ml_research"
        )

    def register_observer(self, obs):
        self._observers.append(obs)

    def unregister_observer(self, obs):
        self._observers.remove(obs)

    def _notify(self, msg, msg_type,job_id=None):
        if job_id is None:
            return

        for obs in self._observers:
            obs.on_log(msg, msg_type,job_id)


    def use_timed_rotating_file_handler(self):
        if self.level is None:
            self.level = logging.INFO

        log_path = os.path.join(self.log_dir, self.log_file_name)

        main_formatter = logging.Formatter(
            fmt='%(asctime)s [%(module)s %(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S')

        console_handler = logging.StreamHandler()
        file_handler = TimedRotatingFileHandler(
            filename=log_path, when=self.when_to_rotate, backupCount=self.backup_count)

        for handler in [console_handler, file_handler]:
            handler.setFormatter(main_formatter)
            self.logger.addHandler(handler)

        self.logger.setLevel(self.level)

    def print(self, msg, msg_type):

        # --- NEW: Push to Loki ---
        if self.grafana_on:
            self.loki.push(msg_type.name, msg)

        if msg_type == MessageType.CRITICAL:
            self.logger.critical(msg)
        if msg_type == MessageType.ERROR:
            self.logger.error(msg)
        if msg_type == MessageType.WARNING:
            self.logger.warning(msg)
        if msg_type == MessageType.INFO:
            self.logger.info(msg)
        if msg_type == MessageType.DEBUG:
            self.logger.debug(msg)

    def do_log(self, msg, msg_type,job_id=None):
        print(msg)
        self._notify(msg, msg_type,job_id)
        self.print(msg, msg_type)


    def do_log_light(self, msg,job_id=None):
        print(msg,flush=True)
        self._notify(msg,  MessageType.INFO,job_id)
        self.print(msg,  MessageType.INFO)
