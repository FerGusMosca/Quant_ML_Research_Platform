from framework.common.logger.message_type import MessageType


class RawFileReader():

    @staticmethod
    def get_raw_text(file_path):
        # Load file text
        try:
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
                return  text
        except Exception as e:
            raise Exception(
                    f"❌ Failed reading {file_path}: {e}")
