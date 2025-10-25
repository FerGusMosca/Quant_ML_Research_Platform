"""
ParamReader.py
--------------
Utility class for parsing key-value parameters from command strings.
Provides automatic type conversion for dates, integers, and floats.
"""
from common.util.financial_calculations.date_handler import DateHandler


class ParamReader:
    """
    Static utility class for parsing and converting command parameters.
    """

    _DATE_FORMAT = "%m/%d/%Y"
    _TIMESTAMP_FORMAT = "%m/%d/%Y %H:%M:%S"

    # ============================================================
    # Extract raw value after "key="
    # ============================================================
    @staticmethod
    def get_value_after_equals(command: str, key: str, optional: bool = False):
        """
        Extracts the raw value that follows 'key=' in a command string.

        Example:
            command = "RunReport report=download_q10 year=2025"
            get_value_after_equals(command, "report") -> "download_q10"
        """
        try:
            key_pattern = f"{key}="
            if key_pattern not in command:
                if not optional:
                    raise ValueError(f"Missing required parameter: {key}")
                return None

            # Extract substring after key=
            after = command.split(key_pattern, 1)[1]
            # Stop at first space or end of string
            value = after.split(" ", 1)[0]
            return value.strip()

        except Exception:
            if optional:
                return None
            raise

    # ============================================================
    # Convert parameter value to correct type
    # ============================================================
    @staticmethod
    def get_param(command: str, key: str, optional: bool = False, def_value=None):
        """
        Retrieves and converts a parameter value from the command string.
        Automatically attempts conversion to date, int, or float.

        Example:
            command = "RunReport report=download_q10 year=2025"
            get_param(command, "year") -> 2025 (int)
        """
        value = ParamReader.get_value_after_equals(command, key, optional)

        if value is None and optional:
            return def_value

        # Attempt conversions in order of priority
        try:
            return DateHandler.convert_str_date(value, ParamReader._DATE_FORMAT)
        except Exception:
            pass

        try:
            return DateHandler.convert_str_date(value, ParamReader._TIMESTAMP_FORMAT)
        except Exception:
            pass

        try:
            return int(value)
        except ValueError:
            pass

        try:
            return float(value)
        except ValueError:
            pass

        # Fallback: return as string
        return value


    @staticmethod
    def get_bool_param(command, key, optional=False, def_value=None):
        str_val = ParamReader.get_param(command, key, optional, def_value)

        if str_val == "True" or str_val == "False":
            return str_val == "True"
        else:
            return def_value


    @staticmethod
    def params_validation(cmd, param_list, exp_len):
        if (len(param_list) != exp_len):
            raise Exception("Command {} expects {} parameters".format(cmd, exp_len))