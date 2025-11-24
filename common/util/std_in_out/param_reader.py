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
    # All comments MUST be in English.
    @staticmethod
    def get_value_after_equals(command: str, key: str, optional: bool = False):
        """
        Extracts EXACT raw value after key=.
        Logs whether the parameter was quoted or unquoted.
        NEVER modifies spaces inside the value.
        """
        key_pattern = f"{key}="
        if key_pattern not in command:
            if optional:
                print(f"[ParamReader] (optional) key '{key}' NOT FOUND")
                return None
            raise ValueError(f"Missing required parameter: {key}")

        after = command.split(key_pattern, 1)[1].lstrip()

        # --- QUOTED VALUE ---
        if after.startswith('"') or after.startswith("'"):
            q = after[0]
            end = after.find(q, 1)
            if end == -1:
                raise ValueError(f"Unclosed quote for key '{key}'")

            val = after[1:end]

            print(f"[ParamReader] key='{key}' → QUOTED value detected: {val}")

            return val

        # --- UNQUOTED VALUE ---
        val = after.split(" ", 1)[0]

        print(f"[ParamReader] key='{key}' → UNQUOTED value detected: {val}")

        return val


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