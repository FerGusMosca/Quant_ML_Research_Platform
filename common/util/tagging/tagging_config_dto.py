import os
from typing import Optional
from common.enums.sec_reports import SECReports
from common.util.std_in_out.K_Q_10_file_locator import KQ10FileLocator


class TaggingConfigDTO:
    DOC_TYPE_K_Q_10 = "K_Q_10"
    SIM_THRESHOLD_DEF = 0.8
    def __init__(
        self,
        tag_model: str,
        tag_file: str,
        tags_csv: str,
        sim_threshold: float = 0.8,
        doc_type: Optional[str] = None,
        tag_json:str=None
    ):
        self.tag_model = tag_model
        self.tag_file = tag_file
        self.tags_csv = tags_csv
        self.sim_threshold = sim_threshold
        self.doc_type = doc_type
        self.tag_json=tag_json

    def is_K_Q_10_doc(self) -> bool:
        return self.doc_type == TaggingConfigDTO.DOC_TYPE_K_Q_10

    def evaluate_file_for_report(self, symbol, source, file_name, year, quarter):
        if not self.is_K_Q_10_doc():
            return True

        file_only = os.path.basename(file_name)

        if SECReports.K10.value in source:
            # e.g. HD_2025_10-K.html
            #return file_only.startswith(f"{symbol}_{year}_10-K")
            return  KQ10FileLocator.find_file(source,file_only,symbol,year,quarter)

        if SECReports.Q10.value in source:
            # e.g. GPI_2025_Q1_10-Q.html
            #return file_only.startswith(f"{symbol}_{year}_{quarter}_10-Q")
            return KQ10FileLocator.find_file(source, file_only, symbol, year, quarter)

        # Not implemented / all files pass
        return True


