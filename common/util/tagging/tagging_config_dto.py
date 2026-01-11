from typing import Optional



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
