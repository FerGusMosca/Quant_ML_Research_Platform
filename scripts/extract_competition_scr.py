from common.util.extractors.K_Q_10.k_q_10_html_structured_block_extractor import KQ10HtmlStructuredBlockExtractor

kq10_stru_down=KQ10HtmlStructuredBlockExtractor()


file_path="C:\\Projects\\Bias\\research_apps\\Quant_ML_Research_Platform\\output\\securities_reports\\US_BIGCAP_EX_TEST_SMALL\\K10\\2022\\ORCL_2022_10-K.html"

with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
    raw_text = f.read()


kq10_stru_down.extract_blocks_adv(raw_text,["COMPETITION"])