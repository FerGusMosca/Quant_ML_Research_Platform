from enum import Enum

class ReportType(Enum):
    DOWNLOAD_K10 = "download_k10"
    DOWNLOAD_Q10 = "download_q10"
    COMPETITION_SUMMARY_REPORT_Q10 = "competition_summary_report_q10"
    COMPETITION_SUMMARY_REPORT_K10 = "competition_summary_report_k10"
    SENTIMENT_SUMMARY_REPORT_K10 = "sentiment_summary_report_k10"
    SENTIMENT_SUMMARY_REPORT_Q10 = "sentiment_summary_report_q10"
