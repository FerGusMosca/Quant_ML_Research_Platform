from enum import Enum

class ReportType(Enum):
    DOWNLOAD_K10 = "download_k10"
    DOWNLOAD_Q10 = "download_q10"
    DOWNLOAD_YEARLY_INCOME_STATEMENT = "download_yearly_income_statement"
    DOWNLOAD_QUARTERLY_INCOME_STATEMENT = "download_quarterly_income_statement"
    FINVIZ_NEWS_DOWNLOAD = "finviz_news_download"
    COMPETITION_SUMMARY_REPORT_Q10 = "competition_summary_report_q10"
    COMPETITION_SUMMARY_REPORT_K10 = "competition_summary_report_k10"
    SENTIMENT_SUMMARY_REPORT_K10 = "sentiment_summary_report_k10"
    SENTIMENT_SUMMARY_REPORT_Q10 = "sentiment_summary_report_q10"
    FINANCIAL_RATIOS_REPORT_K10 = "financial_ratios_report_k10"
    FINANCIAL_RATIOS_REPORT_Q10 = "financial_ratios_report_q10"
