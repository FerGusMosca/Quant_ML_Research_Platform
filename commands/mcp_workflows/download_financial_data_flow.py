import os
import asyncio
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
import argparse
from common.util.std_in_out.mcp_settings_loader import MCPSettingsLoader
from common.util.std_in_out.root_locator import RootLocator
from service_layer.client.mcp.mcp_report_client import ReportMCPClient


# ---------- NODES ----------

def download_securities_calendar(state: dict) -> dict:
    print("[FLOW] download_securities_calendar START", flush=True)

    client = ReportMCPClient(
        uri=state["MCP_REPORTS_URI"],
        report="download_securities_reports_calendar",
        arguments={
            "portfolio": state["portfolio"],
            "year": state["year"],
        },
    )

    async def run():
        async for msg in client.execute_and_stream():
            print(msg, end="", flush=True)

    try:
        asyncio.run(run())
    except Exception as e:
        return {**state, "securities_calendar_status": "fail", "securities_calendar_error": str(e)}

    if not client.success:
        return {**state, "securities_calendar_status": "fail", "securities_calendar_error": client.last_error}

    return {**state, "securities_calendar_status": "ok"}


def download_k10(state: dict) -> dict:
    print("[FLOW] download_k10 START", flush=True)

    client = ReportMCPClient(
        uri=state["MCP_REPORTS_URI"],
        report="download_k10",
        arguments={
            "portfolio": state["portfolio"],
            "year": state["year"],
        },
    )

    async def run():
        async for msg in client.execute_and_stream():
            print(msg, end="", flush=True)

    try:
        asyncio.run(run())
    except Exception as e:
        return {**state, "k10_status": "fail", "error": str(e)}

    if not client.success:
        return {**state, "k10_status": "fail", "error": client.last_error}

    return {**state, "k10_status": "ok"}


def download_q10(state: dict) -> dict:
    print("[FLOW] download_q10 START", flush=True)

    client = ReportMCPClient(
        uri=state["MCP_REPORTS_URI"],
        report="download_q10",
        arguments={
            "portfolio": state["portfolio"],
            "year": state["year"],
        },
    )

    async def run():
        async for msg in client.execute_and_stream():
            print(msg, end="", flush=True)

    try:
        asyncio.run(run())
    except Exception as e:
        return {**state, "q10_status": "fail", "error": str(e)}

    if not client.success:
        return {**state, "q10_status": "fail", "error": client.last_error}

    return {**state, "q10_status": "ok"}


# ---------- ROUTING ----------

def after_download_securities_calendar(state: dict):
    return "download_k10" if state.get("securities_calendar_status") == "ok" else END

def after_k10(state: dict):
    return "download_q10" if state.get("k10_status") == "ok" else END


# ---------- MAIN ----------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MCP download workflow")
    parser.add_argument("--portfolio", required=True, help="Portfolio code")
    parser.add_argument("--year", required=True, help="Year or year range")
    args = parser.parse_args()

    loader = MCPSettingsLoader()
    config_file = os.path.join(RootLocator.get_root(), "configs/mcp_config.ini")
    config = loader.load_settings(config_file)

    MCP_REPORTS_URI = config["MCP_REPORTS_URI"]
    if not MCP_REPORTS_URI:
        raise RuntimeError("Missing MCP_REPORTS_URI")

    portfolio = args.portfolio
    year = args.year

    graph = StateGraph(dict)

    graph.add_node("download_securities_calendar", download_securities_calendar)
    graph.add_node("download_k10", download_k10)
    graph.add_node("download_q10", download_q10)

    graph.set_entry_point("download_securities_calendar")

    graph.add_conditional_edges("download_securities_calendar", after_download_securities_calendar)
    graph.add_conditional_edges("download_k10", after_k10)
    graph.add_edge("download_q10", END)

    final_state = graph.compile().invoke({
        "portfolio": portfolio,
        "year": year,
        "MCP_REPORTS_URI": MCP_REPORTS_URI,
    })

    print("\n[FLOW] FINAL STATE")
    print(final_state)


if __name__ == "__main__":
    main()