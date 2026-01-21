import os
import asyncio
import argparse
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from common.util.std_in_out.mcp_settings_loader import MCPSettingsLoader
from common.util.std_in_out.root_locator import RootLocator
from service_layer.client.mcp.mcp_report_client import ReportMCPClient


# ---------- NODES ----------

def run_sentiment_k10(state: dict) -> dict:
    print("[FLOW] sentiment_summary_report_k10 START", flush=True)

    client = ReportMCPClient(
        uri=state["MCP_REPORTS_URI"],
        report="sentiment_summary_report_k10",
        arguments={
            "portfolio": state["portfolio"],
            "year": state["year"],
            "dest_folder": state["dest_folder"],
            "rank_folder": state["rank_folder"],
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


def run_sentiment_q10(state: dict) -> dict:
    print("[FLOW] sentiment_summary_report_q10 START", flush=True)

    client = ReportMCPClient(
        uri=state["MCP_REPORTS_URI"],
        report="sentiment_summary_report_q10",
        arguments={
            "portfolio": state["portfolio"],
            "year": state["year"],
            "dest_folder": state["dest_folder"],
            "rank_folder": state["rank_folder"],
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

def after_k10(state: dict):
    return "run_sentiment_q10" if state.get("k10_status") == "ok" else END


# ---------- MAIN ----------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MCP sentiment summary workflow")
    parser.add_argument("--portfolio", required=True)
    parser.add_argument("--year", required=True)
    parser.add_argument("--dest-folder", required=True)
    parser.add_argument("--rank-folder", required=True)
    args = parser.parse_args()

    loader = MCPSettingsLoader()
    config_file = os.path.join(RootLocator.get_root(), "configs/mcp_config.ini")
    config = loader.load_settings(config_file)

    MCP_REPORTS_URI = config["MCP_REPORTS_URI"]
    if not MCP_REPORTS_URI:
        raise RuntimeError("Missing MCP_REPORTS_URI")

    graph = StateGraph(dict)

    graph.add_node("run_sentiment_k10", run_sentiment_k10)
    graph.add_node("run_sentiment_q10", run_sentiment_q10)

    graph.set_entry_point("run_sentiment_k10")
    graph.add_conditional_edges("run_sentiment_k10", after_k10)
    graph.add_edge("run_sentiment_q10", END)

    final_state = graph.compile().invoke({
        "portfolio": args.portfolio,
        "year": args.year,
        "dest_folder": args.dest_folder,
        "rank_folder": args.rank_folder,
        "MCP_REPORTS_URI": MCP_REPORTS_URI,
    })

    print("\n[FLOW] FINAL STATE")
    print(final_state)


if __name__ == "__main__":
    main()
