import os
import asyncio
import argparse
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END

from common.util.std_in_out.mcp_settings_loader import MCPSettingsLoader
from common.util.std_in_out.root_locator import RootLocator
from service_layer.client.mcp.mcp_ingest_client import RAGIngestMCPClient

# ---------- NODE ----------

def run_rag_ingest(state: dict) -> dict:
    print("[FLOW] run_rag_ingest START", flush=True)

    client = RAGIngestMCPClient(
        mode=state["mode"],
        source=state["source"],
        dest_root=state["dest_root"],
        chunk_name=state["chunk_name"],
        embedding_model=None,
        clustering_model=None,
        log_posfix=None,
        uri=state["MCP_INGEST_URI"],
    )

    async def run():
        async for msg in client.execute_and_stream():
            print(msg, end="", flush=True)

    try:
        try:
            loop = asyncio.get_running_loop()
            loop.run_until_complete(run())
        except RuntimeError:
            asyncio.run(run())
    except Exception as e:
        return {**state, "status": "fail", "error": str(e)}

    if not client.success:
        return {**state, "status": "fail", "error": client.last_error}

    return {
        **state,
        "status": "ok",
        "out_folder": client.last_output_folder,
    }

# ---------- MAIN ----------

def main():
    load_dotenv()

    parser = argparse.ArgumentParser(description="MCP RAG ingest workflow")
    parser.add_argument("--mode", required=True)
    parser.add_argument("--source", required=True)
    parser.add_argument("--dest-root", required=True)
    parser.add_argument("--chunk-name", required=True)
    args = parser.parse_args()

    loader = MCPSettingsLoader()
    config_file = os.path.join(RootLocator.get_root(), "configs/mcp_config.ini")
    config = loader.load_settings(config_file)

    MCP_INGEST_URI = config.get("MCP_INGEST_URI")
    if not MCP_INGEST_URI:
        raise RuntimeError("Missing MCP_INGEST_URI")

    graph = StateGraph(dict)
    graph.add_node("run_rag_ingest", run_rag_ingest)
    graph.set_entry_point("run_rag_ingest")
    graph.add_edge("run_rag_ingest", END)

    final_state = graph.compile().invoke({
        "mode": args.mode,
        "source": args.source,
        "dest_root": args.dest_root,
        "chunk_name": args.chunk_name,
        "MCP_INGEST_URI": MCP_INGEST_URI,
    })

    print("\n[FLOW] FINAL STATE")
    print(final_state)

if __name__ == "__main__":
    main()
