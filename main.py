# main.py
import os
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.controller_agent import ControllerAgent
from agents.memory_agent import MemoryAgent

def main():
    print("Starting Vision-Enhanced Intent-Aware Multi-Agent RAG Assistant")

    intent_agent = IntentAgent()
    retrieval_agent = RetrievalAgent()
    vision_agent = VisionAgent()
    reasoning_agent = ReasoningAgent()
    memory_agent = MemoryAgent(max_turns=12)

    controller = ControllerAgent(intent_agent, retrieval_agent, vision_agent, reasoning_agent, memory_agent)

    try:
        retrieval_agent.ingest_pdfs(data_dir="data")
    except Exception:
        pass
    try:
        retrieval_agent.ingest_images_folder(images_dir=os.path.join("data", "images"))
    except Exception:
        pass

    print("Ready. Type queries (or 'exit'):\n")
    while True:
        q = input("> ").strip()
        if not q:
            continue
        if q.lower() in ("exit", "quit"):
            break
        res = controller.handle_query(q, top_k=5)
        print("\n=== Intent:", res["intent"], "===\n")
        print(res["answer"])
        print("\n--- Retrieved (top) ---")
        for i, r in enumerate(res["retrieved"]):
            m = r.get("metadata", {})
            print(f"{i+1}. ({m.get('type','')}) {m.get('source','')} page:{m.get('page','')}")
            print("    ", r.get("document")[:300].replace("\n"," "))
        if res["images"]:
            print("\n--- Image contexts ---")
            for im in res["images"]:
                print(im["meta"].get("img_name"), "=>", im.get("caption"))
        if memory_agent:
            print("\n--- Recent memory (last turns) ---")
            print(memory_agent.get_context_text(8))
        print("\n")

if __name__ == "__main__":
    main()
