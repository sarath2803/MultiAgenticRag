# main.py
import os
from agents.intent_agent import IntentAgent
from agents.retrieval_agent import RetrievalAgent
from agents.vision_agent import VisionAgent
from agents.reasoning_agent import ReasoningAgent
from agents.controller_agent import ControllerAgent

def main():
    print("Starting Vision-Enhanced Intent-Aware Multi-Agent RAG Assistant")
    print("Make sure OPENAI_API_KEY is set in env to use LLM reasoning.\n")

    intent_agent = IntentAgent()
    retrieval_agent = RetrievalAgent()
    vision_agent = VisionAgent()
    reasoning_agent = ReasoningAgent()
    controller = ControllerAgent(intent_agent, retrieval_agent, vision_agent, reasoning_agent)

    print("Ingesting data from 'data/' (PDFs + data/images)...")  #repetative
    retrieval_agent.ingest_pdfs(data_dir="data")
    retrieval_agent.ingest_images_folder(images_dir=os.path.join("data","images"))

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
                print(im["meta"].get("img_name"), "=>", im["caption"])
        print("\n")

if __name__ == "__main__":
    main()
