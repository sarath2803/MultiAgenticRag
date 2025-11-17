# #main 2

# # main.py
# import os
# from agents.intent_agent import IntentAgent
# from agents.retrieval_agent import RetrievalAgent
# from agents.vision_agent import VisionAgent
# from agents.reasoning_agent import ReasoningAgent
# from agents.controller_agent import ControllerAgent
# # from agents.feedback_agent import FeedbackAgent


# def main():
#     print("🚀 Starting Vision-Enhanced Intent-Aware Multi-Agent RAG Assistant")
#     print("➡ Make sure OPENAI_API_KEY is set in your environment.\n")

#     # ---------------------------
#     # Initialize Agents
#     # ---------------------------
#     intent_agent = IntentAgent()
#     retrieval_agent = RetrievalAgent()
#     vision_agent = VisionAgent()
#     reasoning_agent = ReasoningAgent()
#     # feedback_agent = FeedbackAgent()

#     controller = ControllerAgent(
#         intent_agent=intent_agent,
#         retrieval_agent=retrieval_agent,
#         vision_agent=vision_agent,
#         reasoning_agent=reasoning_agent,
#         # feedback_agent=feedback_agent
#     )

#     # ---------------------------
#     # Data Ingestion
#     # ---------------------------
#     print("📥 Ingesting knowledge base from 'data/' (PDFs + images)...")

#     retrieval_agent.ingest_pdfs(data_dir="data")
#     retrieval_agent.ingest_images_folder(images_dir=os.path.join("data", "images"))

#     print("✅ Ready. Type your queries (or 'exit'):\n")

#     # ---------------------------
#     # Main Loop
#     # ---------------------------
#     while True:
#         user_query = input("> ").strip()
#         if not user_query:
#             continue

#         if user_query.lower() in ("exit", "quit"):
#             print("Shutting down assistant...")
#             break

#         # ---------------------------
#         # Controller Pipeline
#         # ---------------------------
#         result = controller.handle_query(user_query, top_k=5)

#         # ---------------------------
#         # Display Output
#         # ---------------------------
#         print("\n==============================")
#         print("🤖 INTENT:", result["intent"])
#         print("==============================\n")

#         print(result["answer"])

#         # ---------------------------
#         # Retrieved text chunks
#         # ---------------------------
#         print("\n--- Retrieved Documents (Top Results) ---")
#         for i, item in enumerate(result["retrieved"], start=1):
#             meta = item.get("metadata", {})
#             print(f"{i}. ({meta.get('type','')}) Source: {meta.get('source','')} Page: {meta.get('page','')}")
#             print("    ", item.get("document", "")[:300].replace("\n", " "))

#         # ---------------------------
#         # Image Contexts (Image Captioning)
#         # ---------------------------
#         if result.get("images"):
#             print("\n--- Image Contexts (Detected Images + Captions) ---")
#             for img in result["images"]:
#                 print(f"Image: {img['meta'].get('img_name')}  →  Caption: {img['caption']}")

#         # ---------------------------
#         # Feedback Evaluation (Optional)
#         # ---------------------------
#         # if "evaluation" in result:
#         #     print("\n--- Evaluation (LLM Confidence) ---")
#         #     print(result["evaluation"])

#         print("\n")


# if __name__ == "__main__":
#     main()

# import os
# from agents.intent_agent import IntentAgent
# from agents.retrieval_agent import RetrievalAgent
# from agents.vision_agent import VisionAgent
# from agents.reasoning_agent import ReasoningAgent
# from agents.controller_agent import ControllerAgent

# def main():
#     print("🚀 Starting Vision-Enhanced Intent-Aware Multi-Agent RAG Assistant")


#     # Initialize agents
#     intent_agent = IntentAgent()
#     retrieval_agent = RetrievalAgent()
#     vision_agent = VisionAgent()
#     reasoning_agent = ReasoningAgent()

#     controller = ControllerAgent(
#         intent_agent=intent_agent,
#         retrieval_agent=retrieval_agent,
#         vision_agent=vision_agent,
#         reasoning_agent=reasoning_agent
#     )

#     # Data ingestion
#     print("📥 Ingesting knowledge base from 'data/' (PDFs + images)...")
#     retrieval_agent.ingest_pdfs(data_dir="data")
#     retrieval_agent.ingest_images_folder(images_dir=os.path.join("data", "images"))
#     print("✅ Ready. Type your queries (or 'exit'):\n")

#     while True:
#         user_query = input("> ").strip()
#         if not user_query:
#             continue
#         if user_query.lower() in ("exit", "quit"):
#             print("Shutting down assistant...")
#             break

#         result = controller.handle_query(user_query, top_k=5)

#         print("\n==============================")
#         print("🤖 INTENT:", result["intent"])
#         print("==============================\n")
#         print(result["answer"])

#         print("\n--- Retrieved Documents (Top Results) ---")
#         for i, item in enumerate(result["retrieved"], start=1):
#             meta = item.get("metadata", {})
#             print(f"{i}. ({meta.get('type','')}) Source: {meta.get('source','')} Page: {meta.get('page','')}")
#             print("    ", item.get("document", "")[:300].replace("\n", " "))

#         if result.get("images"):
#             print("\n--- Image Contexts (Captions + OCR) ---")
#             for img in result["images"]:
#                 print(f"Image: {img['meta'].get('img_name')}  →  Caption+OCR: {img['caption']}")

#         print("\n")

# if __name__ == "__main__":
#     main()

#memory updated

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

    # ingest as before (wrapped in try/except if you prefer)
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
