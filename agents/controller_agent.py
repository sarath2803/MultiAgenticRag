

# # agents/controller_agent.py
# from typing import List, Dict, Any

# class ControllerAgent:
#     def __init__(self, intent_agent, retrieval_agent, vision_agent, reasoning_agent):
#         self.intent_agent = intent_agent
#         self.retrieval_agent = retrieval_agent
#         self.vision_agent = vision_agent
#         self.reasoning_agent = reasoning_agent
#         #self.feedback_agent = feedback_agent

#     def handle_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
#         intent = self.intent_agent.predict(query)
#         print(f"[Controller] Intent: {intent}")

#         #retrieved = self.retrieval_agent.retrieve(query, top_k=top_k)
#         image_context = []
#         if intent == "fact":
#             docs = self.retrieval_agent.retrieve(query, top_k=top_k)
#             answer = self.reasoning_agent.answer(query,docs)

#         elif intent == "analysis":
#             docs = self.retrieval_agent.retrieve_multi_round(query)
#             answer = self.reasoning_agent.reason(query, docs)
#         elif intent == "summary":
#             all_docs = self.retrieval_agent.get_all_docs()
#             answer = self.reasoning_agent.summarize(all_docs)
#             doc=all_docs[:5]
#         elif intent == "visual":
#             image_context = self.vision_agent.describe_related_images(query)
#             text_context = self.retrieval_agent.retrieve(query)
#             answer = self.reasoning_agent.answer_with_visual(query, text_context, image_context)



#         return {
#             "intent": intent,
#             "answer": answer,
#             "retrieved": docs if intent in ("fact", "analysis","summary","visual") else [],
#             "images": image_context,
#             #"evaluation": feedback,
#         }

#Integrated OCR

# from typing import List, Dict, Any

# class ControllerAgent:
#     def __init__(self, intent_agent, retrieval_agent, vision_agent, reasoning_agent):
#         self.intent_agent = intent_agent
#         self.retrieval_agent = retrieval_agent
#         self.vision_agent = vision_agent
#         self.reasoning_agent = reasoning_agent

#     def handle_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
#         intent = self.intent_agent.predict(query)
#         print(f"[Controller] Intent: {intent}")

#         image_context = []
#         docs = []

#         if intent == "fact":
#             docs = self.retrieval_agent.retrieve(query, top_k=top_k)
#             answer = self.reasoning_agent.answer(query, docs)

#         elif intent == "analysis":
#             docs = self.retrieval_agent.retrieve_multi_round(query)
#             answer = self.reasoning_agent.reason(query, docs)

#         elif intent == "summary":
#             all_docs = self.retrieval_agent.get_all_docs()
#             answer = self.reasoning_agent.summarize(all_docs)
#             docs = all_docs[:5]

#         elif intent == "visual":
#             # Retrieve images
#             image_results = self.retrieval_agent.retrieve(query, top_k=top_k, intent="visual")
#             image_context = self.retrieval_agent.describe_related_images(query, image_results)

#             # Retrieve text context
#             text_context = self.retrieval_agent.retrieve(query)

#             # Merge OCR text from images into text context
#             combined_context = text_context.copy()
#             for img in image_context:
#                 if "ocr_text" in img and img["ocr_text"].strip():
#                     combined_context.append({"document": img["ocr_text"], "metadata": {"type": "ocr_text"}})

#             answer = self.reasoning_agent.answer_with_visual(query, combined_context, image_context)

#         return {
#             "intent": intent,
#             "answer": answer,
#             "retrieved": docs,
#             "images": image_context
#         }


#memory updated

# agents/controller_agent.py
from typing import List, Dict, Any, Optional

class ControllerAgent:
    def __init__(
        self,
        intent_agent,
        retrieval_agent,
        vision_agent,
        reasoning_agent,
        memory_agent: Optional[Any] = None
    ):
        self.intent_agent = intent_agent
        self.retrieval_agent = retrieval_agent
        self.vision_agent = vision_agent
        self.reasoning_agent = reasoning_agent
        self.memory = memory_agent

    def handle_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        # --- prepare memory text for prompt injection (always a string) ---
        memory_text = ""
        if self.memory:
            try:
                memory_text = self.memory.get_relevant_memory(query, n=6)  # string
            except Exception:
                # defensive: fallback to empty string
                memory_text = ""

        # --- detect intent (preserve existing behavior) ---
        try:
            # if intent agent supports (text, context) signature, it may accept memory; we try simple call first
            intent = self.intent_agent.predict(query)
        except TypeError:
            # fallback: if IntentAgent has a memory-aware signature
            try:
                intent = self.intent_agent.predict(query, conversation_context=memory_text)
            except Exception:
                intent = self.intent_agent.predict(query)

        print(f"[Controller] Intent: {intent}")

        image_context: List[Dict[str, Any]] = []
        docs: List[Dict[str, Any]] = []
        answer = ""

        # FACT
        if intent == "fact":
            docs = self.retrieval_agent.retrieve(query, top_k=top_k)
            answer = self.reasoning_agent.answer(query, docs, memory=memory_text)

        # ANALYSIS
        elif intent == "analysis":
            try:
                docs = self.retrieval_agent.retrieve_multi_round(query)
            except Exception:
                docs = self.retrieval_agent.retrieve(query, top_k=top_k)
            answer = self.reasoning_agent.reason(query, docs, memory=memory_text)

        # SUMMARY
        elif intent == "summary":
            try:
                all_docs = self.retrieval_agent.get_all_docs()
            except Exception:
                all_docs = self.retrieval_agent.get_all_documents()
            answer = self.reasoning_agent.summarize(all_docs, memory=memory_text)
            docs = all_docs[:5] if isinstance(all_docs, list) else []

            # update long-term memory with the summary (optional)
            if self.memory:
                try:
                    self.memory.update_long_term_summary(answer)
                except Exception:
                    pass

        # VISUAL
        elif intent == "visual":
            image_results = self.retrieval_agent.retrieve(query, top_k=top_k, intent="visual")
            image_context = self.retrieval_agent.describe_related_images(query, image_results)

            # retrieve supporting text
            text_context = self.retrieval_agent.retrieve(query, top_k=top_k)
            combined_context = text_context.copy() if isinstance(text_context, list) else text_context

            for img in image_context:
                if "ocr_text" in img and img["ocr_text"].strip():
                    combined_context.append({"document": img["ocr_text"], "metadata": {"type": "ocr_text"}})

            answer = self.reasoning_agent.answer_with_visual(query, combined_context, image_context, memory=memory_text)

        else:
            # fallback: treat as fact query
            docs = self.retrieval_agent.retrieve(query, top_k=top_k)
            answer = self.reasoning_agent.answer(query, docs, memory=memory_text)

        # --- store turn into memory safely ---
        if self.memory:
            try:
                self.memory.store(query, answer)
            except Exception:
                try:
                    self.memory.add_user(query)
                    self.memory.add_assistant(answer)
                except Exception:
                    pass

        return {
            "intent": intent,
            "answer": answer,
            "retrieved": docs,
            "images": image_context,
            "memory_used": memory_text
        }
