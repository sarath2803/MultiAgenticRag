


# agents/controller_agent.py
from typing import List, Dict, Any

class ControllerAgent:
    def __init__(self, intent_agent, retrieval_agent, vision_agent, reasoning_agent):
        self.intent_agent = intent_agent
        self.retrieval_agent = retrieval_agent
        self.vision_agent = vision_agent
        self.reasoning_agent = reasoning_agent

    def handle_query(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        intent = self.intent_agent.predict(query)
        print(f"[Controller] Intent: {intent}")
        retrieved = self.retrieval_agent.retrieve(query, top_k=top_k)

        image_context = []
        # If intent is visual or any retrieved item is an image, describe relevant images
        if intent == "visual" or any(r.get("metadata", {}).get("type") == "image" for r in retrieved):
            for r in retrieved:
                meta = r.get("metadata", {})
                if meta.get("type") == "image":
                    img_path = meta.get("img_path")
                    if img_path:
                        desc = self.vision_agent.describe_image(img_path)
                        image_context.append({"caption": desc["caption"], "meta": meta, "id": r.get("id")})
                        # update image embedding in retrieval DB using caption
                        try:
                            self.retrieval_agent.update_image_caption_embedding(r.get("id"), desc["caption"])
                        except Exception:
                            pass

        # Compose answer via LLM
        answer = self.reasoning_agent.answer(query, retrieved, image_context, intent)
        return {"intent": intent, "answer": answer, "retrieved": retrieved, "images": image_context}
