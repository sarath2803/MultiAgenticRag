# #reasoning 2

# # agents/reasoning_agent.py
# import os
# from typing import List, Dict, Any
# from llama_cpp import Llama


# class ReasoningAgent:
#     def __init__(self, model_path="models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Mistral model not found at {model_path}")
#         print(f"Loading Mistral model from: {model_path}")

#         # Load Mistral model using GPU if available
#         self.llm = Llama(
#             model_path=model_path,
#             n_ctx=4096,
#             n_threads=8,
#             n_gpu_layers=-1,
#             use_mmap=True,
#             use_mlock=False
#         )

#     # ---------------------------------------------------------
#     # Generic prompt builder for any intent (kept from your code)
#     # ---------------------------------------------------------
#     def _build_prompt(
#         self,
#         query: str,
#         retrieved: List[Dict[str, Any]],
#         image_context: List[Dict[str, Any]] = None,
#         intent: str = None
#     ) -> str:

#         intent_instructions = {
#             "fact": "Answer the question using factual details from the context. Be direct and concise.",
#             "analysis": "Provide a structured, deep analytical explanation using evidence.",
#             "summary": "Summarize the main ideas clearly and concisely.",
#             "visual": "Combine both text and image descriptions to answer the question."
#         }

#         header = (
#             "You are a helpful AI research assistant.\n"
#             f"Task: {intent_instructions.get(intent, 'Answer using the given context.')}\n\n"
#         )

#         body = f"User Query: {query}\n\nContext:\n"
#         for i, r in enumerate(retrieved):
#             meta = r.get("metadata", {})
#             src = meta.get("source", "")
#             page = meta.get("page", "")
#             snippet = r.get("document", "")[:800].replace("\n", " ")
#             body += f"{i+1}. [Source: {src}, Page {page}] {snippet}\n"

#         if image_context:
#             body += "\nImage Descriptions:\n"
#             for im in image_context:
#                 body += f"- {im.get('caption', '')} (from {im.get('meta', {}).get('img_name', '')})\n"

#         body += "\nYour Response:\n"
#         return header + body

#     # ---------------------------------------------------------
#     # FACT INTENT
#     # ---------------------------------------------------------
#     def answer(self, query: str, docs: List[Dict[str, Any]], max_tokens: int = 512) -> str:
#         prompt = self._build_prompt(query, docs, intent="fact")
#         return self._run_llm(prompt, max_tokens)

#     # ---------------------------------------------------------
#     # ANALYSIS INTENT
#     # ---------------------------------------------------------
#     def reason(self, query: str, docs: List[Dict[str, Any]], max_tokens: int = 768) -> str:
#         prompt = self._build_prompt(query, docs, intent="analysis")
#         return self._run_llm(prompt, max_tokens)

#     # ---------------------------------------------------------
#     # SUMMARY INTENT
#     # ---------------------------------------------------------
#     def summarize(self, all_docs: List[Dict[str, Any]], max_tokens: int = 768) -> str:
#         prompt = self._build_prompt("Summarize the following documents", all_docs, intent="summary")
#         return self._run_llm(prompt, max_tokens)

#     # ---------------------------------------------------------
#     # VISUAL INTENT
#     # ---------------------------------------------------------
#     def answer_with_visual(
#         self,
#         query: str,
#         text_context: List[Dict[str, Any]],
#         image_context: List[Dict[str, Any]],
#         max_tokens: int = 768
#     ) -> str:
#         prompt = self._build_prompt(query, text_context, image_context, intent="visual")
#         return self._run_llm(prompt, max_tokens)

#     # ---------------------------------------------------------
#     # INTERNAL LLM CALL
#     # ---------------------------------------------------------
#     def _run_llm(self, prompt: str, max_tokens: int) -> str:
#         try:
#             output = self.llm.create_completion(
#                 prompt=prompt,
#                 max_tokens=max_tokens,
#                 temperature=0.3,
#                 top_p=0.9
#             )
#             return output["choices"][0]["text"].strip()

#         except Exception as e:
#             return f"[Error in Mistral reasoning: {e}]"

#Integrated OCR

# import os
# from typing import List, Dict, Any
# from llama_cpp import Llama

# class ReasoningAgent:
#     def __init__(self, model_path="models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Mistral model not found at {model_path}")
#         print(f"Loading Mistral model from: {model_path}")

#         self.llm = Llama(
#             model_path=model_path,
#             n_ctx=4096,
#             n_threads=8,
#             n_gpu_layers=-1,
#             use_mmap=True,
#             use_mlock=False
#         )

#     def _build_prompt(
#         self,
#         query: str,
#         retrieved: List[Dict[str, Any]],
#         image_context: List[Dict[str, Any]] = None,
#         intent: str = None
#     ) -> str:

#         intent_instructions = {
#             "fact": "Answer the question using factual details from the context. Be direct and concise.",
#             "analysis": "Provide a structured, deep analytical explanation using evidence.",
#             "summary": "Summarize the main ideas clearly and concisely.",
#             "visual": "Combine both text and image descriptions (including OCR text) to answer the question."
#         }

#         header = (
#             "You are a helpful AI research assistant.\n"
#             f"Task: {intent_instructions.get(intent, 'Answer using the given context.')}\n\n"
#         )

#         body = f"User Query: {query}\n\nContext:\n"
#         for i, r in enumerate(retrieved):
#             meta = r.get("metadata", {})
#             src = meta.get("source", "")
#             page = meta.get("page", "")
#             snippet = r.get("document", "")[:800].replace("\n", " ")
#             body += f"{i+1}. [Source: {src}, Page {page}] {snippet}\n"

#         if image_context:
#             body += "\nImage Descriptions + OCR Text:\n"
#             for im in image_context:
#                 caption = im.get('caption', '')
#                 ocr = im.get('ocr_text', '')
#                 name = im.get('meta', {}).get('img_name', '')
#                 if ocr.strip():
#                     body += f"- {caption} | OCR: {ocr} (from {name})\n"
#                 else:
#                     body += f"- {caption} (from {name})\n"

#         body += "\nYour Response:\n"
#         return header + body

#     # FACT INTENT
#     def answer(self, query: str, docs: List[Dict[str, Any]], max_tokens: int = 512) -> str:
#         prompt = self._build_prompt(query, docs, intent="fact")
#         return self._run_llm(prompt, max_tokens)

#     # ANALYSIS INTENT
#     def reason(self, query: str, docs: List[Dict[str, Any]], max_tokens: int = 768) -> str:
#         prompt = self._build_prompt(query, docs, intent="analysis")
#         return self._run_llm(prompt, max_tokens)

#     # SUMMARY INTENT
#     def summarize(self, all_docs: List[Dict[str, Any]], max_tokens: int = 768) -> str:
#         prompt = self._build_prompt("Summarize the following documents", all_docs, intent="summary")
#         return self._run_llm(prompt, max_tokens)

#     # VISUAL INTENT
#     def answer_with_visual(
#         self,
#         query: str,
#         text_context: List[Dict[str, Any]],
#         image_context: List[Dict[str, Any]],
#         max_tokens: int = 768
#     ) -> str:
#         prompt = self._build_prompt(query, text_context, image_context, intent="visual")
#         return self._run_llm(prompt, max_tokens)

#     # INTERNAL LLM CALL
#     def _run_llm(self, prompt: str, max_tokens: int) -> str:
#         try:
#             output = self.llm.create_completion(
#                 prompt=prompt,
#                 max_tokens=max_tokens,
#                 temperature=0.3,
#                 top_p=0.9
#             )
#             return output["choices"][0]["text"].strip()
#         except Exception as e:
#             return f"[Error in Mistral reasoning: {e}]"


#memory updated

# agents/reasoning_agent.py
import os
from typing import List, Dict, Any
from llama_cpp import Llama

MAX_SNIPPET_CHARS = 350
MAX_DOCS = 5
MAX_IMAGE_ITEMS = 3

class ReasoningAgent:
    def __init__(self, model_path="models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Mistral model not found at {model_path}")
        print(f"Loading Mistral model from: {model_path}")

        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,
            use_mmap=True,
            use_mlock=False
        )

    def _trim_docs(self, docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trimmed = []
        for d in (docs or [])[:MAX_DOCS]:
            new_doc = d.copy()
            new_doc["document"] = new_doc.get("document", "")[:MAX_SNIPPET_CHARS]
            trimmed.append(new_doc)
        return trimmed

    def _trim_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        trimmed = []
        for im in (images or [])[:MAX_IMAGE_ITEMS]:
            new_im = im.copy()
            if "ocr_text" in new_im:
                new_im["ocr_text"] = new_im["ocr_text"][:MAX_SNIPPET_CHARS]
            if "caption" in new_im:
                new_im["caption"] = new_im["caption"][:MAX_SNIPPET_CHARS]
            trimmed.append(new_im)
        return trimmed

    def _normalize_memory(self, memory: Any) -> str:
        """
        Ensure memory is a safe string for prompt injection.
        Accepts: None, str, list, dict. Returns string (possibly empty).
        """
        if not memory:
            return ""
        # If memory is already a string
        if isinstance(memory, str):
            return memory.strip()
        # If memory is list: join items (handle dict items)
        if isinstance(memory, list):
            out_lines = []
            for m in memory:
                if isinstance(m, dict):
                    role = m.get("role", "")
                    text = m.get("text", "")
                    if role:
                        out_lines.append(f"{role.capitalize()}: {text}")
                    else:
                        out_lines.append(str(text))
                else:
                    out_lines.append(str(m))
            return "\n".join(out_lines).strip()
        # If memory is dict: stringify keys/values
        if isinstance(memory, dict):
            lines = []
            for k, v in memory.items():
                lines.append(f"{k}: {v}")
            return "\n".join(lines).strip()
        # fallback
        return str(memory).strip()

    def _build_prompt(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        image_context: List[Dict[str, Any]] = None,
        intent: str = None,
        memory: Any = None
    ) -> str:

        # Normalize memory to string first
        memory_text = self._normalize_memory(memory)

        # Trim retrieved and images
        retrieved = self._trim_docs(retrieved or [])
        image_context = self._trim_images(image_context or [])

        intent_instructions = {
            "fact": "Answer the question using factual details from the context.",
            "analysis": "Provide a structured, detailed analysis using evidence.",
            "summary": "Summarize clearly and concisely.",
            "visual": "Use both text and image OCR/captions to answer."
        }

        header = (
            "You are a helpful AI assistant.\n"
            f"Task: {intent_instructions.get(intent, 'Answer using context.')}\n\n"
        )

        memory_section = ""
        if memory_text:
            memory_section = f"=== Conversation Memory (recent & summary) ===\n{memory_text}\n\n"

        body = f"{memory_section}User Query: {query}\n\nContext:\n"

        for i, r in enumerate(retrieved):
            meta = r.get("metadata", {})
            src = meta.get("source", "")
            page = meta.get("page", "")
            snippet = r.get("document", "").replace("\n", " ")
            body += f"{i+1}. [Source: {src}, Page {page}] {snippet}\n"

        if image_context:
            body += "\nImage Context:\n"
            for im in image_context:
                caption = im.get("caption", "")
                ocr = im.get("ocr_text", "")
                name = im.get("meta", {}).get("img_name", "")
                body += f"- {caption} | OCR: {ocr} (from {name})\n"

        body += "\nYour Response:\n"
        return header + body

    def _run_llm(self, prompt: str, max_tokens: int) -> str:
        try:
            output = self.llm.create_completion(
                prompt=prompt,
                max_tokens=max_tokens,
                temperature=0.3,
                top_p=0.9
            )
            return output["choices"][0]["text"].strip()
        except Exception as e:
            return f"[Error in Mistral reasoning: {e}]"

    def answer(self, query: str, docs: List[Dict[str, Any]], memory: Any = None, max_tokens: int = 512) -> str:
        prompt = self._build_prompt(query, docs, intent="fact", memory=memory)
        return self._run_llm(prompt, max_tokens)

    def reason(self, query: str, docs: List[Dict[str, Any]], memory: Any = None, max_tokens: int = 700) -> str:
        prompt = self._build_prompt(query, docs, intent="analysis", memory=memory)
        return self._run_llm(prompt, max_tokens)

    def summarize(self, docs: List[Dict[str, Any]], memory: Any = None, max_tokens: int = 700) -> str:
        prompt = self._build_prompt("Summarize the following documents", docs, intent="summary", memory=memory)
        return self._run_llm(prompt, max_tokens)

    def answer_with_visual(self, query: str, text_context: List[Dict[str, Any]], image_context: List[Dict[str, Any]], memory: Any = None, max_tokens: int = 700) -> str:
        prompt = self._build_prompt(query, text_context, image_context, intent="visual", memory=memory)
        return self._run_llm(prompt, max_tokens)
