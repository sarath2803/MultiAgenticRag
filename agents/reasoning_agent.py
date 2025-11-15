# # agents/reasoning_agent.py
# import os
# import openai
# from typing import List, Dict, Any

# OPENAI_KEY = os.environ.get("OPENAI_API_KEY", None)

# class ReasoningAgent:
#     def __init__(self, model_name="gpt-4o-mini", openai_api_key=OPENAI_KEY):
#         self.openai_key = openai_api_key
#         if self.openai_key:
#             openai.api_key = self.openai_key
#         self.model_name = model_name

#     def _build_prompt(self, query: str, retrieved: List[Dict[str,Any]], image_context: List[Dict[str,Any]] = None, intent: str = None) -> str:
#         header = "You are a helpful assistant. Use the provided document excerpts and image descriptions to answer the user's query. Cite sources where possible.\n\n"
#         body = f"User query: {query}\n\n"
#         body += "Textual context (top results):\n"
#         for i, r in enumerate(retrieved):
#             meta = r.get("metadata", {})
#             src = meta.get("source", "")
#             page = meta.get("page", "")
#             snippet = r.get("document", "")[:600].replace("\n", " ")
#             body += f"{i+1}. Source: {src} page:{page}\n{snippet}\n\n"

#         if image_context:
#             body += "\nImage descriptions:\n"
#             for i, im in enumerate(image_context):
#                 body += f"{i+1}. {im.get('caption','')} (source: {im.get('meta',{}).get('img_name','')})\n"

#         body += "\nAnswer (be concise; if summary intent, use bullets; if visual intent, describe the image and reference text):\n"
#         return header + body

#     def answer(self, query: str, retrieved: List[Dict[str,Any]], image_context: List[Dict[str,Any]] = None, intent: str = None, max_tokens: int = 512) -> str:
#         prompt = self._build_prompt(query, retrieved, image_context, intent)
#         if self.openai_key:
#             try:
#                 resp = openai.ChatCompletion.create(
#                     model="gpt-4o-mini" if "gpt-4o-mini" else "gpt-4",
#                     messages=[
#                         {"role":"system", "content":"You are a helpful multimodal RAG assistant."},
#                         {"role":"user", "content": prompt}
#                     ],
#                     max_tokens=max_tokens,
#                     temperature=0.1
#                 )
#                 return resp["choices"][0]["message"]["content"].strip()
#             except Exception as e:
#                 print("OpenAI request failed:", e)
#                 return "Error: OpenAI request failed. " + str(e)
#         else:
#             # fallback: return prompt
#             return "OPENAI_API_KEY not set. Prompt:\n\n" + prompt[:4000]

# agents/reasoning_agent.py
# import os
# from typing import List, Dict, Any
# from llama_cpp import Llama

# class ReasoningAgent:
#     def __init__(self, model_path="models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
#         if not os.path.exists(model_path):
#             raise FileNotFoundError(f"Mistral model not found at {model_path}")
#         print(f"Loading Mistral model from: {model_path}")
#         # Automatically use GPU if available
#         self.llm = Llama(
#             model_path=model_path,
#             n_ctx=4096,
#             n_threads=8,
#             n_gpu_layers=-1,  # Offload as much as possible to GPU
#             use_mmap=True,
#             use_mlock=False
#         )

#     def _build_prompt(self, query: str, retrieved: List[Dict[str,Any]], image_context: List[Dict[str,Any]] = None, intent: str = None) -> str:
#         header = (
#             "You are a helpful assistant. Use the following context (text + image descriptions) "
#             "to answer the user's question accurately and concisely.\n\n"
#         )
#         body = f"User query: {query}\n\nContext:\n"
#         for i, r in enumerate(retrieved):
#             meta = r.get("metadata", {})
#             src = meta.get("source", "")
#             page = meta.get("page", "")
#             snippet = r.get("document", "")[:800].replace("\n", " ")
#             body += f"{i+1}. Source: {src}, page {page}: {snippet}\n"
#         if image_context:
#             body += "\nImage descriptions:\n"
#             for im in image_context:
#                 body += f"- {im.get('caption','')} (from {im.get('meta',{}).get('img_name','')})\n"
#         body += "\nAnswer:\n"
#         return header + body

#     def answer(self, query: str, retrieved: List[Dict[str,Any]], image_context: List[Dict[str,Any]] = None, intent: str = None, max_tokens: int = 512) -> str:
#         prompt = self._build_prompt(query, retrieved, image_context, intent)
#         try:
#             output = self.llm.create_completion(
#                 prompt=prompt,
#                 max_tokens=max_tokens,
#                 temperature=0.2,
#                 top_p=0.9
#             )
#             return output["choices"][0]["text"].strip()
#         except Exception as e:
#             return f"[Error in Mistral reasoning: {e}]"


# agents/reasoning_agent.py
import os
from typing import List, Dict, Any
from llama_cpp import Llama

class ReasoningAgent:
    def __init__(self, model_path="models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf"):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Mistral model not found at {model_path}")
        print(f"Loading Mistral model from: {model_path}")

        # Load Mistral model using GPU if available
        self.llm = Llama(
            model_path=model_path,
            n_ctx=4096,
            n_threads=8,
            n_gpu_layers=-1,  # Offload as much as possible to GPU
            use_mmap=True,
            use_mlock=False
        )

    def _build_prompt(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        image_context: List[Dict[str, Any]] = None,
        intent: str = None
    ) -> str:
        """Builds an intent-aware prompt."""

        # Define intent-specific instructions
        intent_instructions = {
            "fact": "Answer the question using factual details from the context. Be direct and concise.",
            "analysis": "Provide an analytical and comparative explanation using evidence from the documents.",
            "summary": "Summarize the main points and findings from the retrieved context clearly and concisely.",
            "visual": "Describe and interpret the charts, figures, or images in the context along with supporting text."
        }

        header = (
            "You are a helpful AI research assistant.\n"
            f"Task: {intent_instructions.get(intent, 'Answer appropriately based on the provided context.')}\n\n"
        )

        body = f"User Query: {query}\n\nContext:\n"
        for i, r in enumerate(retrieved):
            meta = r.get("metadata", {})
            src = meta.get("source", "")
            page = meta.get("page", "")
            snippet = r.get("document", "")[:800].replace("\n", " ")
            body += f"{i+1}. [Source: {src}, Page {page}] {snippet}\n"

        if image_context:
            body += "\nImage Descriptions:\n"
            for im in image_context:
                body += f"- {im.get('caption','')} (from {im.get('meta',{}).get('img_name','')})\n"

        body += "\nYour Response:\n"
        return header + body

    def answer(
        self,
        query: str,
        retrieved: List[Dict[str, Any]],
        image_context: List[Dict[str, Any]] = None,
        intent: str = None,
        max_tokens: int = 512
    ) -> str:
        """Generates an intent-aware answer using the Mistral model."""
        prompt = self._build_prompt(query, retrieved, image_context, intent)

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
