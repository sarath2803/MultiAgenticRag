

# # agents/retrieval_agent.py
# import os
# import uuid
# from typing import List, Dict, Any
# import fitz  # PyMuPDF
# from sentence_transformers import SentenceTransformer
# import chromadb
# import numpy as np

# CHROMA_DIR = "chroma_db"

# class RetrievalAgent:
#     def __init__(self, embed_model_name="all-MiniLM-L6-v2", persist_directory=CHROMA_DIR):
#         import torch
#         device = "cuda" if torch.cuda.is_available() else "cpu"
#         self.device = device
#         self.text_embedder = SentenceTransformer(embed_model_name, device=self.device)

#         # ✅ NEW: Chroma 0.5+ compatible client initialization
#         self.client = chromadb.PersistentClient(path=persist_directory)

#         # Create or get collection
#         coll_name = "multimodal_docs"
#         existing_collections = [c.name for c in self.client.list_collections()]
#         if coll_name in existing_collections:
#             self.col = self.client.get_collection(coll_name)
#         else:
#             self.col = self.client.create_collection(
#                 name=coll_name,
#                 metadata={"description": "text+image multimodal"}
#             )

#     def ingest_pdfs(self, data_dir="data", chunk_chars=900, overlap=200):
#         for root, _, files in os.walk(data_dir):
#             for fn in files:
#                 if fn.lower().endswith(".pdf"):
#                     path = os.path.join(root, fn)
#                     self._process_pdf(path, chunk_chars, overlap)
#         print("✅ PDF ingestion complete. Data persisted to:", os.path.abspath(CHROMA_DIR))

#     def _process_pdf(self, path, chunk_chars, overlap):
#         doc = fitz.open(path)
#         for pno in range(len(doc)):
#             page = doc[pno]
#             text = page.get_text().strip()

#             # 🔹 Process text chunks
#             if text:
#                 chunks = self._chunk_text(text, chunk_chars, overlap)
#                 for i, c in enumerate(chunks):
#                     emb = self.text_embedder.encode(c)
#                     uid = str(uuid.uuid4())
#                     meta = {
#                         "type": "text",
#                         "source": os.path.basename(path),
#                         "page": pno + 1,
#                         "chunk_idx": i
#                     }
#                     self.col.add(
#                         documents=[c],
#                         metadatas=[meta],
#                         ids=[uid],
#                         embeddings=[emb.tolist()]
#                     )

#             # 🔹 Extract and store images
#             images = page.get_images(full=True)
#             for img_index, img in enumerate(images):
#                 xref = img[0]
#                 base_image = doc.extract_image(xref)
#                 ext = base_image["ext"]
#                 image_bytes = base_image["image"]

#                 images_dir = os.path.join(os.path.dirname(path), "images")
#                 os.makedirs(images_dir, exist_ok=True)
#                 img_name = f"{os.path.splitext(os.path.basename(path))[0]}_p{pno + 1}_img{img_index}.{ext}"
#                 img_path = os.path.join(images_dir, img_name)

#                 with open(img_path, "wb") as f:
#                     f.write(image_bytes)

#                 placeholder_text = f"[IMAGE] {img_name}"
#                 emb = self.text_embedder.encode(f"image:{img_name}")
#                 uid = str(uuid.uuid4())
#                 meta = {
#                     "type": "image",
#                     "source": os.path.basename(path),
#                     "page": pno + 1,
#                     "img_path": img_path,
#                     "img_name": img_name
#                 }
#                 self.col.add(
#                     documents=[placeholder_text],
#                     metadatas=[meta],
#                     ids=[uid],
#                     embeddings=[emb.tolist()]
#                 )

#     def _chunk_text(self, text: str, chunk_chars: int, overlap: int) -> List[str]:
#         """Split long text into overlapping chunks"""
#         chunks = []
#         start = 0
#         while start < len(text):
#             end = min(start + chunk_chars, len(text))
#             chunk = text[start:end]
#             chunks.append(chunk)
#             start += chunk_chars - overlap
#         return chunks

#     def ingest_images_folder(self, images_dir="data/images"):
#         if not os.path.exists(images_dir):
#             print("⚠️ No image directory found:", images_dir)
#             return

#         for root, _, files in os.walk(images_dir):
#             for fn in files:
#                 if fn.lower().endswith((".png", ".jpg", ".jpeg")):
#                     p = os.path.join(root, fn)
#                     placeholder = f"[IMAGE] {fn}"
#                     emb = self.text_embedder.encode(f"image:{fn}")
#                     uid = str(uuid.uuid4())
#                     meta = {
#                         "type": "image",
#                         "source": fn,
#                         "img_path": p,
#                         "img_name": fn
#                     }
#                     self.col.add(
#                         documents=[placeholder],
#                         metadatas=[meta],
#                         ids=[uid],
#                         embeddings=[emb.tolist()]
#                     )
#         print("✅ Image ingestion complete.")


#     def retrieve(self, query: str, top_k: int = 5, intent: str = "default") -> List[Dict[str, Any]]:
#             q_emb = self.text_embedder.encode(query)

#     # 🔍 If intent is visual → retrieve only image embeddings
#             if intent == "visual":
#                 res = self.col.query(
#                     query_embeddings=[q_emb.tolist()],
#                     n_results=top_k,
#                     where={"type": "image"}   # 🔥 Only return images
#                 )
#             else:
#                 # Normal retrieval
#                 res = self.col.query(
#                     query_embeddings=[q_emb.tolist()],
#                     n_results=top_k
#                 )

#             hits = []
#             docs = res.get("documents", [[]])[0]
#             metas = res.get("metadatas", [[]])[0]
#             ids = res.get("ids", [[]])[0]
#             distances = res.get("distances", [[]])[0]

#             for idx, doc in enumerate(docs):
#                 hits.append({
#                     "id": ids[idx],
#                     "document": doc,
#                     "metadata": metas[idx],
#                     "distance": distances[idx]
#                 })
#             return hits


#     def get_all_docs(self):
#         res = self.col.get()
#         rows = []
#         for i in range(len(res["ids"])):
#             rows.append({
#                 "id": res["ids"][i],
#                 "document": res["documents"][i],
#                 "metadata": res["metadatas"][i]
#             })
#         return rows

#     def retrieve_multi_round(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
#         """
#         Round 1 → broad retrieval
#         Round 2 → refine using expanded queries (query + top hits)
#         """

#         # ROUND 1: broad context
#         first = self.retrieve(query, top_k=top_k)

#         # Expand query with top 2 retrieved texts
#         expansions = []
#         for hit in first[:2]:
#             expansions.append(hit["document"])

#         expanded_query = query + " " + " ".join(expansions)

#         # ROUND 2: refined retrieval
#         second = self.retrieve(expanded_query, top_k=top_k)

#         # Merge + remove duplicates
#         combined = {h["id"]: h for h in first + second}

#         return list(combined.values())
    
#     def describe_related_images(self, query: str, image_results: List[dict]) -> List[dict]:
#         """
#         image_results = results from RetrievalAgent.retrieve(intent='visual')
#         Each item must have metadata containing ['img_path']
#         """
#         outputs = []

#         for item in image_results:
#             img_path = item["metadata"].get("img_path")
#             if not img_path or not isinstance(img_path, str):
#                 continue

#             try:
#                 caption = self.caption_image(img_path)
#                 outputs.append({
#                     "image_path": img_path,
#                     "caption": caption,
#                     "metadata": item["metadata"]
#                 })
#             except Exception as e:
#                 print("Error processing image:", img_path, e)

#         return outputs



#     def update_image_caption_embedding(self, item_id: str, caption: str):
#         emb = self.text_embedder.encode(caption)
#         self.col.update(
#             ids=[item_id],
#             embeddings=[emb.tolist()],
#             documents=[caption]
#         )


#Integrated OCR

import os
import uuid
from typing import List, Dict, Any
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import chromadb
from PIL import Image
import io
import pytesseract

CHROMA_DIR = "chroma_db"

class RetrievalAgent:
    def __init__(self, embed_model_name="all-MiniLM-L6-v2", persist_directory=CHROMA_DIR):
        import torch
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.text_embedder = SentenceTransformer(embed_model_name, device=self.device)

        self.client = chromadb.PersistentClient(path=persist_directory)
        coll_name = "multimodal_docs"
        existing_collections = [c.name for c in self.client.list_collections()]
        if coll_name in existing_collections:
            self.col = self.client.get_collection(coll_name)
        else:
            self.col = self.client.create_collection(
                name=coll_name,
                metadata={"description": "text+image multimodal"}
            )

    def ingest_pdfs(self, data_dir="data", chunk_chars=900, overlap=200):
        for root, _, files in os.walk(data_dir):
            for fn in files:
                if fn.lower().endswith(".pdf"):
                    path = os.path.join(root, fn)
                    self._process_pdf(path, chunk_chars, overlap)
        print("✅ PDF ingestion complete. Data persisted to:", os.path.abspath(CHROMA_DIR))

    def _process_pdf(self, path, chunk_chars, overlap):
        doc = fitz.open(path)
        images_dir = os.path.join(os.path.dirname(path), "images")
        os.makedirs(images_dir, exist_ok=True)

        for pno in range(len(doc)):
            page = doc[pno]
            text = page.get_text().strip()

            # --- Process text chunks
            if text:
                chunks = self._chunk_text(text, chunk_chars, overlap)
                for i, c in enumerate(chunks):
                    emb = self.text_embedder.encode(c)
                    uid = str(uuid.uuid4())
                    meta = {
                        "type": "text",
                        "source": os.path.basename(path),
                        "page": pno + 1,
                        "chunk_idx": i
                    }
                    self.col.add(
                        documents=[c],
                        metadatas=[meta],
                        ids=[uid],
                        embeddings=[emb.tolist()]
                    )

            # --- Extract images + OCR
            for img_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                base_image = doc.extract_image(xref)
                ext = base_image["ext"]
                image_bytes = base_image["image"]
                img_name = f"{os.path.splitext(os.path.basename(path))[0]}_p{pno+1}_img{img_index}.{ext}"
                img_path = os.path.join(images_dir, img_name)

                with open(img_path, "wb") as f:
                    f.write(image_bytes)

                # OCR
                try:
                    image = Image.open(io.BytesIO(image_bytes))
                    ocr_text = pytesseract.image_to_string(image).strip()
                except Exception:
                    ocr_text = ""

                # Store image placeholder
                placeholder_text = f"[IMAGE] {img_name}"
                emb = self.text_embedder.encode(f"image:{img_name}")
                uid = str(uuid.uuid4())
                meta = {
                    "type": "image",
                    "source": os.path.basename(path),
                    "page": pno + 1,
                    "img_path": img_path,
                    "img_name": img_name,
                    "ocr_text": ocr_text
                }
                self.col.add(
                    documents=[placeholder_text],
                    metadatas=[meta],
                    ids=[uid],
                    embeddings=[emb.tolist()]
                )

                # Store OCR text as separate chunk
                if ocr_text:
                    ocr_uid = str(uuid.uuid4())
                    ocr_emb = self.text_embedder.encode(ocr_text)
                    self.col.add(
                        documents=[ocr_text],
                        metadatas={**meta, "type": "pdf_image_ocr"},
                        ids=[ocr_uid],
                        embeddings=[ocr_emb.tolist()]
                    )

    def _chunk_text(self, text: str, chunk_chars: int, overlap: int) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = min(start + chunk_chars, len(text))
            chunks.append(text[start:end])
            start += chunk_chars - overlap
        return chunks

    def ingest_images_folder(self, images_dir="data/images"):
        if not os.path.exists(images_dir):
            print("⚠️ No image directory found:", images_dir)
            return

        for root, _, files in os.walk(images_dir):
            for fn in files:
                if fn.lower().endswith((".png", ".jpg", ".jpeg")):
                    p = os.path.join(root, fn)
                    # OCR
                    try:
                        image = Image.open(p)
                        ocr_text = pytesseract.image_to_string(image).strip()
                    except Exception:
                        ocr_text = ""

                    placeholder = f"[IMAGE] {fn}"
                    emb = self.text_embedder.encode(f"image:{fn}")
                    uid = str(uuid.uuid4())
                    meta = {
                        "type": "image",
                        "source": fn,
                        "img_path": p,
                        "img_name": fn,
                        "ocr_text": ocr_text
                    }
                    self.col.add(
                        documents=[placeholder],
                        metadatas=[meta],
                        ids=[uid],
                        embeddings=[emb.tolist()]
                    )
        print("✅ Image ingestion complete.")
    
    def retrieve(self, query: str, top_k: int = 5, intent: str = "default") -> List[Dict[str, Any]]:
        q_emb = self.text_embedder.encode(query)
        if intent == "visual":
            res = self.col.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k,
                where={"type": "image"}
            )
        else:
            res = self.col.query(
                query_embeddings=[q_emb.tolist()],
                n_results=top_k
            )
        hits = []
        docs = res.get("documents", [[]])[0]
        metas = res.get("metadatas", [[]])[0]
        ids = res.get("ids", [[]])[0]
        distances = res.get("distances", [[]])[0]
        for idx, doc in enumerate(docs):
            hits.append({
                "id": ids[idx],
                "document": doc,
                "metadata": metas[idx],
                "distance": distances[idx]
            })
        return hits

    def get_all_docs(self):
        res = self.col.get()
        rows = []
        for i in range(len(res["ids"])):
            rows.append({
                "id": res["ids"][i],
                "document": res["documents"][i],
                "metadata": res["metadatas"][i]
            })
        return rows

    def retrieve_multi_round(self, query: str, top_k: int = 8) -> List[Dict[str, Any]]:
        first = self.retrieve(query, top_k=top_k)
        expansions = [hit["document"] for hit in first[:2]]
        expanded_query = query + " " + " ".join(expansions)
        second = self.retrieve(expanded_query, top_k=top_k)
        combined = {h["id"]: h for h in first + second}
        return list(combined.values())

    def describe_related_images(self, query: str, image_results: List[dict]) -> List[dict]:
        """
        image_results = results from retrieve(intent='visual')
        Each item must have metadata containing ['img_path', 'ocr_text']
        """
        from agents.vision_agent import VisionAgent
        vision_agent = VisionAgent()
        outputs = []
        for item in image_results:
            img_path = item["metadata"].get("img_path")
            if not img_path or not isinstance(img_path, str):
                continue
            res = vision_agent.describe_image(img_path)
            outputs.append({
                "caption": res.get("caption", ""),
                "ocr_text": res.get("ocr_text", ""),
                "meta": item["metadata"]
            })
        return outputs

