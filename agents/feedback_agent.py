# # agents/feedback_agent.py
# from typing import Dict, Any, List
# import re

# class FeedbackAgent:
#     def __init__(self):
#         pass

#     def evaluate(self, query: str, answer: str, retrieved_chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
#         """
#         Lightweight heuristic evaluation of the model response.
#         Returns a confidence score and category-level breakdowns.
#         """

#         # ---- 1. RELEVANCE SCORE ----
#         query_keywords = set(re.findall(r"\w+", query.lower()))
#         ans_keywords = set(re.findall(r"\w+", answer.lower()))
#         overlap = len(query_keywords & ans_keywords)
#         relevance = min(10, overlap)  # crude heuristic

#         # ---- 2. FAITHFULNESS SCORE ----
#         retrieved_text = " ".join([r["document"] for r in retrieved_chunks])
#         retrieved_keywords = set(re.findall(r"\w+", retrieved_text.lower()))
#         hallucinations = len(ans_keywords - retrieved_keywords)
#         faithfulness = max(0, 10 - min(hallucinations, 10))

#         # ---- 3. CLARITY SCORE ----
#         clarity = 10
#         if len(answer) < 20:
#             clarity = 4
#         if answer.count(".") == 0:
#             clarity -= 3
#         if len(answer.split()) > 80:
#             clarity -= 2

#         clarity = max(0, min(10, clarity))

#         # ---- OVERALL SCORE ----
#         overall = round((relevance + faithfulness + clarity) / 3, 2)

#         return {
#             "relevance": relevance,
#             "faithfulness": faithfulness,
#             "clarity": clarity,
#             "overall_confidence": overall
#         }
