"""
parse_jd.py — JD understanding and structured extraction.

Extracts hard requirements, domain signals, and embeddings from the
job_description. The JD is embedded in three sections separately:
  - full JD       → general fit / career narrative comparison
  - responsibilities section → what the person will actually DO
  - requirements section     → what they need to HAVE

This matters because max(req_sim, resp_sim) is wrong — they capture
different evidence and need separate weights downstream.
"""

import json
import re
from pathlib import Path

# ── JD is hard-coded from the provided job_description.docx content ──
# This avoids python-docx dependency at rank time and is faster.

JD_FULL = """
Senior AI Engineer — Founding Team
Company: Redrob AI (Series A AI-native talent intelligence platform)
Location: Pune/Noida, India (Hybrid) | Open to relocation from Tier-1 Indian cities
Experience Required: 5–9 years

We need someone simultaneously comfortable with deep technical depth in modern ML systems —
embeddings, retrieval, ranking, LLMs, fine-tuning — and a scrappy product-engineering attitude.
We'd rather you tilt slightly toward shipper than toward researcher.

RESPONSIBILITIES:
Own the intelligence layer of Redrob's product — ranking, retrieval, and matching systems.
Weeks 1-3: Audit existing BM25 + rule-based scoring, identify highest-leverage improvements.
Weeks 4-8: Ship v2 ranking system with embeddings, hybrid retrieval, LLM-based re-ranking.
Weeks 9-12: Set up evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops.
Drive long-term architecture of candidate-JD matching at scale.
Mentor next round of hires — growing team from 4 to 12 engineers.
Work closely with recruiter-experience PM on product decisions.
Design production recommendation and retrieval systems.
Build and deploy ML services to real users at meaningful scale.
Own online A/B testing and recruiter engagement metrics.

REQUIREMENTS (MUST HAVE):
Production experience with embeddings-based retrieval systems — sentence-transformers, OpenAI embeddings, BGE, E5 or similar.
Handled embedding drift, index refresh, retrieval-quality regression in production.
Production experience with vector databases or hybrid search — Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, Elasticsearch, FAISS.
Strong Python — we care about code quality.
Hands-on experience designing evaluation frameworks for ranking systems — NDCG, MRR, MAP, offline-to-online correlation, A/B test interpretation.
Product company experience — not pure services background.
5-9 years experience in applied ML/AI at product companies.
Has shipped at least one end-to-end ranking, search, or recommendation system to real users at meaningful scale.

NICE TO HAVE:
LLM fine-tuning experience — LoRA, QLoRA, PEFT.
Experience with learning-to-rank models — XGBoost-based or neural.
Prior exposure to HR-tech, recruiting tech, or marketplace products.
Background in distributed systems or large-scale inference optimization.
Open-source contributions in AI/ML space.

DISQUALIFIERS:
Pure research environments without production deployment.
AI experience consisting primarily of recent LangChain/OpenAI API projects under 12 months.
Senior engineers who haven't written production code in last 18 months.
Only worked at consulting firms (TCS, Infosys, Wipro, Accenture, Cognizant, Capgemini) entire career.
Primary expertise in computer vision, speech, or robotics without NLP/IR exposure.
Title-chasers switching every 1.5 years.
"""

JD_RESPONSIBILITIES = """
Own the intelligence layer — ranking, retrieval, and matching systems that decide what recruiters see.
Audit existing BM25 plus rule-based scoring and identify highest-leverage improvements.
Ship v2 ranking system with embeddings, hybrid retrieval, and LLM-based re-ranking.
Set up evaluation infrastructure — offline benchmarks, online A/B testing, recruiter-feedback loops.
Drive long-term architecture of candidate-JD matching at scale.
Mentor engineers growing team from 4 to 12.
Design production recommendation systems and retrieval pipelines.
Deploy ML services to real users at meaningful scale.
Build and maintain vector search infrastructure.
Own online A/B testing and recruiter engagement metrics.
Build ranking, search, and recommendation systems for production traffic.
Reduce latency and improve retrieval quality at scale.
"""

JD_REQUIREMENTS = """
Production experience with embeddings-based retrieval systems.
Sentence-transformers, OpenAI embeddings, BGE, E5, or similar models deployed to real users.
Handled embedding drift, index refresh, retrieval-quality regression in production.
Production experience with vector databases — Pinecone, Weaviate, Qdrant, Milvus, FAISS, Elasticsearch.
Strong Python and production code quality.
Evaluation frameworks for ranking systems — NDCG, MRR, MAP, A/B test interpretation.
5 to 9 years applied ML/AI at product companies not pure services.
Shipped end-to-end ranking search or recommendation system to real users at scale.
Product company background required not consulting or IT services.
Located in or willing to relocate to Pune or Noida India.
Sub-30-day notice period preferred.
"""

# ── Structured requirements ──
JD_STRUCTURED = {
    "min_years_exp": 5,
    "max_years_exp": 9,
    "product_company_required": True,
    "preferred_locations": ["pune", "noida"],
    "acceptable_countries": ["india"],
    "preferred_notice_days": 30,
    "max_reasonable_notice_days": 90,

    # Hard domain skills — must have at least some of these in career evidence
    "core_skills": [
        "embeddings", "retrieval", "ranking", "vector search",
        "semantic search", "faiss", "pinecone", "weaviate", "qdrant",
        "milvus", "elasticsearch", "opensearch", "sentence-transformers",
        "bge", "e5", "rag", "llm", "fine-tuning", "recommendation",
        "information retrieval", "hybrid search", "ann", "approximate nearest neighbor",
        "dense retrieval", "bi-encoder", "cross-encoder", "reranking",
        "ndcg", "mrr", "map", "a/b testing", "evaluation framework",
        "pytorch", "transformers", "huggingface", "bert", "nlp",
        "machine learning", "deep learning", "neural network",
    ],

    # Nice-to-have skills
    "bonus_skills": [
        "lora", "qlora", "peft", "learning to rank", "xgboost",
        "distributed systems", "inference optimization", "hr-tech",
        "marketplace", "open source", "mlops", "kubeflow",
    ],

    # These in the ONLY background = disqualifier
    "services_companies": [
        "tcs", "infosys", "wipro", "accenture", "cognizant",
        "capgemini", "hcl", "tech mahindra", "mphasis", "hexaware",
        "mindtree", "l&t infotech", "ltimindtree", "niit technologies",
        "kpit", "cyient", "zensar", "persistent systems", "sonata",
    ],

    # Product/startup signals — positive
    "product_industries": [
        "software", "saas", "technology", "fintech", "edtech",
        "healthtech", "ai", "internet", "e-commerce", "marketplace",
        "product", "platform", "startup",
    ],

    "product_company_sizes": ["1-10", "11-50", "51-200", "201-500"],
}


def get_jd():
    """Return the structured JD dict. Used by all other modules."""
    return {
        "full_text": JD_FULL.strip(),
        "responsibilities_text": JD_RESPONSIBILITIES.strip(),
        "requirements_text": JD_REQUIREMENTS.strip(),
        "structured": JD_STRUCTURED,
        # Embeddings are added later by precompute.py
        "full_emb": None,
        "resp_emb": None,
        "req_emb": None,
    }


if __name__ == "__main__":
    jd = get_jd()
    print("JD structured requirements:")
    print(json.dumps(jd["structured"], indent=2, default=str))
    print(f"\nFull text length: {len(jd['full_text'])} chars")
    print(f"Responsibilities: {len(jd['responsibilities_text'])} chars")
    print(f"Requirements: {len(jd['requirements_text'])} chars")
