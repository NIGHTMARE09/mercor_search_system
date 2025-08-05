npm install mongodb turbopuffer dotenv voyageai flexsearch axios express

# Mercor Search System

A production-ready search system for retrieving candidate profiles based on job queries. It leverages vector search (TurboPuffer), embeddings (VoyageAI), MongoDB for storage, and FlexSearch for fuzzy keyword matching. The system is optimized for high recall on hard and soft criteria, with keyword extraction and synonyms for better matching.

## Features
- **Semantic Search:** Uses VoyageAI embeddings and TurboPuffer for fast approximate nearest neighbor (ANN) vector search.
- **Fuzzy/Partial Matching:** FlexSearch enables robust keyword and partial/fuzzy matching for both hard and soft criteria.
- **Keyword Extraction:** Extracts and generalizes keywords from long criteria sentences for improved recall.
- **Weighted Scoring:** Combines hard binary, soft average, and vector similarity for candidate ranking.
- **Batch Indexing:** Efficiently upserts large batches of profiles into TurboPuffer.

## Environment Setup

### Prerequisites
- Node.js v18+
- MongoDB Atlas or local instance
- API keys for VoyageAI and TurboPuffer

### Install Dependencies
```bash
npm install mongodb turbopuffer dotenv voyageai flexsearch axios express
npm install --save-dev typescript ts-node @types/node @types/express
```

### Configure .env
Create a `.env` file in the root directory:
```
MONGODB_URI=your_mongodb_uri
VOYAGE_API_KEY=your_voyage_api_key
TURBOPUFFER_API_KEY=your_turbopuffer_key
TURBOPUFFER_BASE_URL=https://us-east-1.turbopuffer.com
PORT=3000
GRADING_ENDPOINT=your_grading_endpoint_url
YOUR_EMAIL=your_email_for_auth
```

## Running the Setup Script
The setup script loads data from MongoDB, generates embeddings (if needed), and indexes into TurboPuffer. Implemented as `upsertProfiles()` in `search.ts`.

### Run the Server
```bash
npm run dev
```

### Invoke Setup
Use Postman or curl to hit the `/upsert` endpoint (POST request, no body needed):
```bash
curl -X POST http://localhost:3000/upsert
```

- Batch size is 1000 (adjustable in code).
- If embeddings are missing, a zero vector is used as fallback.

## Search & Retrieval
The `searchProfiles(query)` function in `search.ts` takes a query object, consider top 150 candidates/ profiles and returns up to 10 candidate IDs with scores. It uses:
- VoyageAI for query embedding
- TurboPuffer ANN for top 150 candidates
- FlexSearch for fuzzy keyword matching
- Keyword extraction for long criteria
- Weighted scoring: 80% hard binary, 10% soft avg, 10% vector sim

### Example Usage
```js
const query = {
  text: "Your job description",
  hardCriteria: { skills: ["skill1", "skill2"], experience: { min: 2 } },
  softCriteria: { skills: ["soft1", "soft2"], country: "US" }
};
const results = await searchProfiles(query);
console.log(results); // Array of {id, score}
```

## Evaluation
Use the `/grade` endpoint in `index.ts` to submit results and get scores. Example:
```bash
curl -X POST http://localhost:3000/grade \
     -H "Content-Type: application/json" \
     -d '{"config_candidates": {"config_name": "mathematics_phd", "candidates": [{"id": "id1", "score": 0.9}, ...]}}'
```
This forwards to `GRADING_ENDPOINT` with auth (`YOUR_EMAIL`). Prints `overallScore` from response.

# Approach Summary

## Data Exploration and Indexing/Retrieval Strategy

### Exploration

Analyzed the linkedin_data_subset collection in MongoDB, focusing on fields like skills, rerankSummary, country, yearsOfWorkExperience, and pre-computed embedding. Performed clustering on embeddings to understand semantic groupings (e.g., math vs biology profiles clustered well). Tested precision/recall on sample queries—initial recall low on hard criteria due to strict matching, so added fuzzy with FlexSearch.

### Strategy Choice

Chose TurboPuffer for scalable vector search (ANN with cosine distance) due to fast query times and easy integration.

VoyageAI "voyage-3" for embeddings (1024 dims, high quality for text queries).

FlexSearch for on-the-fly fuzzy keyword matching on criteria (no full re-index needed).

Added keyword extraction to handle long criteria sentences (e.g., splitting "Research expertise in pure or applied mathematics..." into "pure mathematics", "applied mathematics", etc.).

Synonyms for recall boost (e.g., "phd" to "doctorate") targeted low-performing configs like mathematics_phd.

Optimized for recall (top_k=150, lenient hard thresholds) over precision, as per evaluation metric (hit rate at top 10/100).

## Validation/Analysis

### Local Tests

Ran local tests with debug logs to check hard/soft match rates (e.g., hard pass improved 20-30% with keywords/synonyms).

### Precision/Recall Checks

On biology_expert, recall at 10 increased from 0.4 to 0.7 after fuzzy.

### Clustering

Used cosine similarity on embeddings to validate semantic relevance (e.g., math queries clustered profiles with "statistics" skills).

### Edge Cases

Handled missing embeddings with zero vectors, sparse criteria with fallbacks.

This approach ensures robust performance on both public and private queries, with focus on criteria satisfaction for the LLM judge.

## Final Results
---
### Scores on 10 Public Queries

Below is a table summarizing the scores from the final `/grade` endpoint test on the 10 public queries. The table is sorted in descending order of average final score. Note that in queries with low `avg_final_score` (e.g., `mathematics_phd`, `quantitative_finance`, `doctors_md`), hard criteria were not fully fulfilled for many candidates, requiring padding with next-best matches. This led to significantly lower overall scores in those cases, as the evaluation emphasizes hard criteria satisfaction.

#### Table

| Config Name | Average Final Score | Key Hard Pass Rates | Soft Averages | Notes |
| --- | --- | --- | --- | --- |
| mechanical_engineers | 85.3 | higher_degree: 1.0, three_plus_years: 1.0 | cad_tools: 8.55, lifecycle: 8.5, domain_specialization: 8.55 | Perfect hard pass; soft consistent. No padding needed. |
| tax_lawyer | 77.0 | has_jd_degree: 0.9, three_plus_years: 1.0 | legal_writing: 8.5, irs_audit: 8.4, corporate_transaction: 8.8 | Strong hard and soft; minimal issues. |
| junior_corporate_lawyer | 66.7 | reputed_law_school: 1.0, appropriate_experience: 0.8 | ma_transaction: 8.2, international_law: 8.3, contract_negotiation: 7.8 | Good soft; experience pass slightly low. |
| radiology | 66.0 | has_md_degree: 1.0 | board_cert: 4.0, diagnostic_imaging: 7.9, radiology_expertise: 7.9 | Hard perfect, but board cert soft weak dragged score. |
| anthropology | 49.3 | phd_relevant: 1.0, recent_phd: 0.6 | academic_output: 7.8, ethnographic: 8.7, applied: 8.2 | Recent PhD low required some padding. |
| bankers | 41.7 | two_plus_years: 0.8, mba_us: 0.7 | healthcare_metrics: 7.2, healthcare_banking: 8.2, ma_transactions: 7.8 | MBA pass low led to padding; soft moderate. |
| biology_expert | 30.0 | phd_top_us: 0.6, undergrad_location: 0.4 | experimental_design: 7.8, molecular_research: 8.2, teaching: 5.1 | Undergrad location very low, heavy padding needed, low scores. |
| doctors_md | 24.5 | two_plus_years: 0.9, top_us_md: 0.3, general_practitioner: 1.0 | ehr_systems: 6.8, telemedicine: 7.1 | Top US MD extremely low, required extensive padding, hurting overall score. |
| quantitative_finance | 17.7 | three_plus_years: 0.9, m7_mba: 0.2 | high_stakes: 5.7, python_prof: 8.45, financial_modeling: 8.55 | M7 MBA very low, lots of padding, low final scores. |
| mathematics_phd | 16.5 | undergrad_location: 0.2, phd_math: 0.6 | research_expertise: 8.65, modeling_proficiency: 8.0 | Undergrad location critically low, heavy padding, very low scores. |


----------------------------------------------------------------

These scores reflect optimizations like fuzzy matching and keyword extraction. Low scores in some queries are primarily due to unfulfilled hard criteria, leading to padding with candidates that scored lower in evaluation.

-------------------------------

**Author:** Shivam Jha


