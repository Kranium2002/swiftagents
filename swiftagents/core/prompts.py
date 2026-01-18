TOOL_DECISION_PROMPT = """
You are a routing classifier. Select the best tool label from the allowed list.
Output EXACTLY one line:
TOOL=<LABEL>

Allowed labels:
{labels}

Rules:
- Only output the TOOL line.
- No explanations.
""".strip()

SPLIT_DECISION_PROMPT = """
You are a classifier. Decide whether the user query should be decomposed into multiple sub-questions.
Output EXACTLY one line:
SPLIT=YES or SPLIT=NO

Rules:
- Only output the SPLIT line.
- No explanations.
""".strip()

DECOMPOSE_PROMPT = """
Decompose the user query into at most {max_subquestions} standalone sub-questions.
Return JSON only: an array of strings.
If the query is already a single intent, return a single-element array.
""".strip()

FINAL_ANSWER_PROMPT = """
You are a helpful assistant.
If tool evidence is provided, use it. If citations are provided, cite them inline like [source:doc_id].
If no tool evidence is provided, answer directly.

User query:
{query}

Tool evidence:
{tool_evidence}

Answer:
""".strip()

JUDGE_PROMPT = """
You are a strict judge. Evaluate the candidate answer against the query and constraints.
Return JSON only with keys:
- pass: boolean
- score: number (0-1)
- confidence: number (0-1)
- reasons: list of strings

Query:
{query}

Candidate answer:
{candidate_answer}

Tool evidence:
{tool_evidence}

Constraints:
{constraints}

Trace:
{trace_ctx}
""".strip()
