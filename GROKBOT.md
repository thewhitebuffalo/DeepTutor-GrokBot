# DeepTutor as a tutoring backend for Grok Bot

This repository is a fork of [HKUDS/DeepTutor](https://github.com/HKUDS/DeepTutor).
The original project remains under the [Apache License 2.0](LICENSE). Nothing
in this fork changes that license or the upstream authors' copyright. The
addition here is a headless study backend so an external assistant can drive
DeepTutor's learning machinery without DeepTutor calling a paid LLM.

Grok Bot runs on a headless Linux machine. It does the teaching and writes the
questions. DeepTutor stores documents, retrieves cited passages, keeps the
question bank, grades attempts, schedules reviews, and holds memory notes and
a task board.

## Install

Python 3.11 or newer. From a checkout of this fork:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

`pip install deeptutor` from PyPI is upstream DeepTutor and does not include
this interface. Install from this fork.

No API key is required. Indexing uses a local embedder:

| `DEEPTUTOR_EMBED_BACKEND` | What runs |
| --- | --- |
| `hash` (forced in the smoke test) | Built-in feature hashing. No download, no network, no key. |
| `auto` (default) | sentence-transformers when that extra is installed, otherwise hashing. |
| `sentence-transformers` | A CPU model. Default `BAAI/bge-small-en-v1.5`. |

```bash
pip install -e '.[local-embed]'
export DEEPTUTOR_EMBED_BACKEND=sentence-transformers
export DEEPTUTOR_EMBED_MODEL=BAAI/bge-small-en-v1.5
```

`local-embed` pulls PyTorch. Leave it unset and the hashing embedder still
retrieves by overlapping terms, which is enough for standards, notes, and the
smoke test.

Data lives in one directory:

```bash
export DEEPTUTOR_STUDY_DIR="$HOME/.deeptutor/study"
# or: export DEEPTUTOR_HOME="$HOME/.deeptutor"
#     which stores this backend at $DEEPTUTOR_HOME/data/study
```

The web UI is not required. Upstream `deeptutor start` still launches it when
you want the full app.

### Docker

The stock image starts the web stack. For the study server only, override the
entrypoint after building this fork:

```bash
docker build -t deeptutor-grokbot .
docker run --rm -i --entrypoint python \
  -v "$HOME/.deeptutor/study:/data" \
  -e DEEPTUTOR_STUDY_DIR=/data \
  -e DEEPTUTOR_EMBED_BACKEND=hash \
  deeptutor-grokbot -m deeptutor.agent_study
```

### MCP client config

The server speaks newline-delimited JSON-RPC on stdin/stdout. Point the
assistant at either entry point:

```json
{
  "mcpServers": {
    "deeptutor": {
      "command": "deeptutor-study-mcp",
      "env": {
        "DEEPTUTOR_STUDY_DIR": "/home/me/.deeptutor/study",
        "DEEPTUTOR_EMBED_BACKEND": "hash"
      }
    }
  }
}
```

From a source tree that is not installed yet:

```json
{
  "command": "python",
  "args": ["-m", "deeptutor.agent_study"],
  "env": {
    "PYTHONPATH": "/path/to/DeepTutor-GrokBot",
    "DEEPTUTOR_STUDY_DIR": "/home/me/.deeptutor/study",
    "DEEPTUTOR_EMBED_BACKEND": "hash"
  }
}
```

`deeptutor study mcp` is the same server behind the CLI.

## Tool list

Every tool returns JSON. The CLI prints the same JSON. Answers are stored for
the assistant; `mastery_next` omits them so a quiz prompt is not spoiled.

| Tool | CLI | Purpose |
| --- | --- | --- |
| `study_status` | `deeptutor study status` | Data directory and embedder. `api_key_required` is false. |
| `kb_create` | `deeptutor study kb create NAME` | Create a knowledge base. |
| `kb_list` | `deeptutor study kb list` | List knowledge bases. |
| `kb_ingest` | `deeptutor study kb ingest NAME PATH...` | Index files or a folder. |
| `kb_retrieve` | `deeptutor study kb retrieve NAME "QUERY"` | Passages with file, page, section. |
| `question_save` | `deeptutor study question save ...` | Save an assistant-written question. |
| `question_list` | `deeptutor study question list` | List a question bank. |
| `attempt_record` | `deeptutor study attempt record ID --answer ...` | Grade an attempt and update mastery. |
| `mastery_status` | `deeptutor study mastery status` | Per-topic mastery and schedule. |
| `mastery_next` | `deeptutor study mastery next` | What to review next. |
| `memory_list` | `deeptutor study memory list` | List memory notes. |
| `memory_read` | `deeptutor study memory read ID` | Read one note. |
| `memory_write` | `deeptutor study memory write --title ... --body ...` | Create or replace a note. |
| `memory_delete` | `deeptutor study memory delete ID` | Delete a note. |
| `task_list` | `deeptutor study task list` | Read the agent task board. |
| `task_write` | `deeptutor study task write --title ...` | Create or update a card. |

Supported ingest types: PDF, Markdown, plain text, `.docx`, `.pptx`, `.xlsx`.
PDF text comes from PyMuPDF. Office text comes from python-docx, python-pptx,
and openpyxl. None of these call a model. A knowledge base remembers the
embedder it was created with; mixing hash and sentence-transformers means
creating a new knowledge base.

### Examples

Create and index:

```bash
deeptutor study kb create pe-exam --description "IEEE standards"
deeptutor study kb ingest pe-exam ./standards ./notes.md
deeptutor study kb retrieve pe-exam "insulation resistance megaohm" --k 4
```

A hit looks like:

```json
{
  "file": "ieee-43.pdf",
  "page": 12,
  "section": "5. Insulation resistance",
  "text": "...",
  "score": 0.71,
  "kb": "pe-exam",
  "chunk_id": "c_..."
}
```

Save a question the assistant wrote from that passage:

```bash
deeptutor study question save \
  --bank pe-exam \
  --topic "IEEE 43" \
  --question "What is the minimum insulation resistance per IEEE Std 43?" \
  --answer "one megaohm plus one megaohm per kilovolt" \
  --explanation "Std 43 scales the floor with machine rating." \
  --tag ieee --tag insulation \
  --sources-json '[{"file":"ieee-43.pdf","page":12,"section":"5. Insulation resistance","kb":"pe-exam"}]'
```

Grade the learner. Omit `--correct` to compare with the stored answer. Pass
`--correct` or `--score` when the assistant judged a free-text explanation
itself (`--score` is 0 to 1):

```bash
deeptutor study attempt record q_abc123 --answer "one megaohm per kilovolt"
deeptutor study mastery status --topic "IEEE 43"
deeptutor study mastery next
```

A failed attempt stays due, so the next quiz can ask it again in the same
session. A pass schedules the topic for 1 day, then 6 days, then
`interval * ease` (SM-2, ease floored at 1.3).

Memory and tasks:

```bash
deeptutor study memory write --title "Exam focus" --body "Weak on insulation tests."
deeptutor study task write --title "Reread IEEE 43 section 5" --status todo
deeptutor study task write --id t_abc123 --status doing
```

The same calls over MCP are `tools/call` with those names. A retrieve looks like:

```json
{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"kb_retrieve","arguments":{"name":"pe-exam","query":"insulation resistance","k":4}}}
```

`--data-dir PATH` on any `deeptutor study` command overrides `DEEPTUTOR_STUDY_DIR`.

## A study loop

1. Ingest the PDFs, Word documents, slides, and notes for one course into a knowledge base.
2. Retrieve passages for the topic the learner is on. Draft questions from those passages, including the file, page, and section as `sources`.
3. Save each question with its answer, explanation, tags, and sources.
4. Quiz the learner with `mastery_next` (the payload has no answer key).
5. Record the attempt. Use local grading for short answers, or pass your own `correct` / `score` for an explanation you judged.
6. Read `mastery_status`. Failed topics stay on the review queue. Passed topics come back on the spaced-repetition date.
7. Write a memory note when the learner reveals a preference or a lasting gap. Put follow-up reading on the task board.

## What still needs a model

The study tools above do not. These upstream features still call an LLM (or a
local server standing in for one). Configure Ollama, llama.cpp, LM Studio, or
the existing Codex sign-in for them; do not put a paid key in the environment
if you are not using them.

| Feature | Why it still wants a model |
| --- | --- |
| `deeptutor run chat` and the other capabilities (`deep_solve`, `deep_question`, `deep_research`, `visualize`, `math_animator`, `mastery_path`, `immersive_reading`, `course_study`) | The in-app tutor writes the reply. |
| Upstream question generation | `deep_question` drafts items itself. This fork's question bank is for questions the assistant already wrote. |
| Upstream mastery path | The guided tutor, qualitative judging, and lesson prep. Per-topic scheduling for assistant-authored questions is the study backend's SM-2 state instead. |
| Memory consolidation | `deeptutor memory` L2/L3 summaries. `deeptutor study memory` is plain notes and does not consolidate. |
| GraphRAG, LightRAG, PageIndex | Graph builds and agentic page reasoning. |
| Image captions while indexing a web-UI knowledge base | Describing figures. The study ingest path keeps extracted text only. |
| Query rewriting | Optional LLM rewrite on the upstream retriever. Study retrieval embeds the query as given. |

The web UI knowledge bases (`deeptutor kb`) can also avoid a paid embedding key.
In Settings, or in the model catalog, choose the embedding binding `local`,
model `hashing` (or `BAAI/bge-small-en-v1.5` after `local-embed` is installed),
and endpoint `local://embeddings`. Leave the API key empty. That path still
uses the LlamaIndex pipeline; text extraction follows the document-parsing
engine configured for the app. The study commands are the ones guaranteed to
run with no catalog and no key.

Ollama and LM Studio remain available as upstream local providers when you do
want a generative model. Codex sign-in (`deeptutor provider`) is unchanged and
optional.
