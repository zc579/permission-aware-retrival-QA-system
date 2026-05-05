import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict

from .data import chunk_documents, load_docs
from .qa import answer_query


ROOT = Path(__file__).resolve().parents[1]
DOCS_PATH = ROOT / "data" / "docs.jsonl"

USERS = {
    "manager": {"user": "bob", "role": "manager"},
    "researcher": {"user": "alice", "role": "researcher"},
    "intern": {"user": "charlie", "role": "intern"},
}

SAMPLE_QUERIES = [
    "What improvement in throughput did Alpha show?",
    "Summarize confidential design choices for Project Alpha",
    "What does Project Alpha say about differential privacy?",
    "What future experiments does the Alpha literature review suggest?",
    "List intern observations on Beta project",
]


def _load_chunks():
    return chunk_documents(load_docs(str(DOCS_PATH)))


def _load_file_index():
    files = []
    for doc in load_docs(str(DOCS_PATH)):
        files.append({
            "doc_id": doc.get("id"),
            "title": doc.get("title"),
            "source": doc.get("source"),
            "project": doc.get("project"),
            "sensitivity": doc.get("sensitivity"),
            "allowed_roles": doc.get("allowed_roles", []),
            "allowed_users": doc.get("allowed_users", []),
            "text_preview": (doc.get("text") or "")[:180],
        })
    return files


def _html() -> str:
    return """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Permission-Aware RAG Demo</title>
  <style>
    :root {
      color-scheme: light;
      --bg: #f6f8fb;
      --panel: #ffffff;
      --text: #1d2733;
      --muted: #5d6b7a;
      --line: #d8e0ea;
      --accent: #0f766e;
      --accent-dark: #115e59;
      --warn: #a16207;
    }
    * { box-sizing: border-box; }
    html { min-width: 0; }
    body {
      margin: 0;
      min-width: 0;
      background: var(--bg);
      color: var(--text);
      font: 15px/1.45 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    header {
      background: #16202c;
      color: white;
      padding: 18px clamp(16px, 4vw, 32px);
    }
    header h1 {
      width: min(1180px, 100%);
      margin: 0 auto;
      font-size: 22px;
      font-weight: 700;
      letter-spacing: 0;
    }
    main {
      width: min(1180px, calc(100vw - 28px));
      margin: 20px auto 34px;
      display: grid;
      gap: 16px;
      min-width: 0;
    }
    .controls, .section {
      min-width: 0;
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 16px;
    }
    .controls {
      display: grid;
      grid-template-columns: minmax(140px, 180px) minmax(260px, 1fr) minmax(136px, 150px) minmax(96px, auto);
      gap: 12px;
      align-items: end;
    }
    label {
      display: grid;
      min-width: 0;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
      font-weight: 650;
      white-space: nowrap;
    }
    select, input {
      width: 100%;
      min-width: 0;
      min-height: 42px;
      border: 1px solid var(--line);
      border-radius: 6px;
      padding: 10px 11px;
      color: var(--text);
      background: white;
      font: inherit;
    }
    .query-wrap {
      position: relative;
      min-width: 0;
    }
    .examples {
      position: absolute;
      z-index: 5;
      top: calc(100% + 6px);
      left: 0;
      right: 0;
      display: grid;
      gap: 4px;
      max-height: 220px;
      overflow: auto;
      padding: 6px;
      border: 1px solid var(--line);
      border-radius: 8px;
      background: white;
      box-shadow: 0 12px 30px rgba(22, 32, 44, 0.14);
    }
    .examples[hidden] { display: none; }
    .example-option {
      width: 100%;
      min-height: auto;
      border: 0;
      border-radius: 6px;
      padding: 8px 9px;
      background: transparent;
      color: var(--text);
      font-weight: 600;
      text-align: left;
    }
    .example-option:hover {
      background: #eef6f5;
      color: var(--accent-dark);
    }
    button {
      width: 100%;
      border: 0;
      border-radius: 6px;
      padding: 11px 16px;
      background: var(--accent);
      color: white;
      font: inherit;
      font-weight: 700;
      cursor: pointer;
      min-height: 42px;
    }
    button:hover { background: var(--accent-dark); }
    button:disabled { opacity: 0.6; cursor: wait; }
    .grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 16px;
      align-items: stretch;
      min-width: 0;
    }
    .grid > .section {
      display: flex;
      flex-direction: column;
      min-height: 300px;
    }
    h2 {
      margin: 0 0 10px;
      font-size: 16px;
      letter-spacing: 0;
    }
    .answer {
      min-height: 118px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .meta {
      min-width: 0;
      color: var(--muted);
      font-size: 13px;
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
    }
    .pill {
      max-width: 100%;
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 3px 9px;
      background: #f9fbfd;
      overflow-wrap: anywhere;
    }
    #citations {
      width: 100%;
      flex: 1;
      overflow-x: auto;
    }
    table {
      width: 100%;
      min-width: 430px;
      border-collapse: collapse;
      font-size: 13px;
    }
    th, td {
      border-bottom: 1px solid var(--line);
      padding: 8px 6px;
      text-align: left;
      vertical-align: top;
      overflow-wrap: anywhere;
    }
    th { color: var(--muted); font-weight: 700; }
    .chunks {
      display: grid;
      gap: 10px;
      min-width: 0;
    }
    .chunk {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      background: #fbfcfe;
    }
    .chunk strong {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      margin-bottom: 5px;
      overflow-wrap: anywhere;
    }
    .files-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
      min-width: 0;
    }
    .file-card {
      min-width: 0;
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 11px;
      background: #fbfcfe;
    }
    .file-title {
      display: flex;
      flex-wrap: wrap;
      align-items: center;
      gap: 6px;
      margin-bottom: 6px;
      font-weight: 750;
      overflow-wrap: anywhere;
    }
    .file-preview {
      margin-top: 8px;
      color: var(--muted);
      font-size: 13px;
      overflow-wrap: anywhere;
    }
    pre {
      flex: 1;
      margin: 0;
      min-height: 214px;
      max-height: 300px;
      overflow: auto;
      background: #101820;
      color: #e7edf5;
      border-radius: 8px;
      padding: 12px;
      font-size: 12px;
      white-space: pre-wrap;
      overflow-wrap: anywhere;
    }
    .warning { color: var(--warn); }
    @media (max-width: 980px) {
      .controls {
        grid-template-columns: minmax(160px, 0.7fr) minmax(240px, 1fr);
      }
      .controls button {
        grid-column: 2;
      }
      .grid {
        grid-template-columns: 1fr;
      }
    }
    @media (max-width: 640px) {
      body { font-size: 14px; }
      header { padding: 15px 14px; }
      header h1 { font-size: 19px; }
      main {
        width: calc(100vw - 20px);
        margin-top: 10px;
        gap: 10px;
      }
      .controls, .section {
        padding: 12px;
        border-radius: 7px;
      }
      .controls {
        grid-template-columns: 1fr;
        gap: 10px;
      }
      .controls button {
        grid-column: auto;
      }
      .answer {
        min-height: 96px;
      }
      table {
        min-width: 390px;
      }
      pre {
        max-height: 240px;
      }
      .files-grid {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <header><h1>Permission-Aware RAG Demo</h1></header>
  <main>
    <section class="controls">
      <label>Identity
        <select id="identity">
          <option value="manager">Manager</option>
          <option value="researcher">Researcher</option>
          <option value="intern">Intern</option>
        </select>
      </label>
      <label>Question
        <div class="query-wrap">
          <input id="query" autocomplete="off" value="What improvement in throughput did Alpha show?">
          <div id="examples" class="examples" hidden></div>
        </div>
      </label>
      <label>Retrieved chunks
        <select id="topK" title="Number of authorized text chunks to retrieve before answer synthesis">
          <option value="3">3 chunks</option>
          <option value="5">5 chunks</option>
        </select>
      </label>
      <button id="ask">Ask</button>
    </section>

    <section class="section">
      <h2>Final Answer</h2>
      <div id="answer" class="answer">Select an identity and ask a question.</div>
      <div id="meta" class="meta"></div>
    </section>

    <div class="grid">
      <section class="section">
        <h2>Citations</h2>
        <div id="citations"></div>
      </section>
      <section class="section">
        <h2>Permission / Retrieval Trace</h2>
        <pre id="trace">{}</pre>
      </section>
    </div>

    <section class="section">
      <h2>Retrieved Authorized Chunks</h2>
      <div id="chunks" class="chunks"></div>
    </section>

    <section class="section">
      <h2>All Files in Dataset</h2>
      <div id="files" class="files-grid"></div>
    </section>
  </main>

  <script>
    const askButton = document.getElementById('ask');
    const queryEl = document.getElementById('query');
    const examplesEl = document.getElementById('examples');
    const answerEl = document.getElementById('answer');
    const metaEl = document.getElementById('meta');
    const citationsEl = document.getElementById('citations');
    const chunksEl = document.getElementById('chunks');
    const traceEl = document.getElementById('trace');
    const filesEl = document.getElementById('files');
    const sampleQueries = ["What improvement in throughput did Alpha show?", "Summarize confidential design choices for Project Alpha", "What does Project Alpha say about differential privacy?", "What future experiments does the Alpha literature review suggest?", "List intern observations on Beta project"];

    function escapeHtml(value) {
      return String(value ?? '').replace(/[&<>"']/g, ch => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#039;'
      }[ch]));
    }

    function renderExamples() {
      examplesEl.innerHTML = sampleQueries.map(q => `
        <button type="button" class="example-option">${escapeHtml(q)}</button>
      `).join('');
      examplesEl.hidden = false;
    }

    function hideExamplesSoon() {
      window.setTimeout(() => { examplesEl.hidden = true; }, 120);
    }

    function renderCitations(citations) {
      if (!citations.length) {
        citationsEl.innerHTML = '<p class="warning">No citations returned.</p>';
        return;
      }
      citationsEl.innerHTML = `
        <table>
          <thead><tr><th>Chunk</th><th>Document</th><th>Score</th></tr></thead>
          <tbody>
            ${citations.map(c => `
              <tr>
                <td>${escapeHtml(c.chunk_id)}</td>
                <td>${escapeHtml(c.doc_id)}</td>
                <td>${Number(c.score || 0).toFixed(4)}</td>
              </tr>
            `).join('')}
          </tbody>
        </table>
      `;
    }

    function renderChunks(chunks) {
      if (!chunks.length) {
        chunksEl.innerHTML = '<p class="warning">No authorized chunks were retrieved.</p>';
        return;
      }
      chunksEl.innerHTML = chunks.map(c => `
        <article class="chunk">
          <strong>${escapeHtml(c.rank + 1)}. ${escapeHtml(c.title)} <span class="pill">${escapeHtml(c.chunk_id)}</span></strong>
          <div>${escapeHtml(c.text)}</div>
          <div class="meta">
            <span class="pill">doc: ${escapeHtml(c.doc_id)}</span>
            <span class="pill">score: ${Number(c.score || 0).toFixed(4)}</span>
            <span class="pill">sensitivity: ${escapeHtml(c.metadata?.sensitivity)}</span>
            <span class="pill">source: ${escapeHtml(c.metadata?.source)}</span>
          </div>
        </article>
      `).join('');
    }

    function renderFiles(files) {
      if (!files.length) {
        filesEl.innerHTML = '<p class="warning">No files found.</p>';
        return;
      }
      filesEl.innerHTML = files.map(file => {
        const roles = file.allowed_roles?.length ? file.allowed_roles.join(', ') : 'public';
        const users = file.allowed_users?.length ? file.allowed_users.join(', ') : 'none';
        return `
          <article class="file-card">
            <div class="file-title">
              ${escapeHtml(file.title)} <span class="pill">${escapeHtml(file.doc_id)}</span>
            </div>
            <div class="meta">
              <span class="pill">source: ${escapeHtml(file.source)}</span>
              <span class="pill">project: ${escapeHtml(file.project)}</span>
              <span class="pill">sensitivity: ${escapeHtml(file.sensitivity)}</span>
              <span class="pill">roles: ${escapeHtml(roles)}</span>
              <span class="pill">users: ${escapeHtml(users)}</span>
            </div>
            <div class="file-preview">${escapeHtml(file.text_preview)}</div>
          </article>
        `;
      }).join('');
    }

    async function loadFiles() {
      try {
        const response = await fetch('/api/files');
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Failed to load files');
        renderFiles(data.files || []);
      } catch (error) {
        filesEl.innerHTML = `<p class="warning">${escapeHtml(error.message)}</p>`;
      }
    }

    async function ask() {
      askButton.disabled = true;
      answerEl.textContent = 'Loading...';
      try {
        const response = await fetch('/api/answer', {
          method: 'POST',
          headers: {'Content-Type': 'application/json'},
          body: JSON.stringify({
            identity: document.getElementById('identity').value,
            query: queryEl.value,
            top_k: Number(document.getElementById('topK').value)
          })
        });
        const data = await response.json();
        if (!response.ok) throw new Error(data.error || 'Request failed');
        answerEl.textContent = data.answer;
        const llmMode = data.llm?.used_ai_api ? `AI API: ${data.llm.model}` : 'AI API: offline fallback';
        metaEl.innerHTML = `
          <span class="pill">identity: ${escapeHtml(data.trace.user.role)}</span>
          <span class="pill">filtered docs: ${escapeHtml(data.filtered_out_docs)}</span>
          <span class="pill">retrieved chunks: ${escapeHtml(data.retrieved_chunks.length)}</span>
          <span class="pill">${escapeHtml(llmMode)}</span>
        `;
        renderCitations(data.citations || []);
        renderChunks(data.retrieved_chunks || []);
        traceEl.textContent = JSON.stringify({
          trace: data.trace,
          llm: data.llm
        }, null, 2);
      } catch (error) {
        answerEl.textContent = error.message;
      } finally {
        askButton.disabled = false;
      }
    }

    queryEl.addEventListener('focus', renderExamples);
    queryEl.addEventListener('click', renderExamples);
    queryEl.addEventListener('input', renderExamples);
    queryEl.addEventListener('blur', hideExamplesSoon);
    examplesEl.addEventListener('mousedown', event => {
      const option = event.target.closest('.example-option');
      if (!option) return;
      queryEl.value = option.textContent.trim();
      examplesEl.hidden = true;
    });
    askButton.addEventListener('click', ask);
    loadFiles();
    ask();
  </script>
</body>
</html>"""


class DemoHandler(BaseHTTPRequestHandler):
    def _send_json(self, data: Dict[str, Any], status: int = 200):
        body = json.dumps(data).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/" or self.path == "/index.html":
            body = _html().encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        if self.path == "/api/users":
            self._send_json({"users": USERS, "sample_queries": SAMPLE_QUERIES})
            return
        if self.path == "/api/files":
            self._send_json({"files": _load_file_index()})
            return
        self._send_json({"error": "Not found"}, status=404)

    def do_POST(self):
        if self.path != "/api/answer":
            self._send_json({"error": "Not found"}, status=404)
            return

        length = int(self.headers.get("Content-Length", "0"))
        try:
            payload = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            self._send_json({"error": "Invalid JSON"}, status=400)
            return

        identity = payload.get("identity", "manager")
        user = USERS.get(identity)
        if not user:
            self._send_json({"error": f"Unknown identity: {identity}"}, status=400)
            return

        query = (payload.get("query") or "").strip()
        if not query:
            self._send_json({"error": "Query is required"}, status=400)
            return

        top_k = int(payload.get("top_k") or 3)
        chunks = _load_chunks()
        result = answer_query(user, query, chunks, top_k=top_k, use_llm=True)
        self._send_json(result)


def run(host: str = "127.0.0.1", port: int = 8000):
    server = ThreadingHTTPServer((host, port), DemoHandler)
    print(f"Permission-Aware RAG demo running at http://{host}:{port}")
    server.serve_forever()


if __name__ == "__main__":
    run()
