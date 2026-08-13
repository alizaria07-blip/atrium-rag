const $ = (id) => document.getElementById(id);
const DEFAULTS = {
  base_url: "https://openrouter.ai/api/v1",
  api_key: "",
  chat_model: "openai/gpt-4o-mini",
  embed_model: "local",
  top_k: 5,
  temperature: 0.3,
};

const SUGGESTIONS = [
  { kind: "Library", text: "What does the handbook say about parental leave?" },
  { kind: "Warehouse", text: "Which customers generated the most revenue?" },
  { kind: "Both", text: "Compare refund policy language with actual refunds this year." },
  { kind: "Analysis", text: "Break down open orders by region and channel." },
];

let settings = loadSettings();
let mode = "auto";
let documents = [];
let databases = [];
let selectedDb = null;
let schemaCache = {};
let threads = loadThreads();
let activeId = threads[0]?.id || newThread().id;
let sending = false;

function loadSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem("atrium_settings") || "{}");
    return { ...DEFAULTS, ...stored };
  } catch {
    return { ...DEFAULTS };
  }
}

function persistSettings() {
  settings = {
    base_url: $("baseUrl").value.trim() || DEFAULTS.base_url,
    api_key: $("apiKey").value.trim(),
    chat_model: $("chatModel").value.trim() || DEFAULTS.chat_model,
    embed_model: $("embedModel").value.trim() || DEFAULTS.embed_model,
    top_k: parseInt($("topK").value, 10) || 5,
    temperature: parseFloat($("temperature").value) || 0.3,
  };
  localStorage.setItem("atrium_settings", JSON.stringify(settings));
}

function loadThreads() {
  try {
    const raw = JSON.parse(localStorage.getItem("atrium_threads") || "[]");
    return Array.isArray(raw) && raw.length ? raw : [];
  } catch {
    return [];
  }
}

function saveThreads() {
  localStorage.setItem("atrium_threads", JSON.stringify(threads.slice(0, 40)));
}

function newThread() {
  const t = { id: crypto.randomUUID(), title: "New conversation", created: Date.now(), messages: [] };
  threads.unshift(t);
  saveThreads();
  return t;
}

function activeThread() {
  return threads.find((t) => t.id === activeId) || threads[0];
}

function escapeHtml(s) {
  return String(s).replace(/[&<>"']/g, (c) => (
    { "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]
  ));
}

function toast(msg) {
  $("toastMsg").textContent = msg;
  $("toast").hidden = false;
}
function hideToast() {
  $("toast").hidden = true;
}

function setStatus(text, ok) {
  const el = $("status");
  el.classList.toggle("ok", !!ok);
  el.querySelector("span").textContent = text;
  const meta = $("sideApiMeta");
  if (meta) meta.textContent = ok ? "connected" : "offline";
}

function fillSettings() {
  $("baseUrl").value = settings.base_url || DEFAULTS.base_url;
  $("apiKey").value = settings.api_key || "";
  $("chatModel").value = settings.chat_model || DEFAULTS.chat_model;
  $("embedModel").value = settings.embed_model || DEFAULTS.embed_model;
  $("topK").value = settings.top_k || 5;
  $("temperature").value = settings.temperature ?? 0.3;
  setStatus(settings.api_key ? "Key saved" : "Add an API key", !!settings.api_key);
}

function kindOf(name) {
  const ext = (name.split(".").pop() || "").toLowerCase();
  if (ext === "pdf") return "pdf";
  if (ext === "md" || ext === "markdown") return "md";
  if (ext === "docx" || ext === "doc") return "docx";
  return "txt";
}

function fmtTime(ts) {
  if (!ts) return "";
  const d = new Date(ts * (ts < 1e12 ? 1000 : 1));
  return d.toLocaleDateString(undefined, { month: "short", day: "numeric" });
}

function setView(name) {
  const app = document.getElementById("app");
  app.dataset.view = name;
  setInspector(false);
  document.querySelectorAll(".rail [data-nav]").forEach((b) => b.classList.toggle("on", b.dataset.nav === name));
  document.querySelectorAll(".side-pane").forEach((p) => {
    p.hidden = p.dataset.side !== name;
  });
  document.querySelectorAll(".view").forEach((v) => {
    v.hidden = v.id !== "view-" + name;
  });
  const labels = { ask: "Ask anything", library: "Documents", data: "Databases", settings: "Connection" };
  $("tagline").textContent = labels[name] || "Ask anything";
}

function setInspector(open) {
  const app = $("app");
  const visible = !!open && app.dataset.view === "ask";
  app.dataset.inspector = visible ? "open" : "closed";
  $("inspectorToggle").setAttribute("aria-expanded", String(visible));
  $("inspectorToggle").title = visible ? "Hide grounding details" : "Show grounding details";
}

function deleteThread(id) {
  threads = threads.filter((t) => t.id !== id);
  if (!threads.length) {
    const fresh = newThread();
    activeId = fresh.id;
  } else if (activeId === id) {
    activeId = threads[0].id;
  }
  saveThreads();
  renderThreads();
  renderTranscript();
  const last = activeThread()?.messages?.filter((m) => m.role === "assistant")?.at(-1);
  renderInspector(last || null);
}

function renderThreads() {
  const box = $("threadList");
  if (!threads.length) {
    box.innerHTML = '<p class="muted">No conversations yet.</p>';
    return;
  }
  box.innerHTML = threads.map((t) => `
    <div class="thread ${t.id === activeId ? "on" : ""}" data-id="${t.id}" role="button" tabindex="0">
      <div class="thread-content">
        <b>${escapeHtml(t.title)}</b>
        <time>${fmtTime(t.created / 1000)}</time>
      </div>
      <button type="button" class="thread-del" data-del-thread="${t.id}" title="Delete conversation" aria-label="Delete conversation">
        <svg viewBox="0 0 24 24" aria-hidden="true"><path d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" stroke="currentColor" stroke-width="1.6" fill="none" stroke-linecap="round" stroke-linejoin="round"/></svg>
      </button>
    </div>
  `).join("");
}

function renderPills() {
  $("libPill").textContent = `Documents · ${documents.length}`;
  $("libCount").textContent = documents.length
    ? `${documents.length} document${documents.length === 1 ? "" : "s"}`
    : "No documents yet";
  if (!databases.length) {
    $("dbPill").textContent = "Database · none";
    $("dbCount").textContent = "No database attached";
    $("sideDbMeta").textContent = "none";
  } else {
    const current = databases.find((d) => d.id === selectedDb) || databases[0];
    $("dbPill").textContent = `Database · ${current.name}`;
    $("dbCount").textContent = `${databases.length} attached`;
    $("sideDbMeta").textContent = `${databases.length} attached`;
  }
  const sel = $("dbSelect");
  if (databases.length) {
    sel.hidden = false;
    sel.innerHTML = databases.map((d) =>
      `<option value="${d.id}" ${d.id === selectedDb ? "selected" : ""}>${escapeHtml(d.name)}</option>`
    ).join("");
  } else {
    sel.hidden = true;
  }
}

function renderDocs() {
  const q = ($("docSearch").value || "").toLowerCase();
  const list = documents.filter((d) => (d.name || "").toLowerCase().includes(q));
  const grid = $("docGrid");
  if (!list.length) {
    grid.innerHTML = `
      <div class="empty-block" style="grid-column:1/-1">
        <p class="serif">${documents.length ? "Nothing matches." : "The shelves are empty."}</p>
        <p>Drop a handbook, memo, or contract to start a company library.</p>
      </div>`;
    return;
  }
  grid.innerHTML = list.map((d) => {
    const k = kindOf(d.name);
    return `
      <article class="doc-card" data-id="${d.id}">
        <span class="kind ${k}">${k}</span>
        <h3>${escapeHtml(d.name)}</h3>
        <div class="row">
          <span class="meta">${d.chunks ?? "…"} passages · ${fmtTime(d.added_at)}</span>
          <button type="button" class="del" data-del="${d.id}">Remove</button>
        </div>
      </article>`;
  }).join("");
}

function renderDbList() {
  const box = $("dbList");
  if (!databases.length) {
    box.innerHTML = `
      <div class="empty-block">
        <p class="serif">No database yet.</p>
        <p>Upload a SQLite file or load the Northline sample to explore the agent.</p>
      </div>`;
    return;
  }
  box.innerHTML = databases.map((d) => `
    <button type="button" class="db-card ${d.id === selectedDb ? "on" : ""}" data-db="${d.id}">
      <b>${escapeHtml(d.name)}</b>
      <span>${escapeHtml(d.engine)} · ${d.kind}</span>
    </button>
    <button type="button" class="del" data-db-del="${d.id}" style="align-self:flex-start;border:0;background:none;color:var(--rose);font-size:12px;padding:0 4px 10px;">Detach</button>
  `).join("");
}

async function renderSchema(dbId) {
  const pane = $("schemaPane");
  if (!dbId) {
    pane.innerHTML = `<div class="empty-block"><p class="serif">Attach a warehouse to inspect its schema.</p><p>SQLite files work immediately. Postgres and MySQL attach with a connection string.</p></div>`;
    return;
  }
  pane.innerHTML = `<div class="empty-block"><p>Reading schema…</p></div>`;
  try {
    if (!schemaCache[dbId]) {
      const r = await fetch("/api/databases/" + dbId + "/schema");
      const j = await r.json();
      if (!r.ok) throw new Error(j.detail || "Could not read schema");
      schemaCache[dbId] = j;
    }
    const spec = schemaCache[dbId];
    const db = databases.find((d) => d.id === dbId);
    pane.innerHTML = `
      <div class="schema-head">
        <h3>${escapeHtml(db?.name || "Database")}</h3>
        <span class="muted">${spec.tables.length} tables</span>
      </div>
      <div class="tables">
        ${spec.tables.map((t, i) =>
          `<button type="button" class="tbl ${i === 0 ? "on" : ""}" data-table="${escapeHtml(t.name)}">${escapeHtml(t.name)}${t.rows != null ? " · " + t.rows : ""}</button>`
        ).join("")}
      </div>
      <div class="cols" id="colPane"></div>`;
    if (spec.tables[0]) showTable(dbId, spec.tables[0].name);
  } catch (e) {
    pane.innerHTML = `<div class="empty-block"><p class="serif">Could not inspect this database.</p><p>${escapeHtml(e.message)}</p></div>`;
  }
}

async function showTable(dbId, table) {
  const spec = schemaCache[dbId];
  const t = spec.tables.find((x) => x.name === table);
  document.querySelectorAll(".tbl").forEach((b) => b.classList.toggle("on", b.dataset.table === table));
  const colPane = $("colPane");
  if (!t) return;
  let preview = "";
  try {
    const r = await fetch(`/api/databases/${dbId}/preview/${encodeURIComponent(table)}`);
    const j = await r.json();
    if (r.ok && j.columns?.length) preview = renderTable(j.columns, j.rows);
  } catch { /* preview is optional */ }
  colPane.innerHTML = `
    <table>
      ${t.columns.map((c) => `<tr><td>${escapeHtml(c.name)}${c.pk ? " · key" : ""}</td><td>${escapeHtml(c.type)}</td></tr>`).join("")}
    </table>
    ${preview ? `<div class="table-wrap" style="margin-top:14px">${preview}</div>` : ""}`;
}

function renderTable(columns, rows) {
  if (!columns?.length) return "";
  return `<table class="data"><thead><tr>${
    columns.map((c) => `<th>${escapeHtml(c)}</th>`).join("")
  }</tr></thead><tbody>${
    rows.map((row) => `<tr>${row.map((c) => `<td>${escapeHtml(c ?? "")}</td>`).join("")}</tr>`).join("")
  }</tbody></table>`;
}

function renderMarkdown(text) {
  let s = escapeHtml(text || "");
  const blocks = [];
  s = s.replace(/```([\s\S]*?)```/g, (_, code) => {
    blocks.push(`<pre>${code}</pre>`);
    return `@@B${blocks.length - 1}@@`;
  });
  s = s.replace(/`([^`]+)`/g, "<code>$1</code>");
  s = s.replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>");
  s = s.replace(/\*([^*]+)\*/g, "<em>$1</em>");
  s = s.replace(/\[(\d+)\]/g, '<sup class="cite" title="Source $1">[$1]</sup>');
  const lines = s.split("\n");
  const cells = (row) => row.trim().replace(/^\|/, "").replace(/\|$/, "").split("|").map((c) => c.trim());
  const isTableRow = (row) => /^\s*\|.*\|\s*$/.test(row);
  const isTableSep = (row) => /^\s*\|[\s:|-]+\|\s*$/.test(row);
  let html = "";
  let list = null;
  const flush = () => {
    if (list) { html += list === "ul" ? "</ul>" : "</ol>"; list = null; }
  };
  let i = 0;
  while (i < lines.length) {
    const line = lines[i];
    if (isTableRow(line) && i + 1 < lines.length && isTableSep(lines[i + 1])) {
      flush();
      html += `<div class="md-table"><table class="data"><thead><tr>${
        cells(line).map((c) => `<th>${c}</th>`).join("")
      }</tr></thead><tbody>`;
      i += 2;
      while (i < lines.length && isTableRow(lines[i]) && lines[i].trim()) {
        html += `<tr>${cells(lines[i]).map((c) => `<td>${c}</td>`).join("")}</tr>`;
        i += 1;
      }
      html += "</tbody></table></div>";
      continue;
    }
    const head = line.match(/^(#{2,4})\s+(.+)/);
    const ul = line.match(/^\s*[-*]\s+(.+)/);
    const ol = line.match(/^\s*\d+\.\s+(.+)/);
    const quote = line.match(/^\s*>\s?(.*)/);
    if (head) {
      flush();
      const level = head[1].length;
      html += `<h${level}>${head[2]}</h${level}>`;
    } else if (quote) {
      flush();
      html += `<blockquote>${quote[1]}</blockquote>`;
    } else if (ul) {
      if (list !== "ul") { flush(); html += "<ul>"; list = "ul"; }
      html += `<li>${ul[1]}</li>`;
    } else if (ol) {
      if (list !== "ol") { flush(); html += "<ol>"; list = "ol"; }
      html += `<li>${ol[1]}</li>`;
    } else if (!line.trim()) {
      flush();
    } else {
      flush();
      html += `<p>${line}</p>`;
    }
    i += 1;
  }
  flush();
  return html.replace(/@@B(\d+)@@/g, (_, i) => blocks[Number(i)]);
}

function emptyAskHtml() {
  return `
    <div class="col empty-ask">
      <p class="kicker">Company intelligence</p>
      <h2>Ask your <em>company data.</em></h2>
      <p class="lead">Upload the documents people already trust. Connect the databases they already run. Then ask — every answer comes back grounded in the source.</p>
      <div class="suggestions">
        ${SUGGESTIONS.map((s) =>
          `<button type="button" class="suggestion" data-suggest="${escapeHtml(s.text)}"><small>${s.kind}</small>${escapeHtml(s.text)}</button>`
        ).join("")}
      </div>
    </div>`;
}

function renderTranscript() {
  const t = activeThread();
  const box = $("transcript");
  if (!t.messages.length) {
    box.innerHTML = emptyAskHtml();
    return;
  }
  box.innerHTML = `<div class="col" id="turns"></div>`;
  const turns = $("turns");
  for (const msg of t.messages) turns.appendChild(renderTurn(msg));
  box.scrollTop = box.scrollHeight;
}

function renderTurn(msg) {
  const wrap = document.createElement("div");
  wrap.className = "turn " + msg.role;
  if (msg.role === "user") {
    wrap.innerHTML = `<div class="bubble">${escapeHtml(msg.content)}</div>`;
    return wrap;
  }
  const steps = (msg.steps || []).map((s) => `<span><i></i>${escapeHtml(s)}</span>`).join("");
  let sql = "";
  if (msg.sql) {
    if (msg.sql.error) {
      sql = `<div class="sql-card"><header>Warehouse query</header><pre>${escapeHtml(msg.sql.sql || "")}</pre><div class="err">${escapeHtml(msg.sql.error)}</div></div>`;
    } else {
      sql = `<div class="sql-card">
        <header><span>Warehouse query · ${msg.sql.row_count} row${msg.sql.row_count === 1 ? "" : "s"}</span></header>
        <pre>${escapeHtml(msg.sql.sql || "")}</pre>
        <div class="table-wrap">${renderTable(msg.sql.columns, msg.sql.rows)}</div>
      </div>`;
    }
  }
  wrap.innerHTML = `
    ${steps ? `<div class="steps">${steps}</div>` : ""}
    ${sql}
    <div class="bubble prose">${msg.pending ? '<span class="cursor"></span>' : renderMarkdown(msg.content)}</div>`;
  if (msg.pending && msg.content) {
    wrap.querySelector(".prose").innerHTML = renderMarkdown(msg.content) + '<span class="cursor"></span>';
  }
  return wrap;
}

function updateStreamingTurn(assistant) {
  const turns = $("turns");
  if (!turns) {
    renderTranscript();
    return;
  }
  const lastTurn = turns.lastElementChild;
  if (lastTurn && lastTurn.classList.contains("assistant")) {
    const prose = lastTurn.querySelector(".prose");
    if (prose) {
      prose.innerHTML = renderMarkdown(assistant.content) + '<span class="cursor"></span>';
    }
    const box = $("transcript");
    box.scrollTop = box.scrollHeight;
  } else {
    renderTranscript();
  }
}

function renderInspector(msg) {
  const body = $("inspectBody");
  if (!msg) {
    body.innerHTML = `<p class="muted">Sources, citations, and any warehouse query will appear here after you ask.</p>`;
    return;
  }
  const src = (msg.sources || []).map((s, i) => `
    <div class="source">
      <b>[${i + 1}] ${escapeHtml(s.source)}</b>
      <p>${escapeHtml(s.text || "Retrieved passage")}</p>
    </div>`).join("");
  body.innerHTML = `
    <div class="inspect-block">
      <h3>Documents</h3>
      ${src || "<p class='muted'>No passages retrieved.</p>"}
    </div>
    <div class="inspect-block">
      <h3>Warehouse</h3>
      ${msg.sql?.sql
        ? `<pre style="font-family:var(--mono);font-size:12px;white-space:pre-wrap;color:var(--steel)">${escapeHtml(msg.sql.sql)}</pre>`
        : "<p class='muted'>No query ran for this turn.</p>"}
    </div>`;
}

async function refreshDocs() {
  try {
    const r = await fetch("/api/documents");
    const j = await r.json();
    documents = j.documents || [];
    renderDocs();
    renderPills();
  } catch { /* keep last known */ }
}

async function refreshDbs() {
  try {
    const r = await fetch("/api/databases");
    const j = await r.json();
    databases = j.databases || [];
    if (!selectedDb || !databases.some((d) => d.id === selectedDb)) {
      selectedDb = databases[0]?.id || null;
    }
    renderDbList();
    renderPills();
    if (document.getElementById("app").dataset.view === "data") {
      renderSchema(selectedDb);
    }
  } catch { /* keep last known */ }
}

function validateForChat() {
  persistSettings();
  const localOnly = settings.base_url.includes("127.0.0.1:59998") || settings.base_url.includes("localhost:59998");
  if (!settings.api_key && !localOnly) {
    toast("Add your API key in Settings before asking.");
    return false;
  }
  if (!settings.base_url || !settings.chat_model) {
    toast("Base URL and chat model are required.");
    return false;
  }
  if ((mode === "database" || mode === "both" || mode === "auto") && !databases.length && mode !== "auto") {
    toast("Attach a database first, or switch search to Documents / Auto.");
    return false;
  }
  return true;
}

function validateForUpload() {
  persistSettings();
  if (!embed_local(settings.embed_model) && (!settings.base_url || !settings.api_key || !settings.embed_model)) {
    toast("Remote embeddings need a base URL, API key, and model. Or set embedding model to <b>local</b>.");
    return false;
  }
  return true;
}

function embed_local(model) {
  return ["", "local", "builtin", "hash", "offline"].includes((model || "").trim().toLowerCase());
}

async function handleFiles(files) {
  if (!validateForUpload()) return;
  hideToast();
  for (const file of files) {
    const fd = new FormData();
    fd.append("file", file);
    fd.append("settings", JSON.stringify({
      base_url: settings.base_url,
      api_key: settings.api_key,
      embed_model: settings.embed_model,
    }));
    try {
      const r = await fetch("/api/documents", { method: "POST", body: fd });
      const j = await r.json();
      if (!r.ok) throw new Error(detail(j));
    } catch (e) {
      toast(`Could not index ${escapeHtml(file.name)}: ${escapeHtml(e.message)}`);
    }
  }
  await refreshDocs();
}

function detail(j) {
  if (!j) return "Request failed";
  if (typeof j.detail === "string") return j.detail;
  return JSON.stringify(j.detail || j);
}

async function ask(text) {
  const question = (text ?? $("question").value).trim();
  if (!question || sending) return;
  if (!validateForChat()) return;
  hideToast();

  const thread = activeThread();
  if (!thread.messages.length) {
    thread.title = question.slice(0, 72);
  }
  thread.messages.push({ role: "user", content: question });
  const assistant = { role: "assistant", content: "", steps: [], sources: [], sql: null, pending: true };
  thread.messages.push(assistant);
  saveThreads();
  $("question").value = "";
  autosize();
  renderThreads();
  renderTranscript();
  sending = true;
  $("sendBtn").disabled = true;

  const history = thread.messages
    .slice(0, -2)
    .filter((m) => m.content && (m.role === "user" || m.role === "assistant"))
    .slice(-10)
    .map((m) => ({ role: m.role, content: m.content }));

  try {
    const resp = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        question,
        history,
        mode,
        database_id: selectedDb,
        settings: {
          base_url: settings.base_url,
          api_key: settings.api_key,
          embed_model: settings.embed_model,
          chat_model: settings.chat_model,
          top_k: settings.top_k,
          temperature: settings.temperature,
        },
      }),
    });
    if (!resp.ok) {
      const j = await resp.json().catch(() => ({}));
      throw new Error(detail(j) || "HTTP " + resp.status);
    }
    const reader = resp.body.getReader();
    const decoder = new TextDecoder();
    let buf = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buf += decoder.decode(value, { stream: true });
      const frames = buf.split("\n\n");
      buf = frames.pop();
      for (const frame of frames) {
        for (const line of frame.split("\n")) {
          if (!line.startsWith("data:")) continue;
          let ev;
          try { ev = JSON.parse(line.slice(5).trim()); } catch { continue; }
          let shouldFullRender = false;
          if (ev.type === "status" && ev.message) {
            assistant.steps.push(ev.message);
            shouldFullRender = true;
          } else if (ev.type === "sql") {
            assistant.sql = ev;
            assistant.steps.push(ev.error ? "Warehouse query failed" : `Queried warehouse · ${ev.row_count} rows`);
            shouldFullRender = true;
          } else if (ev.type === "delta") {
            assistant.content += ev.content;
          } else if (ev.type === "sources") {
            assistant.sources = ev.sources || [];
            shouldFullRender = true;
          } else if (ev.type === "error") {
            assistant.content = assistant.content ? `${assistant.content}\n\nError: ${ev.message}` : `Error: ${ev.message}`;
            shouldFullRender = true;
          }
          if (shouldFullRender) {
            renderTranscript();
          } else {
            updateStreamingTurn(assistant);
          }
        }
      }
    }
  } catch (e) {
    assistant.content = "Error: " + e.message;
  } finally {
    assistant.pending = false;
    saveThreads();
    renderTranscript();
    renderInspector(assistant);
    sending = false;
    $("sendBtn").disabled = false;
    $("question").focus();
  }
}

function autosize() {
  const q = $("question");
  q.style.height = "auto";
  q.style.height = Math.min(q.scrollHeight, 180) + "px";
}

// ---- events ----
document.querySelectorAll(".rail [data-nav]").forEach((btn) => {
  btn.addEventListener("click", () => {
    setView(btn.dataset.nav);
    if (btn.dataset.nav === "data") renderSchema(selectedDb);
  });
});

$("newChat").addEventListener("click", () => {
  activeId = newThread().id;
  renderThreads();
  renderTranscript();
  renderInspector(null);
  $("question").focus();
});

$("threadList").addEventListener("click", (e) => {
  const delBtn = e.target.closest("[data-del-thread]");
  if (delBtn) {
    e.stopPropagation();
    deleteThread(delBtn.dataset.delThread);
    return;
  }
  const item = e.target.closest(".thread");
  if (!item) return;
  activeId = item.dataset.id;
  renderThreads();
  renderTranscript();
  const last = activeThread()?.messages?.filter((m) => m.role === "assistant")?.at(-1);
  renderInspector(last || null);
});

$("threadList").addEventListener("keydown", (e) => {
  if (e.key === "Enter" || e.key === " ") {
    const delBtn = e.target.closest("[data-del-thread]");
    if (delBtn) {
      e.preventDefault();
      e.stopPropagation();
      deleteThread(delBtn.dataset.delThread);
      return;
    }
    const item = e.target.closest(".thread");
    if (item && e.target === item) {
      e.preventDefault();
      activeId = item.dataset.id;
      renderThreads();
      renderTranscript();
      const last = activeThread()?.messages?.filter((m) => m.role === "assistant")?.at(-1);
      renderInspector(last || null);
    }
  }
});

document.querySelectorAll(".mode").forEach((btn) => {
  btn.addEventListener("click", () => {
    mode = btn.dataset.mode;
    document.querySelectorAll(".mode").forEach((b) => b.classList.toggle("on", b === btn));
  });
});

$("dbSelect").addEventListener("change", () => {
  selectedDb = $("dbSelect").value;
  renderPills();
});

$("sendBtn").addEventListener("click", () => ask());
$("question").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    ask();
  }
});
$("question").addEventListener("input", autosize);

$("transcript").addEventListener("click", (e) => {
  const s = e.target.closest("[data-suggest]");
  if (s) ask(s.dataset.suggest);
});

const drop = $("drop");
const fileInput = $("fileInput");
$("libraryUploadBtn").addEventListener("click", () => fileInput.click());
drop.addEventListener("click", () => fileInput.click());
drop.addEventListener("keydown", (e) => { if (e.key === "Enter" || e.key === " ") fileInput.click(); });
["dragenter", "dragover"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.add("drag"); }));
["dragleave", "drop"].forEach((ev) => drop.addEventListener(ev, (e) => { e.preventDefault(); drop.classList.remove("drag"); }));
drop.addEventListener("drop", (e) => handleFiles(e.dataTransfer.files));
fileInput.addEventListener("change", () => { handleFiles(fileInput.files); fileInput.value = ""; });
$("docSearch").addEventListener("input", renderDocs);

$("docGrid").addEventListener("click", async (e) => {
  const id = e.target.dataset.del;
  if (!id) return;
  await fetch("/api/documents/" + id, { method: "DELETE" });
  await refreshDocs();
});

$("dbFileBtn").addEventListener("click", () => $("dbFileInput").click());
$("dbFileInput").addEventListener("change", async () => {
  const file = $("dbFileInput").files[0];
  $("dbFileInput").value = "";
  if (!file) return;
  const fd = new FormData();
  fd.append("file", file);
  fd.append("name", file.name);
  try {
    const r = await fetch("/api/databases", { method: "POST", body: fd });
    const j = await r.json();
    if (!r.ok) throw new Error(detail(j));
    selectedDb = j.database.id;
    await refreshDbs();
    setView("data");
  } catch (e) {
    toast(escapeHtml(e.message));
  }
});

$("dbUrlBtn").addEventListener("click", () => {
  $("urlForm").hidden = !$("urlForm").hidden;
});

$("urlForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  try {
    const r = await fetch("/api/databases/url", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ name: $("dbName").value, url: $("dbUrl").value }),
    });
    const j = await r.json();
    if (!r.ok) throw new Error(detail(j));
    selectedDb = j.database.id;
    $("urlForm").hidden = true;
    $("dbUrl").value = "";
    await refreshDbs();
    setView("data");
  } catch (err) {
    toast(escapeHtml(err.message));
  }
});

$("dbSampleBtn").addEventListener("click", async () => {
  $("dbSampleBtn").disabled = true;
  try {
    const r = await fetch("/api/databases/sample", { method: "POST" });
    const j = await r.json();
    if (!r.ok) throw new Error(detail(j));
    selectedDb = j.database.id;
    await refreshDbs();
    setView("data");
  } catch (e) {
    toast(escapeHtml(e.message));
  } finally {
    $("dbSampleBtn").disabled = false;
  }
});

$("dbList").addEventListener("click", async (e) => {
  const del = e.target.dataset.dbDel;
  if (del) {
    await fetch("/api/databases/" + del, { method: "DELETE" });
    delete schemaCache[del];
    await refreshDbs();
    return;
  }
  const card = e.target.closest("[data-db]");
  if (card) {
    selectedDb = card.dataset.db;
    renderDbList();
    renderPills();
    renderSchema(selectedDb);
  }
});

$("schemaPane").addEventListener("click", (e) => {
  const t = e.target.closest("[data-table]");
  if (t && selectedDb) showTable(selectedDb, t.dataset.table);
});

$("localEmbedBtn").addEventListener("click", () => {
  $("embedModel").value = "local";
  persistSettings();
  $("saveNote").textContent = "Embedding model set to local.";
});

["baseUrl", "apiKey", "chatModel", "embedModel"].forEach((id) => {
  $(id).addEventListener("focus", () => $(id).select());
});

$("settingsForm").addEventListener("submit", async (e) => {
  e.preventDefault();
  persistSettings();
  $("saveBtn").textContent = "Testing…";
  $("saveBtn").disabled = true;
  try {
    const headers = { "Content-Type": "application/json" };
    if (settings.api_key) headers.Authorization = "Bearer " + settings.api_key;
    const r = await fetch(settings.base_url.replace(/\/+$/, "") + "/models", { headers });
    if (!r.ok) throw new Error("HTTP " + r.status);
    setStatus("Connected", true);
    $("saveNote").textContent = "Connection looks good.";
  } catch (err) {
    setStatus("Unreachable", false);
    $("saveNote").textContent = "Could not reach the API: " + err.message;
  } finally {
    $("saveBtn").textContent = "Save & test";
    $("saveBtn").disabled = false;
  }
});

document.querySelectorAll("[data-open-panel]").forEach((button) => {
  button.addEventListener("click", () => {
    const panel = $(button.dataset.openPanel === "api" ? "apiPanel" : "databasePanel");
    if (panel) {
      panel.open = true;
      panel.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }
  });
});

$("toastClose").addEventListener("click", hideToast);
$("inspectorToggle").addEventListener("click", () => {
  setInspector($("app").dataset.inspector !== "open");
  if ($("app").dataset.inspector === "open") {
    const last = activeThread().messages.filter((m) => m.role === "assistant").at(-1);
    renderInspector(last || null);
  }
});
$("inspectorClose").addEventListener("click", () => setInspector(false));

fillSettings();
setView("ask");
renderThreads();
renderTranscript();
refreshDocs();
refreshDbs();
