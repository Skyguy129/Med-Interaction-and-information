// ============================================================================
// Medication Interaction Checker: frontend
// Talks to the FastAPI backend (python -m ddi_backend serve).
// No frameworks, no build step. Everything the backend sends is inserted with
// textContent (never innerHTML), so drug names or notes can never inject HTML.
// ============================================================================

// 1) CHANGE THIS if your backend runs somewhere else.
const API_BASE = "http://127.0.0.1:8000";

const SAMPLE_QUESTIONS = [
  "Which drugs should I avoid with warfarin?",
  "Which pain reliever should I not take with warfarin?",
  "Which drugs interact with warfarin and cause bleeding?",
  "Is it safe for a child to take ibuprofen?",
];

// ---------- small helpers ----------
const $ = (selector, root = document) => root.querySelector(selector);

// Build a DOM element safely: el("div", {class: "x"}, "text", otherElement)
function el(tag, attrs = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attrs)) {
    if (value === null || value === undefined || value === false) continue;
    if (key === "class") node.className = value;
    else if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child.nodeType ? child : document.createTextNode(String(child)));
  }
  return node;
}

// Call the backend. Returns {ok, status, data}. Throws only if the server is unreachable.
async function api(path, options = {}) {
  let response;
  try {
    response = await fetch(API_BASE + path, {
      headers: { "Content-Type": "application/json" },
      ...options,
    });
  } catch (err) {
    throw new Error(
      `Cannot reach the backend at ${API_BASE}. Is it running ("python -m ddi_backend serve") and is DDI_CORS_ORIGINS set?`
    );
  }
  let data = null;
  try { data = await response.json(); } catch (_) { /* non-JSON body */ }
  return { ok: response.ok, status: response.status, data };
}

// FastAPI validation errors come back as {detail: [...]}; turn them into one sentence.
function errorText(result) {
  const detail = result.data && result.data.detail;
  if (typeof detail === "string") return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg).join("; ");
  return `The backend returned an error (HTTP ${result.status}).`;
}

function sevClass(severity) {
  return ["major", "moderate", "minor"].includes((severity || "").toLowerCase())
    ? severity.toLowerCase() : "";
}

function showError(container, message) {
  container.replaceChildren(el("div", { class: "card result warn" }, el("p", { class: "msg err" }, message)));
}

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((tab) => {
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".panel").forEach((p) => (p.hidden = p.id !== `tab-${tab.dataset.tab}`));
    if (tab.dataset.tab === "queue") loadQueue();
  });
});

// ---------- backend health ----------
async function checkHealth() {
  const box = $("#health");
  try {
    const r = await api("/health");
    box.className = r.ok ? "status ok" : "status bad";
    $("#health-text").textContent = r.ok ? "Backend connected" : "Backend error";
  } catch (err) {
    box.className = "status bad";
    $("#health-text").textContent = "Backend offline";
  }
}

// ============================================================================
// TAB 1: CHECK MEDICATIONS
// ============================================================================
const currentList = $("#current-list");

function addMedRow(name = "", dose = "") {
  const row = el("div", { class: "row med-row" },
    el("input", { type: "text", class: "med-name", placeholder: "Drug name, e.g. Warfarin", "aria-label": "Current medication name", value: name }),
    el("input", { type: "text", class: "dose med-dose", placeholder: "Dose (optional)", "aria-label": "Current medication dose", value: dose }),
    el("button", { type: "button", class: "icon", title: "Remove", "aria-label": "Remove this medication",
      onclick: () => { row.remove(); if (!currentList.children.length) addMedRow(); } }, "✕")
  );
  currentList.append(row);
}

// The backend accepts a plain name or {name, dose}.
function medPayload(name, dose) {
  name = name.trim(); dose = dose.trim();
  return dose ? { name, dose } : name;
}

$("#add-med").addEventListener("click", () => addMedRow());

$("#example-btn").addEventListener("click", () => {
  currentList.replaceChildren();
  addMedRow("Warfarin", "5 mg");
  addMedRow("Omeprazole");
  $("#new-name").value = "Ibuprofen";
  $("#new-dose").value = "";
  runCheck();
});

$("#check-form").addEventListener("submit", (event) => { event.preventDefault(); runCheck(); });

async function runCheck() {
  const results = $("#check-results");
  const current = [...document.querySelectorAll(".med-row")]
    .map((row) => medPayload($(".med-name", row).value, $(".med-dose", row).value))
    .filter((m) => (typeof m === "string" ? m : m.name));
  const newMed = medPayload($("#new-name").value, $("#new-dose").value);
  if (!(typeof newMed === "string" ? newMed : newMed.name)) {
    showError(results, "Please enter the new medication.");
    return;
  }

  const button = $("#check-btn");
  button.disabled = true; button.textContent = "Checking…";
  try {
    const r = await api("/check", {
      method: "POST",
      body: JSON.stringify({
        current, new: newMed,
        options: { include_current_pairs: $("#include-pairs").checked },
      }),
    });
    if (!r.ok) showError(results, errorText(r));
    else renderCheck(r.data);
  } catch (err) {
    showError(results, err.message);
  } finally {
    button.disabled = false; button.textContent = "Check interactions";
  }
}

function renderCheck(data) {
  const out = $("#check-results");
  out.replaceChildren();

  const s = data.summary || {};
  out.append(el("div", { class: "summary" },
    el("span", {}, el("strong", {}, s.pairs_checked ?? 0), " pair(s) checked"),
    s.major_documented ? el("span", {}, el("strong", {}, s.major_documented), " documented ", el("strong", {}, "Major"), " interaction(s)") : null,
  ));

  if (!data.results || !data.results.length) {
    out.append(el("div", { class: "card" }, el("p", { class: "empty" },
      "No pairs to check. Add at least one current medication and a new medication.")));
  } else {
    data.results.forEach((r) => out.append(pairCard(r)));
  }

  (data.notes || []).forEach((n) => out.append(el("p", { class: "hint" }, n)));
}

// One card per medication pair. The status decides how it looks.
function pairCard(r) {
  const title = (r.pair || []).join("  +  ");

  // --- documented interaction ---
  if (r.status === "documented_interaction") {
    const sev = sevClass(r.severity);
    const card = el("article", { class: `card result ${sev}` },
      el("div", { class: "result-head" },
        el("p", { class: "pairname" }, title),
        el("span", { class: `badge ${sev}` }, `${r.severity} interaction`)),
    );

    if (r.severity_conflict) {
      card.append(el("div", { class: "callout danger" },
        el("strong", {}, "Conflicting records. "),
        `Our data lists this pair with different severities (${(r.severities || []).join(" / ")}). The highest, ${r.severity}, is shown.`));
    }

    const facts = el("dl", { class: "facts" });
    if (r.mechanism) facts.append(el("dt", {}, "Mechanism"), el("dd", {}, r.mechanism));
    if (r.effect) facts.append(el("dt", {}, "Clinical effect"), el("dd", {}, r.effect));
    const doses = Object.entries(r.doses || {}).filter(([, d]) => d);
    if (doses.length) facts.append(el("dt", {}, "Doses you entered"), el("dd", {}, doses.map(([n, d]) => `${n} ${d}`).join(", "), " (not evaluated)"));
    card.append(facts);

    card.append(...alternativesBlock(r.alternatives));

    if (r.needs_review) {
      card.append(el("div", { class: "chips" },
        el("span", { class: "chip flag" }, "Flagged for pharmacist review"),
        ...(r.review_reasons || []).map((x) => el("span", { class: "chip" }, x))));
    }

    card.append(sourcesBlock(r));
    card.append(el("div", { class: "actions" },
      el("button", { type: "button", class: "ghost", onclick: () => openReviewDialog(r.pair) }, "Request pharmacist review (simulated)")));
    return card;
  }

  // --- not documented: NEVER shown as "safe" ---
  if (r.status === "no_documented_interaction") {
    return el("article", { class: "card result" },
      el("div", { class: "result-head" },
        el("p", { class: "pairname" }, title),
        el("span", { class: "badge" }, "Not documented")),
      el("div", { class: "callout" },
        el("strong", {}, "Not found in our records. "),
        "This does not mean the combination is safe. Our data covers a limited set of drugs and pairs. Ask a pharmacist or clinician."));
  }

  // --- evidence excluded by the eligibility filter ---
  if (r.status === "evidence_ineligible") {
    return el("article", { class: "card result warn" },
      el("div", { class: "result-head" },
        el("p", { class: "pairname" }, title),
        el("span", { class: "badge moderate" }, "Evidence not eligible")),
      el("p", {}, r.message || r.text || "The only evidence for this pair did not pass our source checks."),
      el("div", { class: "callout" }, "This is not the same as \"no interaction\". Ask a pharmacist or clinician."));
  }

  // --- unknown / ambiguous drug names: offer clickable suggestions ---
  const card = el("article", { class: "card result warn" },
    el("div", { class: "result-head" },
      el("p", { class: "pairname" }, title),
      el("span", { class: "badge moderate" }, r.status === "ambiguous_name" ? "Name is ambiguous" : "Drug not in our data")),
    el("p", {}, r.message || r.text || "We could not match one of the names."));
  (r.unresolved || []).forEach((u) => {
    const candidates = u.candidates || u.did_you_mean || [];
    if (candidates.length) {
      card.append(el("p", { class: "hint" }, `Did you mean (for "${u.query}")? Click one to re-run:`),
        el("div", { class: "chips" }, candidates.map((c) =>
          el("button", { type: "button", class: "chip", onclick: () => applySuggestion(u.query, c) }, c))));
    }
  });
  card.append(el("p", { class: "hint" }, "We never guess a drug name. Brand names and abbreviations are not in our data."));
  return card;
}

// Replace the misspelled name in the form with the clicked suggestion, then re-run.
function applySuggestion(query, suggestion) {
  const q = query.toLowerCase();
  const inputs = [$("#new-name"), ...document.querySelectorAll(".med-name")];
  const target = inputs.find((i) => i.value.toLowerCase().includes(q));
  if (target) { target.value = suggestion; runCheck(); }
}

function alternativesBlock(alternatives) {
  const seen = new Set();
  const blocks = [];
  (alternatives || []).forEach((a) => {
    const key = (a.text || "").toLowerCase().trim(); // same alternative from repeated records shows once
    if (seen.has(key)) return; // the backend can list the same alternative from repeated records
    seen.add(key);
    const warnings = [];
    (a.cross_check || []).forEach((c) => {
      if (c.warning) warnings.push(String(c.warning).replace(/^warning:\s*/i, ""));
      else if (c.in_data === false) warnings.push(`${c.alternative} is not in our data, so it could not be checked. That does not mean it is safe.`);
    });
    blocks.push(el("div", { class: "alt" },
      el("div", { class: "kind" }, a.type === "management_action" ? "Suggested action (as written in our data)" : "Suggested alternative (as written in our data)"),
      el("div", {}, el("strong", {}, a.text)),
      a.rationale ? el("div", { class: "why" }, a.rationale) : null,
      ...warnings.map((w) => el("div", { class: "callout" }, el("strong", {}, "Warning: "), w))));
  });
  return blocks;
}

function sourcesBlock(r) {
  const citations = (r.source && r.source.citations) || [];
  const ids = (r.source && r.source.record_ids) || (r.records || []).map((x) => x.id);
  const box = el("div", { class: "sources" }, "Source: ");
  if (!ids.length) { box.append("none"); return box; }
  ids.forEach((id, i) => {
    const ref = (citations.find((c) => c.id === id) || {}).reference;
    box.append(el("code", {}, id), ref ? ` (${ref}) ` : " ");
    if (i < ids.length - 1) box.append("· ");
  });
  return box;
}

// ============================================================================
// TAB 2: ASK A QUESTION
// ============================================================================
SAMPLE_QUESTIONS.forEach((q) => {
  $("#sample-questions").append(el("button", { type: "button", class: "chip",
    onclick: () => { $("#question").value = q; runAsk(); } }, q));
});

$("#ask-form").addEventListener("submit", (event) => { event.preventDefault(); runAsk(); });

async function runAsk() {
  const results = $("#ask-results");
  const question = $("#question").value.trim();
  if (!question) return;
  const button = $("#ask-btn");
  button.disabled = true; button.textContent = "Asking…";
  try {
    const r = await api("/ask", { method: "POST", body: JSON.stringify({ question }) });
    if (!r.ok) showError(results, errorText(r));
    else renderAsk(r.data);
  } catch (err) {
    showError(results, err.message);
  } finally {
    button.disabled = false; button.textContent = "Ask";
  }
}

function renderAsk(d) {
  const out = $("#ask-results");
  out.replaceChildren();
  const card = el("article", { class: "card" }, el("h2", {}, d.question || "Answer"));

  if (d.status === "needs_clarification") {
    card.classList.add("result", "warn");
    (d.clarifications || []).forEach((c) => card.append(el("div", { class: "callout" }, c.question)));
    if (!(d.clarifications || []).length) card.append(el("div", { class: "callout" }, "We need a little more detail to answer. Please rephrase using exact generic drug names."));
  }

  if (d.clarifying_question) card.append(el("div", { class: "callout" }, d.clarifying_question));

  // "which drugs should I avoid with X?"
  if (d.avoid && d.avoid.status === "resolved") {
    card.append(el("p", {}, `Documented Major and Moderate interactions with `, el("strong", {}, d.avoid.drug), ":"));
    for (const level of ["Major", "Moderate"]) {
      const items = d.avoid.avoid.filter((a) => a.severity === level);
      if (!items.length) continue;
      const group = el("div", { class: "avoid-group" },
        el("h4", {}, el("span", { class: `badge ${level.toLowerCase()}` }, level), ` (${items.length})`));
      items.forEach((a) => group.append(el("div", { class: "avoid-item" },
        el("div", {}, el("strong", {}, a.drug), a.severity_conflict ? el("span", { class: "chip flag" }, "conflicting records") : null,
          el("div", { class: "effect" }, a.effect || "")),
        el("code", {}, (a.record_ids || []).join(", ")))));
      card.append(group);
    }
  } else if (d.avoid && d.avoid.message) {
    card.append(el("div", { class: "callout" }, d.avoid.message));
  }

  // "tell me about drug X" style answers
  if (d.intent === "drug" && Array.isArray(d.interactions)) {
    card.append(el("p", {}, `Documented interactions involving `, el("strong", {}, d.drug), ` (${d.interactions.length}):`));
    d.interactions.forEach((i) => {
      const sev = sevClass(i.severity);
      card.append(el("div", { class: "avoid-item" },
        el("div", {}, el("strong", {}, `${i.drug_a} + ${i.drug_b}`),
          el("div", { class: "effect" }, [i.mechanism, i.effect].filter(Boolean).join(" · "))),
        el("div", {}, el("span", { class: `badge ${sev}` }, i.severity), " ", el("code", {}, i.id))));
    });
  }

  // a pair answer: reuse the same cards as the checker
  if (Array.isArray(d.results)) d.results.forEach((r) => card.append(pairCard(r)));

  if ((d.phenotype_records || []).length) {
    card.append(el("details", {},
      el("summary", {}, `Symptom terms used to find matches (${d.phenotype_records.length})`),
      el("p", {}, "Matches were found in the interaction text; these terms only helped search."),
      el("ul", {}, d.phenotype_records.slice(0, 15).map((p) => el("li", {}, `${p.label} (${p.hp_id})`)))));
  }

  if ((d.notes || []).length) card.append(el("ul", { class: "notes" }, d.notes.map((n) => el("li", {}, n))));

  card.append(el("details", {}, el("summary", {}, "How this was answered"),
    el("div", { class: "chips" }, (d.rules_fired || []).map((r) => el("span", { class: "chip" }, r)))));
  out.append(card);
}

// ============================================================================
// REVIEW QUEUE (simulated) + approval dialog
// ============================================================================
const dialog = $("#review-dialog");
let reviewPair = null;

function openReviewDialog(pair) {
  reviewPair = pair;
  $("#review-pair").textContent = pair.join("  +  ");
  $("#review-approve").checked = false;
  $("#review-msg").textContent = ""; $("#review-msg").className = "msg";
  dialog.showModal();
}
$("#review-cancel").addEventListener("click", () => dialog.close());

$("#review-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const msg = $("#review-msg");
  const approved = $("#review-approve").checked;
  if (!approved) { msg.className = "msg err"; msg.textContent = "Tick the approval box first. Nothing is staged without your approval."; return; }
  const requestId = (crypto.randomUUID ? crypto.randomUUID() : `req-${Date.now()}`);
  try {
    const r = await api("/review-queue/requests", {
      method: "POST",
      body: JSON.stringify({ drug_pair: reviewPair, request_id: requestId, approved }),
    });
    const d = r.data || {};
    if (r.status === 201) { msg.className = "msg ok"; msg.textContent = "Staged in the SIMULATED review queue. Nothing reached a real person."; }
    else if (r.status === 404) { msg.className = "msg err"; msg.textContent = "Not staged: there is no documented, eligible interaction to review for this pair."; }
    else { msg.className = "msg err"; msg.textContent = `Not staged (${d.status || r.status}). ${d.note || errorText(r)}`; }
  } catch (err) {
    msg.className = "msg err"; msg.textContent = err.message;
  }
});

async function loadQueue() {
  const list = $("#queue-list");
  try {
    const r = await api("/review-queue");
    const rows = (r.data && r.data.review_queue) || [];
    list.replaceChildren();
    if (!rows.length) { list.append(el("p", { class: "empty" }, "The simulated queue is empty.")); return; }
    rows.forEach((row) => list.append(el("div", { class: "queue-item" },
      el("div", { class: "pairname" }, (row.drug_pair || []).join("  +  ")),
      el("div", { class: "meta" }, `Evidence ${row.evidence_id} · ${row.purpose} · staged ${row.staged_at || "unknown"} · request ${row.request_id}`),
      el("span", { class: "chip flag" }, "SIMULATED"))));
  } catch (err) {
    list.replaceChildren(el("p", { class: "msg err" }, err.message));
  }
}
$("#refresh-queue").addEventListener("click", loadQueue);

// ---------- start ----------
$("#api-url").textContent = API_BASE;
addMedRow();
checkHealth();