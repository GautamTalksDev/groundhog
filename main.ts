/**
 * Groundhog
 *
 * Reads your local Claude Code / Codex / Cursor session history and ranks the
 * chores you keep paying to redo. Local only — never sends data anywhere.
 *
 * @rote-frontmatter
 * ---
 * name: groundhog
 * description: "Groundhog reads your local Claude Code, Codex, and Cursor session files on this machine and tells you which chores you keep paying to redo. It reads local history only, writes nothing unless you ask, and sends nothing — no network, no accounts, no keys. Needs only python3."
 * provenance:
 *   author: groundhog
 * metadata:
 *   rote_version: 0.79.0
 *   version: 0.0.5
 *   status: released
 *   kind: atomic
 *   flow_type: sequential
 *   execution_model: steps_with_presentation
 *   format: typescript
 *   requires_endpoints: []
 *   requires_sessions: false
 *   discoverability:
 *     tags:
 *     - typescript
 *     - local
 *     - python3
 *     - agent-history
 *     - cost
 *     - read-only
 * parameters:
 * - name: days
 *   param_type: integer
 *   required: false
 *   default: 30
 *   description: "How many days of local session history to scan"
 * - name: top
 *   param_type: integer
 *   required: false
 *   default: 3
 *   description: "How many top repeated chores to show"
 * - name: min_runs
 *   param_type: integer
 *   required: false
 *   default: 3
 *   description: "Minimum times a chore must appear before it is listed"
 * - name: redact
 *   param_type: string
 *   required: false
 *   default: "true"
 *   description: "Scrub secret-like strings and truncate evidence (true/false)"
 * steps:
 *   discover_claude:
 *     type: process.exec
 *     timeout_ms: 60000
 *     argv:
 *       - python3
 *       - "@resource{python/steps/discover_claude.py}"
 *       - $days
 *       - artifacts/discover_claude.json
 *   discover_codex:
 *     type: process.exec
 *     timeout_ms: 60000
 *     argv:
 *       - python3
 *       - "@resource{python/steps/discover_codex.py}"
 *       - $days
 *       - artifacts/discover_codex.json
 *   discover_cursor:
 *     type: process.exec
 *     timeout_ms: 60000
 *     argv:
 *       - python3
 *       - "@resource{python/steps/discover_cursor.py}"
 *       - $days
 *       - artifacts/discover_cursor.json
 *   parse:
 *     type: process.exec
 *     timeout_ms: 60000
 *     depends_on: [discover_claude, discover_codex, discover_cursor]
 *     argv:
 *       - python3
 *       - "@resource{python/steps/parse.py}"
 *       - artifacts/discover_claude.json
 *       - artifacts/discover_codex.json
 *       - artifacts/discover_cursor.json
 *       - artifacts/parse.json
 *   intents:
 *     type: process.exec
 *     timeout_ms: 60000
 *     depends_on: [parse]
 *     argv:
 *       - python3
 *       - "@resource{python/steps/intents.py}"
 *       - artifacts/parse.json
 *       - artifacts/intents.json
 *   cluster:
 *     type: process.exec
 *     timeout_ms: 60000
 *     depends_on: [intents]
 *     argv:
 *       - python3
 *       - "@resource{python/steps/cluster.py}"
 *       - artifacts/intents.json
 *       - $min_runs
 *       - artifacts/cluster.json
 *   cost:
 *     type: process.exec
 *     timeout_ms: 60000
 *     depends_on: [cluster]
 *     argv:
 *       - python3
 *       - "@resource{python/steps/cost.py}"
 *       - artifacts/cluster.json
 *       - artifacts/cost.json
 *   rank:
 *     type: process.exec
 *     timeout_ms: 60000
 *     depends_on: [cost]
 *     argv:
 *       - python3
 *       - "@resource{python/steps/rank.py}"
 *       - artifacts/cost.json
 *       - artifacts/rank.json
 *   report:
 *     type: process.exec
 *     timeout_ms: 60000
 *     depends_on: [rank, parse]
 *     argv:
 *       - python3
 *       - "@resource{python/steps/report.py}"
 *       - artifacts/rank.json
 *       - $days
 *       - $top
 *       - $min_runs
 *       - $redact
 *       - artifacts/report.json
 *       - artifacts/parse.json
 * ---
 */

const { FlowOutput, isProcessExecBody, loadPresentationContext, stepName } =
  await import("__ROTE_PRESENTATION_SDK__");

const out = new FlowOutput();
const ctx = await loadPresentationContext();

if (ctx.run.status === "failed") {
  out.human("Groundhog could not finish. Check NOT COUNTED / step errors.");
  out.summary("groundhog failed");
  out.result({ ok: false, status: ctx.run.status });
} else {
  const step = ctx.requireAvailable(stepName("report"));
  if (!isProcessExecBody(step.body)) {
    throw new Error("report did not record a process.exec observation");
  }
  if (step.body.status.exit.kind !== "code" || step.body.status.exit.code !== 0) {
    throw new Error(
      `report failed: ${step.body.stderr?.text ?? "no stderr captured"}`,
    );
  }
  const raw = step.body.stdout?.text ?? "";
  let payload: { text?: string; json?: unknown } = {};
  try {
    payload = JSON.parse(raw.trim().split("\n").filter(Boolean).pop() || "{}");
  } catch {
    payload = { text: raw };
  }
  const text = typeof payload.text === "string" ? payload.text : raw;
  out.human(text.trimEnd());
  const n =
    payload.json &&
    typeof payload.json === "object" &&
    payload.json !== null &&
    Array.isArray((payload.json as { candidates?: unknown }).candidates)
      ? (payload.json as { candidates: unknown[] }).candidates.length
      : 0;
  out.summary(
    n > 0
      ? `Groundhog found ${n} repeated chore${n === 1 ? "" : "s"} in local agent history.`
      : "Groundhog found no repeated chores in this window.",
  );
  out.result(payload.json ?? { text });
}
