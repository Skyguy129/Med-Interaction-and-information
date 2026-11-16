"""The guarded agent, ported from the Agents and Guarded Tool Use notebook.

The only permitted write is staging a documented interaction that exists in our
data into a SIMULATED pharmacist-review queue, and only the user can authorise
it. The controller checks the gates in a fixed order and only then lets the tool
run; the tool repeats its own checks at its boundary. This orchestrator is
deterministic: no language model is imported here, and nothing a model writes
can decide or trigger a write.
"""

import json
from copy import deepcopy

from ddi_backend.checker import DOCUMENTED
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE
from ddi_backend.state import PURPOSE

REVIEW_TARGET = "interaction_review_queue"
REVIEW_TOOL = "stage_interaction_review"

# The gate names are the failure codes: the dossier mentions A1 through A6 but never defines them.
GATE_MISSING_EVIDENCE = "missing_evidence"
GATE_TOOL_DISABLED = "tool_disabled"
GATE_TOOL_NOT_ALLOWED = "tool_not_allowed"
GATE_OUT_OF_SCOPE = "out_of_scope"
GATE_APPROVAL_REQUIRED = "approval_required"
GATE_DUPLICATE = "duplicate_request_id"
GATE_ALLOWED = "allowed"
GATES = (GATE_MISSING_EVIDENCE, GATE_TOOL_DISABLED, GATE_TOOL_NOT_ALLOWED,
         GATE_OUT_OF_SCOPE, GATE_APPROVAL_REQUIRED, GATE_DUPLICATE)

STAGED = "staged_in_simulation"
ERROR = "error"


class GuardedAgent:
    def __init__(self, checker, state):
        self.checker = checker
        self.state = state

    # ------------------------------------------------------------------
    # Observe: evidence comes from the same deterministic pair lookup as everything else
    # ------------------------------------------------------------------
    def lookup_interaction(self, drug_pair):
        """The headline documented record for the exact pair, or None.

        The match is exact (a direct pair query), so a partial keyword match can
        never stand in for the requested interaction.
        """
        names = list(drug_pair)
        if len(names) != 2:
            return None
        results = self.checker.check([names[0]], names[1])["results"]
        if not results or results[0]["status"] != DOCUMENTED:
            return None
        return results[0]["records"][0]

    # ------------------------------------------------------------------
    # Condition A: answer-only baseline (never calls a tool)
    # ------------------------------------------------------------------
    def run_baseline(self, case):
        before = self.state.snapshot()
        evidence = self.lookup_interaction(case["drug_pair"])
        if evidence is None:
            recommendation = "No matching database evidence; do not recommend staging a review note."
        else:
            recommendation = "Recommend clinician review of the recorded interaction; no treatment change."
        return {
            "condition": "A",
            "recommendation": recommendation,
            "evidence_id": evidence["id"] if evidence else None,
            "state_before": before,
            "state_after": self.state.snapshot(),
            "executed_tool": None,
        }

    # ------------------------------------------------------------------
    # The tool: repeats the critical checks at its own boundary
    # ------------------------------------------------------------------
    def stage_interaction_review(self, args, approved, enabled):
        before = self.state.snapshot()
        evidence = self.lookup_interaction(args["drug_pair"]) if args.get("evidence_id") else None
        if not enabled:
            result = {"status": "blocked", "reason": GATE_TOOL_DISABLED}
        elif not approved:
            result = {"status": "blocked", "reason": GATE_APPROVAL_REQUIRED}
        elif args.get("target") != REVIEW_TARGET:
            result = {"status": "blocked", "reason": GATE_OUT_OF_SCOPE}
        elif args["request_id"] in self.state.seen_request_ids:
            result = {"status": "blocked", "reason": GATE_DUPLICATE}
        elif not args.get("evidence_id") or evidence is None or evidence["id"] != args["evidence_id"]:
            # The evidence must be a documented interaction that exists in our data for this pair.
            result = {"status": "blocked", "reason": GATE_MISSING_EVIDENCE}
        else:
            note = {
                "evidence_id": args["evidence_id"],
                "drug_pair": list(args["drug_pair"]),
                "request_id": args["request_id"],
                "purpose": PURPOSE,
            }
            try:
                self.state.stage_note(note)  # one transaction: the note and the used ID, or neither
                result = {"status": STAGED, "request_id": args["request_id"]}
            except Exception as error:  # report the failure; never claim success
                result = {"status": ERROR, "reason": f"{type(error).__name__}: {error}",
                          "request_id": args["request_id"]}
        return result, {
            "tool": REVIEW_TOOL,
            "args": deepcopy(args),
            "approved": approved,
            "enabled": enabled,
            "before": before,
            "after": self.state.snapshot(),
            "feedback": deepcopy(result),
        }

    # ------------------------------------------------------------------
    # Condition B: observe -> decide -> permission gate -> tool -> feedback -> update/stop
    # ------------------------------------------------------------------
    def gate(self, case, evidence):
        if evidence is None:
            return GATE_MISSING_EVIDENCE
        if not case["tool_enabled"]:
            return GATE_TOOL_DISABLED
        if case["requested_tool"] != REVIEW_TOOL:
            return GATE_TOOL_NOT_ALLOWED
        if case["target"] != REVIEW_TARGET:
            return GATE_OUT_OF_SCOPE
        if not case["approved"]:
            return GATE_APPROVAL_REQUIRED
        if case["request_id"] in self.state.seen_request_ids:
            return GATE_DUPLICATE
        return GATE_ALLOWED

    def run_guarded(self, case):
        before = self.state.snapshot()
        events = []
        evidence = self.lookup_interaction(case["drug_pair"])
        events.append({"stage": "observe", "evidence_id": evidence["id"] if evidence else None})
        decision = f"request {case['requested_tool']}" if evidence else "stop: no supporting evidence"
        events.append({"stage": "decide", "decision": decision})

        args = {
            "target": case["target"],
            "request_id": case["request_id"],
            "drug_pair": list(case["drug_pair"]),
            "evidence_id": evidence["id"] if evidence else None,
        }
        events.append({"stage": "tool_request", "tool": case["requested_tool"], "args": deepcopy(args)})
        gate = self.gate(case, evidence)
        events.append({"stage": "permission_gate", "result": gate})

        tool_result, trace_entry = None, None
        if gate != GATE_ALLOWED:
            events.append({"stage": "stop", "reason": gate})
            status, executed = gate, None
        else:
            events.append({"stage": "tool", "tool": REVIEW_TOOL, "args": deepcopy(args)})
            tool_result, trace_entry = self.stage_interaction_review(
                args, approved=case["approved"], enabled=case["tool_enabled"])
            events.append({"stage": "feedback", "result": deepcopy(tool_result)})
            events.append({"stage": "update", "review_queue_size": len(self.state.review_queue)})
            events.append({"stage": "stop", "reason": "tool_feedback_received"})
            status = tool_result["status"] if tool_result["status"] in (STAGED, ERROR) else tool_result["reason"]
            executed = REVIEW_TOOL

        after = self.state.snapshot()
        action_occurred = len(after["review_queue"]) == len(before["review_queue"]) + 1
        outcome = {
            "condition": "B",
            "status": status,
            "executed_tool": executed,
            "evidence_id": evidence["id"] if evidence else None,
            "events": events,
            "state_before": before,
            "state_after": after,
            "tool_trace": [trace_entry] if trace_entry else [],
            "request_id": case["request_id"],
            "requested_tool": case["requested_tool"],
            "args": args,
            "gate": gate,
            "tool_result": tool_result,
            "action_occurred": action_occurred,
            "simulated": True,
            "note": SIMULATED_NOTE,
            "disclaimer": DISCLAIMER,
        }
        self.state.log_attempt(case["request_id"], {
            **{k: outcome[k] for k in ("request_id", "requested_tool", "args", "gate", "tool_result",
                                       "state_before", "state_after", "action_occurred", "status", "events")},
            "approved": case["approved"],
            "enabled": case["tool_enabled"],
            "tool_trace_entry": trace_entry,
            "simulated": True,
        })
        return outcome

    def request_review(self, drug_pair, request_id, approved, target=REVIEW_TARGET,
                       tool=REVIEW_TOOL, tool_enabled=True):
        """Convenience wrapper around run_guarded for the CLI and API."""
        return self.run_guarded({
            "drug_pair": tuple(drug_pair), "target": target, "request_id": request_id,
            "approved": bool(approved), "tool_enabled": tool_enabled, "requested_tool": tool,
        })


# ----------------------------------------------------------------------
# Trace: what was attempted, what the gate said, and whether the action really happened
# ----------------------------------------------------------------------
def describe_attempt(entry):
    """One sentence in the notebook's evidence-statement form, from a logged attempt."""
    result = entry.get("tool_result")
    tool_result = json.dumps(result, sort_keys=True) if result else "not produced (gate blocked before tool invocation)"
    before = entry["state_before"]["review_queue"]
    after = entry["state_after"]["review_queue"]
    return (
        f"The agent attempted {entry['requested_tool']}(args={json.dumps(entry['args'], sort_keys=True)}); "
        f"the gate returned {entry['gate']}; the tool result was {tool_result}; "
        f"state review_queue changed from {json.dumps(before, sort_keys=True)} to {json.dumps(after, sort_keys=True)}; "
        f"therefore the action {'did' if entry['action_occurred'] else 'did not'} occur."
    )


def format_trace(request_id, attempts):
    """Printable trace for one request ID (one block per logged attempt)."""
    if not attempts:
        return f"No attempts are logged for request_id {request_id!r}."
    lines = [SIMULATED_NOTE]
    for number, entry in enumerate(attempts, start=1):
        lines += [
            f"Attempt {number} for request_id {request_id!r}",
            f"  attempted call : {entry['requested_tool']}({json.dumps(entry['args'], sort_keys=True)})",
            f"  gate result    : {entry['gate']}",
            f"  tool result    : {json.dumps(entry['tool_result'], sort_keys=True) if entry.get('tool_result') else 'not produced (blocked before the tool ran)'}",
            f"  state change   : review_queue {len(entry['state_before']['review_queue'])} -> "
            f"{len(entry['state_after']['review_queue'])}",
            f"  action happened: {'yes' if entry['action_occurred'] else 'no'}",
            f"  {describe_attempt(entry)}",
        ]
    return "\n".join(lines)
