# Module docstring: explains the guarded agent and its single permitted write
"""The guarded agent, ported from the Agents and Guarded Tool Use notebook.

The only permitted write is staging a documented interaction that exists in our
data into a SIMULATED pharmacist-review queue, and only the user can authorise
it. The controller checks the gates in a fixed order and only then lets the tool
run; the tool repeats its own checks at its boundary. This orchestrator is
deterministic: no language model is imported here, and nothing a model writes
can decide or trigger a write.
"""

import json  # turns dicts into text for the trace output
from copy import deepcopy  # independent copies so logged data cannot be changed later

from ddi_backend.checker import DOCUMENTED  # the status that means a documented interaction exists
from ddi_backend.common import DISCLAIMER, SIMULATED_NOTE  # standard texts: not medical advice, and "this is simulated"
from ddi_backend.state import PURPOSE  # the fixed purpose text for queue notes

REVIEW_TARGET = "interaction_review_queue"  # the only place the tool may write
REVIEW_TOOL = "stage_interaction_review"  # the only tool the agent may use

# The gate names are the failure codes: the dossier mentions A1 through A6 but never defines them.
GATE_MISSING_EVIDENCE = "missing_evidence"  # no documented interaction supports the request
GATE_TOOL_DISABLED = "tool_disabled"  # the tool switch is off
GATE_TOOL_NOT_ALLOWED = "tool_not_allowed"  # a different tool was asked for
GATE_OUT_OF_SCOPE = "out_of_scope"  # the write target is not the review queue
GATE_APPROVAL_REQUIRED = "approval_required"  # the user did not approve
GATE_DUPLICATE = "duplicate_request_id"  # the request ID was already used
GATE_ALLOWED = "allowed"  # every gate passed
GATES = (GATE_MISSING_EVIDENCE, GATE_TOOL_DISABLED, GATE_TOOL_NOT_ALLOWED,
         GATE_OUT_OF_SCOPE, GATE_APPROVAL_REQUIRED, GATE_DUPLICATE)  # the six blocking gates, in checking order

STAGED = "staged_in_simulation"  # success status: the note went on the simulated queue
ERROR = "error"  # the tool failed; never reported as success


class GuardedAgent:  # controller that decides, gates and then runs the tool
    def __init__(self, checker, state):  # needs the checker (evidence) and the state file (queue)
        self.checker = checker  # source of documented evidence
        self.state = state  # the review queue and request log

    # ------------------------------------------------------------------
    # Observe: evidence comes from the same deterministic pair lookup as everything else
    # ------------------------------------------------------------------
    def lookup_interaction(self, drug_pair):  # find the evidence for a pair
        """The headline documented record for the exact pair, or None.

        The match is exact (a direct pair query), so a partial keyword match can
        never stand in for the requested interaction.
        """
        names = list(drug_pair)  # accept a tuple or list
        if len(names) != 2:  # a pair needs exactly two names
            return None
        results = self.checker.check([names[0]], names[1])["results"]  # run the normal deterministic check
        if not results or results[0]["status"] != DOCUMENTED:  # only a documented interaction counts as evidence
            return None  # unknown, absent or ineligible all mean no evidence
        return results[0]["records"][0]  # the headline (most severe) record

    # ------------------------------------------------------------------
    # Condition A: answer-only baseline (never calls a tool)
    # ------------------------------------------------------------------
    def run_baseline(self, case):  # baseline that only recommends, never writes
        before = self.state.snapshot()  # state before, to show nothing changed
        evidence = self.lookup_interaction(case["drug_pair"])  # look for evidence
        if evidence is None:  # nothing documented
            recommendation = "No matching database evidence; do not recommend staging a review note."
        else:  # something documented
            recommendation = "Recommend clinician review of the recorded interaction; no treatment change."
        return {
            "condition": "A",  # label from the notebook
            "recommendation": recommendation,
            "evidence_id": evidence["id"] if evidence else None,  # the ddi: ID if any
            "state_before": before,
            "state_after": self.state.snapshot(),  # should equal state_before
            "executed_tool": None,  # no tool is ever run in the baseline
        }

    # ------------------------------------------------------------------
    # The tool: repeats the critical checks at its own boundary
    # ------------------------------------------------------------------
    def stage_interaction_review(self, args, approved, enabled):  # the one write tool
        before = self.state.snapshot()  # state before the attempt
        evidence = self.lookup_interaction(args["drug_pair"]) if args.get("evidence_id") else None  # re-check the evidence itself
        if not enabled:  # switch off
            result = {"status": "blocked", "reason": GATE_TOOL_DISABLED}
        elif not approved:  # user did not approve
            result = {"status": "blocked", "reason": GATE_APPROVAL_REQUIRED}
        elif args.get("target") != REVIEW_TARGET:  # wrong write target
            result = {"status": "blocked", "reason": GATE_OUT_OF_SCOPE}
        elif args["request_id"] in self.state.seen_request_ids:  # request ID reused
            result = {"status": "blocked", "reason": GATE_DUPLICATE}
        elif not args.get("evidence_id") or evidence is None or evidence["id"] != args["evidence_id"]:  # evidence missing or mismatched
            # The evidence must be a documented interaction that exists in our data for this pair.
            result = {"status": "blocked", "reason": GATE_MISSING_EVIDENCE}
        else:  # every check passed
            note = {  # the queue note
                "evidence_id": args["evidence_id"],  # which ddi: record supports it
                "drug_pair": list(args["drug_pair"]),  # the two drug names
                "request_id": args["request_id"],  # the unique request ID
                "purpose": PURPOSE,  # "clinician review only"
            }
            try:  # the write may fail
                self.state.stage_note(note)  # one transaction: the note and the used ID, or neither
                result = {"status": STAGED, "request_id": args["request_id"]}  # success
            except Exception as error:  # report the failure; never claim success
                result = {"status": ERROR, "reason": f"{type(error).__name__}: {error}",
                          "request_id": args["request_id"]}
        return result, {  # the result plus a trace entry describing the call
            "tool": REVIEW_TOOL,
            "args": deepcopy(args),  # copy so later changes cannot alter the trace
            "approved": approved,
            "enabled": enabled,
            "before": before,  # state before
            "after": self.state.snapshot(),  # state after
            "feedback": deepcopy(result),  # what the tool reported back
        }

    # ------------------------------------------------------------------
    # Condition B: observe -> decide -> permission gate -> tool -> feedback -> update/stop
    # ------------------------------------------------------------------
    def gate(self, case, evidence):  # the permission gate, checked in a fixed order
        if evidence is None:  # nothing to stage
            return GATE_MISSING_EVIDENCE
        if not case["tool_enabled"]:  # tool switched off
            return GATE_TOOL_DISABLED
        if case["requested_tool"] != REVIEW_TOOL:  # not the permitted tool
            return GATE_TOOL_NOT_ALLOWED
        if case["target"] != REVIEW_TARGET:  # not the permitted target
            return GATE_OUT_OF_SCOPE
        if not case["approved"]:  # only the user can authorise
            return GATE_APPROVAL_REQUIRED
        if case["request_id"] in self.state.seen_request_ids:  # replay protection
            return GATE_DUPLICATE
        return GATE_ALLOWED  # all gates passed

    def run_guarded(self, case):  # the full guarded flow for one request
        before = self.state.snapshot()  # state before the attempt
        events = []  # step-by-step log of what happened
        evidence = self.lookup_interaction(case["drug_pair"])  # observe
        events.append({"stage": "observe", "evidence_id": evidence["id"] if evidence else None})
        decision = f"request {case['requested_tool']}" if evidence else "stop: no supporting evidence"  # decide
        events.append({"stage": "decide", "decision": decision})

        args = {  # the arguments the tool would receive
            "target": case["target"],
            "request_id": case["request_id"],
            "drug_pair": list(case["drug_pair"]),
            "evidence_id": evidence["id"] if evidence else None,
        }
        events.append({"stage": "tool_request", "tool": case["requested_tool"], "args": deepcopy(args)})  # request the tool
        gate = self.gate(case, evidence)  # run the permission gate
        events.append({"stage": "permission_gate", "result": gate})

        tool_result, trace_entry = None, None  # stay None if the tool never runs
        if gate != GATE_ALLOWED:  # blocked: stop without calling the tool
            events.append({"stage": "stop", "reason": gate})
            status, executed = gate, None  # status is the gate name, nothing was executed
        else:  # allowed: call the tool
            events.append({"stage": "tool", "tool": REVIEW_TOOL, "args": deepcopy(args)})
            tool_result, trace_entry = self.stage_interaction_review(
                args, approved=case["approved"], enabled=case["tool_enabled"])  # the tool repeats its own checks
            events.append({"stage": "feedback", "result": deepcopy(tool_result)})  # feed the result back
            events.append({"stage": "update", "review_queue_size": len(self.state.review_queue)})  # note the new queue size
            events.append({"stage": "stop", "reason": "tool_feedback_received"})
            status = tool_result["status"] if tool_result["status"] in (STAGED, ERROR) else tool_result["reason"]  # staged, error or the block reason
            executed = REVIEW_TOOL  # the tool did run

        after = self.state.snapshot()  # state after the attempt
        action_occurred = len(after["review_queue"]) == len(before["review_queue"]) + 1  # proof from state: exactly one note added
        outcome = {  # the full report for this request
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
            "action_occurred": action_occurred,  # whether the action really happened
            "simulated": True,  # always labelled simulated
            "note": SIMULATED_NOTE,
            "disclaimer": DISCLAIMER,
        }
        self.state.log_attempt(case["request_id"], {  # log every attempt, even blocked ones
            **{k: outcome[k] for k in ("request_id", "requested_tool", "args", "gate", "tool_result",
                                       "state_before", "state_after", "action_occurred", "status", "events")},  # copy selected fields
            "approved": case["approved"],
            "enabled": case["tool_enabled"],
            "tool_trace_entry": trace_entry,
            "simulated": True,
        })
        return outcome

    def request_review(self, drug_pair, request_id, approved, target=REVIEW_TARGET,
                       tool=REVIEW_TOOL, tool_enabled=True):  # defaults point at the only permitted tool and target
        """Convenience wrapper around run_guarded for the CLI and API."""
        return self.run_guarded({  # build a case dict and run it
            "drug_pair": tuple(drug_pair), "target": target, "request_id": request_id,
            "approved": bool(approved), "tool_enabled": tool_enabled, "requested_tool": tool,
        })


# ----------------------------------------------------------------------
# Trace: what was attempted, what the gate said, and whether the action really happened
# ----------------------------------------------------------------------
def describe_attempt(entry):  # one-sentence evidence statement for a logged attempt
    """One sentence in the notebook's evidence-statement form, from a logged attempt."""
    result = entry.get("tool_result")  # None when the gate blocked the call
    tool_result = json.dumps(result, sort_keys=True) if result else "not produced (gate blocked before tool invocation)"  # text of the tool result
    before = entry["state_before"]["review_queue"]  # queue before
    after = entry["state_after"]["review_queue"]  # queue after
    return (
        f"The agent attempted {entry['requested_tool']}(args={json.dumps(entry['args'], sort_keys=True)}); "  # the call
        f"the gate returned {entry['gate']}; the tool result was {tool_result}; "  # gate and tool outcome
        f"state review_queue changed from {json.dumps(before, sort_keys=True)} to {json.dumps(after, sort_keys=True)}; "  # state change
        f"therefore the action {'did' if entry['action_occurred'] else 'did not'} occur."  # the conclusion
    )


def format_trace(request_id, attempts):  # printable trace for the CLI and API
    """Printable trace for one request ID (one block per logged attempt)."""
    if not attempts:  # nothing logged for this ID
        return f"No attempts are logged for request_id {request_id!r}."
    lines = [SIMULATED_NOTE]  # start with the simulated label
    for number, entry in enumerate(attempts, start=1):  # one block per attempt
        lines += [
            f"Attempt {number} for request_id {request_id!r}",  # heading
            f"  attempted call : {entry['requested_tool']}({json.dumps(entry['args'], sort_keys=True)})",  # what was called
            f"  gate result    : {entry['gate']}",  # what the gate said
            f"  tool result    : {json.dumps(entry['tool_result'], sort_keys=True) if entry.get('tool_result') else 'not produced (blocked before the tool ran)'}",  # tool output or none
            f"  state change   : review_queue {len(entry['state_before']['review_queue'])} -> "  # queue size before
            f"{len(entry['state_after']['review_queue'])}",  # and after
            f"  action happened: {'yes' if entry['action_occurred'] else 'no'}",  # whether it really happened
            f"  {describe_attempt(entry)}",  # the one-sentence summary
        ]
    return "\n".join(lines)  # join into the final text


# ---------------------------------------------------------------------------
# FILE SUMMARY: The guarded agent. It looks up a documented interaction as evidence, then runs a fixed-order
# permission gate (missing evidence, tool disabled, tool not allowed, out of scope, approval required, duplicate
# request ID) before the one permitted tool, which stages a note into the simulated pharmacist review queue. Every
# attempt is logged, and the trace functions print what was attempted, what the gate said and whether the action
# really happened. It is deterministic and imports no language model.
# ASSIGNMENT TOOL: Guarded tool use with a simulated review queue (the agent, ported from the Agents and Guarded Tool
# Use notebook).
# ---------------------------------------------------------------------------
