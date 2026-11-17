# Local Backport Notes

> Historical notes for the pinned v0.30 branch. The tool-choice backport is
> already in current upstream and is not carried separately after the upgrade.
> The residual-gap discussion below describes the old branch, not a verified
> assessment of current upstream. See [upgrade notes](UPGRADE_NOTES.md).

Upstream changes carried by this fork that are **not** in our base release.

## response_format + tool_choice=auto (upstream #56086, fixes #39929)

- **Commit:** `34f40991ac` — cherry-pick of upstream `2b9b55c7f1`
  ("[Bugfix] Support response_format + tool_choice=auto (#56086)", merged
  2026-09-27, first shipped in upstream v0.31.0).
- **What it does:** when a chat request sets both `tools` and `response_format`
  with `tool_choice: "auto"` (or unset/null), the structured-output constraint
  used to box decoding into plain JSON so tool-call tokens could never be
  emitted (`tool_calls: []`). This composes the tool-call structural tag with
  the response_format schema (OR of the two), so the model may either call a
  tool or emit a schema-conforming answer.
- **Touches:** `vllm/parser/abstract_parser.py`, `vllm/parser/harmony.py`,
  `vllm/parser/cohere_command.py`, `vllm/tool_parsers/structural_tag_registry.py`,
  and `tests/parser/{test_abstract_parser,test_harmony}.py`. Pure Python, no C
  extensions, so a service restart is all that's needed (editable install).
- **Revert:** `git revert 34f40991ac` then restart the vLLM service.

### Known residual gap (open upstream)

With `--reasoning-parser` active, the `\n\n` the chat template emits after
`</think>` sits *inside* the constrained region. The tool-vs-JSON branch is
chosen at that first constrained token, so its natural probability mass is
masked out and the branch can be picked from residual mass — intermittently
under load. Upstream did not fix this in #56086; it is being tracked by
reviving the closed #48157 (composed grammar + leading-whitespace handling).
If you see the model sporadically pick the wrong branch, that is this gap, not
a backport defect. A second small backport will follow once it lands.
