# First scan failure (derived note, not original stderr)

The first invocation completed vllm and sglang queries, then stopped on the
FlashInfer query with GitHub secondary rate limit (HTTP 403); process exit 1.
The original tool stderr was not redirected to a file. This note was added
after the fact; manifest.json contains only successful requests.
No claim of a complete six-repository scan is justified.
After a pause and script changes, a second invocation in a NEW directory
completed FlashInfer and LMDeploy, then hit secondary rate limiting on
LightLLM. Its failure.json preserves the returned error. No further API retry
was performed in this turn. Public web search and fixed source reads continued.
