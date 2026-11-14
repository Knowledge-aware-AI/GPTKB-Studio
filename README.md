A branch that drops the distinction between NEs/generic concepts/literals, and just distinguishes concepts from literals. It expands on all concepts, which makes the process slower, but also more general.

Further minor tweaks on performance:

Model requests have a **total wall-clock deadline per attempt**, not just a
socket inactivity timeout: 300 seconds by default (`request_timeout_seconds`),
and 60 seconds for embeddings. The deadline covers endpoint validation,
connection setup, redirects and receipt of the complete response, even if the
endpoint keeps sending partial data. At expiry, the local HTTP operation is
cancelled and the existing bounded retry/backoff policy applies. Each retry
gets a fresh deadline; rate-limit pacing and retry delays are outside it, so
the total time for all attempts can exceed one deadline.

Relation disambiguation and initial concept disambiguation size their batches
to at least the configured worker count. A 50-thread run can therefore issue
up to 50 requests concurrently in those stages when enough independent work
is ready. Smaller thread settings remain the concurrency limit. 
