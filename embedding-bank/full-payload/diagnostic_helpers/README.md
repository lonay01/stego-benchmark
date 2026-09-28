# Diagnostic stage helpers

Snapshot of the project's previous FIN, HiDDeN and DistSeal diagnostic operations
and audited manual-asset downloader. Bundled as `bank_diagnostics` for independent
reproduction from a clean checkout. The canary calls only `fin_stages`,
`hidden_stages`, `distseal_stages`, `fetch` and their helpers. Its `FullStages`
object scores the complete new payload; the historical common30 runner/scorer
in this snapshot is not called. Original diagnostic files and results are unchanged.
