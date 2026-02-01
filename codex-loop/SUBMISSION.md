# Hackathon Submission Materials

## Project Write-up (500 characters max)

**Character count: 497**

```
Traditional metrics tell you what happened—tool calls, completion rates. They don't tell you how it felt. Codex Closed Loop captures every session, uses LLMs to detect friction patterns that metrics miss—repeated denials, command failures, context churn—and distills them into reproducible evaluations. When patterns cluster and cross thresholds, they become Harbor tasks. The agent's failures become its training data. That's recursive self-improvement: every session makes the next better.
```

---

## OpenAI Usage Write-up (500 characters max)

**Character count: 496**

```
OpenAI models power three layers of abstraction. GPT-5.2 via Responses API analyzes sessions to extract friction signals—repeated rephrasing, escalation tone, platform confusion—without surfacing raw conversations. text-embedding-3-small embeds signals for HDBSCAN clustering, discovering patterns that don't fit existing categories. When clusters emerge, GPT-5.2 proposes new signal types. The taxonomy evolves itself. Codex is both subject and fix—recursive self-improvement.
```

---

## Demo Video Script (2 minutes)

**0:00-0:10** — Hook
> "What if Codex could learn from every session? That's Codex Closed Loop."

**0:10-0:25** — Problem (Stage 1-2)
> "AI agents fail predictably but don't learn. We fix that with a closed loop."

**0:25-0:50** — Data Collection (Stage 3)
> "We captured 500 Codex sessions automatically via trace spines."

**0:50-1:20** — Signal Detection (Stage 4-5)
> "Detected 1,736 signals: exec failures, denials, truncations. Ranked by impact."

**1:20-1:45** — Harbor Task (Stage 6)
> "Each friction pattern becomes a reproducible evaluation task for Harbor."

**1:45-2:00** — Closing (Stage 7)
> "Every session becomes data. Every friction becomes an eval. That's the loop."

---

## Checklist

- [ ] GitHub repo is public
- [ ] README has: what it does, how to run, demo steps
- [ ] Project write-up ≤500 chars ✓
- [ ] OpenAI write-up ≤500 chars ✓
- [ ] Demo video ≤2 minutes
- [ ] (Optional) Deployed prototype
