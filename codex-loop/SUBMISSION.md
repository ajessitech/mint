# Hackathon Submission Materials

## Project Write-up (500 characters max)

**Character count: 489**

```
Codex Closed Loop turns every Codex session into improvement data. It automatically captures trace spines during normal usage, detects friction patterns (command failures, repeated denials, context truncation) and delight patterns (fast completion, zero denials), then distills problems into Harbor evaluation tasks. This creates a self-improving feedback loop: sessions become data, friction becomes evals, and fixes get measured. The result is systematic, evidence-based agent improvement.
```

---

## OpenAI Usage Write-up (500 characters max)

**Character count: 497**

```
We use OpenAI models at three key points: (1) GPT-4o detects semantic friction signals like repeated rephrasing, escalation tone, and platform confusion that rule-based detection misses. (2) text-embedding-3-small embeds signals into vectors for HDBSCAN clustering, automatically discovering new failure patterns. (3) GPT-4o generates human-readable tickets from signal clusters with evidence and reproduction steps. Codex itself is the agent being evaluated and improved through Harbor tasks.
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
