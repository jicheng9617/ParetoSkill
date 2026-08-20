# ParetoSkill

**Regression-Aware Multi-Objective Evolution of Transferable Agent Skills**

ParetoSkill studies how to evolve trajectory-grounded agent-skill patches when
deployment quality is inherently multi-objective. Instead of returning a single
validation-optimal artifact, the method maintains an uncertainty-aware archive
that trades off:

- in-domain accuracy;
- worst-target transfer across models, harnesses, and domains;
- token cost; and
- paired regression on tasks solved by the base skill.

The current repository is the public home for implementation, experiment
configuration, and the project website. The manuscript is maintained separately
in Overleaf so that paper collaboration does not mix with code history.

## Status

The problem formulation, method, and falsifiable evaluation protocol are drafted.
Experiments are not yet complete, so this repository makes no empirical claims.

## Repository layout

```text
code/    implementation and experiment entry points
docs/    GitHub Pages project website
```

Private literature PDFs, initial idea archives, and manuscript sources are
intentionally excluded from this repository.

## Project page

The GitHub Pages URL will be added after the first deployment.

