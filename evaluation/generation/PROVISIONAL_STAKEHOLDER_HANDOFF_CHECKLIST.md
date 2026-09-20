# Provisional Stakeholder Handoff Checklist
**Package Title**: Provisional Audit Package & Verified Subset Reconciliation (11 Verified Pairs / 25 Completed in Prior Workflow—Local Synchronization Pending)  
**Evaluation Scope**: 15 Active Cases (36 Atomic Propositions) | Case 007 Excluded (Truncated)  
**Status**: PROVISIONAL AUDIT PACKAGE ONLY (Full Benchmark Completion & Full-Dataset Cohen's Kappa are ON HOLD)  
**Repository State**: Zero tracked file modifications (`git diff` is empty); branch `main` is ahead of `origin/main` by 1 commit (`d9e26c5`); untracked files preserved untouched.  
**Safety Guardrail**: `ENABLE_CONFORMAL_TRUST = False` strictly confirmed in `config.py:45`.

---

### 1. Scope & Execution Checklist

- [x] **Provisional Package Scope Confirmed**: The package is strictly scoped to the provisional audit of the 11 locally verified pairs and the canonical 36-proposition decomposition framework; full benchmark release is **NOT** claimed.
- [x] **25 Prior-Workflow Pairs Preserved**: Preserved as completed by reviewers during the prior ChatGPT workflow; exact physical label synchronization into the local repository files is **ON HOLD** (these records are not unannotated or missing review).
- [x] **Verified Metric Scoping**: Raw observed agreement of **72.73% (8/11)** applies strictly and solely to the 11 locally verified pairs.
- [ ] **HOLD — Full-Dataset Cohen's Kappa**: Withheld and strictly **ON HOLD** until the 25 external pairs are synchronized locally and protocol conventions are formally ratified.
- [ ] **HOLD — Full Benchmark Completion**: Formally **ON HOLD** pending local label synchronization, policy ratification, and compound precedence sign-off.
- [ ] **HOLD — Substantive Disagreement Ratification**: Three verified reviewer disagreements remain open for formal stakeholder policy decisions:
  - **Case 009 - Proposition A2** (*Thermal Mechanism*): Reviewer 1 `SUPPORTED` vs. Reviewer 2 `UNSUPPORTED` (Macroscopic cooling vs. direct solar-heating reduction).
  - **Case 011 - Proposition A3** (*Nutrient Release Mechanism*): Reviewer 1 `UNSUPPORTED` vs. Reviewer 2 `SUPPORTED` (Biological nitrogen-fixation domain inference vs. textual explicitness).
  - **Case 013 - Proposition A1** (*Weeding Schedule Intervals*): Reviewer 1 `CONTRADICTED` vs. Reviewer 2 `UNSUPPORTED`.
- [x] **Case 013-A1 Adjudication Status**: The proposed adjudicated label `UNSUPPORTED` remains **PROPOSED ONLY — NOT RATIFIED BY STAKEHOLDERS**; schedule-mismatch policy remains unratified.
- [ ] **HOLD — Compound-Label Precedence Approval**: Stakeholder approval for the three compound aggregation precedence mappings remains **PENDING RATIFICATION**; compound-level IAA remains deferred.
- [x] **Case 007 Exclusion**: Formally **EXCLUDED** from active evaluation due to mid-token raw generation truncation at character 349 (`"...and for public‑"`).
- [x] **Orthographic Token Fidelity**: Exact raw generated spelling **`Pedimethalin 30%`** in Case 013-A2 is preserved verbatim without silent normalization.
- [x] **Production Guardrail**: Invariant confirmed: `ENABLE_CONFORMAL_TRUST = False` strictly maintained in `config.py:45`.
- [x] **Read-Only / Zero Side Effects**: No production code, configuration files, embeddings, vector indexes, or benchmark ground truth files were modified. No files staged, committed, pushed, or transmitted over the network.

---

### 2. Consolidated Reconciliation Ledger Summary

* **Active Propositions**: 36 Total
  - **Locally Verified Subset**: 11 Propositions (8 Agreements, 3 Disagreements = 72.73% raw agreement).
  - **Prior-Workflow Completed Subset**: 25 Propositions (Completed in prior workflow; local synchronization pending).
  - **Genuinely Unannotated**: 0 Propositions.
* **Adjudication Ledger State**: 1 Proposed Adjudication (Case 013-A1, unratified); 2 Disagreements open for ratification (009-A2, 011-A3).

---

### 3. Immediate Next Operational Triggers

1. **Local Synchronization**: Ingest the 25 completed prior-workflow pairs from the external ChatGPT export into the local matrix.
2. **Policy Ratification**: Stakeholders convene to ratify the 3 open reviewer disagreements (Cases 009-A2, 011-A3, and 013-A1).
3. **Compound Precedence Ratification**: Formally approve or amend the compound precedence aggregation rules.
4. **Final Reliability Run**: Calculate the complete 36-item Cohen’s kappa ($\kappa$) and publish the finalized benchmark report.
