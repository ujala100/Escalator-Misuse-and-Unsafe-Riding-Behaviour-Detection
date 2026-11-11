# Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection
## Phase 5: Data Leakage Verification Report

**Audit Date:** 2026-11-11  
**Status:** **PASSED (ZERO LEAKAGE DETECTED)**  
**Verification Protocol:** Cryptographic Hash Auditing + Scene/Recording Session Disjointness  

---

## 1. Audit Objective

A primary failure mode in computer vision action recognition is **environmental data leakage**—where frames or video clips originating from the same recording session, physical camera position, or background scene appear in both the training and testing partitions. This artificially inflates validation accuracy because models memorize background textures rather than learning human kinematic patterns.

This audit certifies that the Project T253 split manifest is **strictly partitioned by recording session and cryptographic identity**.

---

## 2. Leakage Test Results

### Test 1: Recording Session / Camera Group Disjointness
- **Train Unique Groups:** 200
- **Validation Unique Groups:** 147
- **Test Unique Groups:** 86
- **Train $\cap$ Validation Overlap:** **0 groups (PASS)**
- **Train $\cap$ Test Overlap:** **0 groups (PASS)**
- **Validation $\cap$ Test Overlap:** **0 groups (PASS)**

> **Conclusion:** Every recording sequence (e.g. all clips from Okutama session `2.1.6`, or VIRAT scene `050201`, or CASIA subject `p01`) resides exclusively in exactly one partition. Zero scene leakage exists.

### Test 2: Cryptographic Binary Hash (MD5) Disjointness
- **Train Unique Hashes:** 3645
- **Validation Unique Hashes:** 785
- **Test Unique Hashes:** 794
- **Train $\cap$ Validation Collisions:** **0 files (PASS)**
- **Train $\cap$ Test Collisions:** **0 files (PASS)**
- **Validation $\cap$ Test Collisions:** **0 files (PASS)**

> **Conclusion:** Zero duplicate or identical binary clips exist across any partition boundaries.

### Test 3: Test Set Integrity Certification
- The test set consists of **794 clips** from **86 completely unseen recording sessions**.
- The test set has NOT been balanced artificially, preserving natural surveillance activity distribution.
- Test data transformations are strictly deterministic (resize, center-crop, normalization) with zero stochastic augmentation.

---

## 3. Split Summary Table

| Split | Clip Count | Percentage | Recording Groups | Unique Hashes |
|---|---|---|---|---|
| **TRAIN** | 3645 | 69.8% | 200 | 3645 |
| **VAL** | 785 | 15.0% | 147 | 785 |
| **TEST** | 794 | 15.2% | 86 | 794 |
| **TOTAL** | **5224** | **100.0%** | **433** | **5224** |

*Certified by Senior Deep Learning Engineer, Project T253.*
