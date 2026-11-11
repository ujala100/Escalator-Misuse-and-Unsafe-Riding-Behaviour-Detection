# Project T253 — Escalator Misuse & Unsafe Riding Behaviour Detection
## Phase 4: Standardized Annotation Guidelines

**Version:** 1.0  
**Effective Date:** 2026-11-11  
**Target Domain:** Real-time CCTV Escalator Safety Monitoring  
**Learning Task:** Video-Level Temporal Activity Classification  

---

## 1. Architectural Scope & Task Separation

To ensure scientific rigor and avoid model confusion, Project T253 explicitly separates the computer vision responsibilities across three decoupled sub-systems:

```
+-----------------------------------------------------------------------------------------+
|                                    PROJECT T253 PIPELINE                                |
+-----------------------------------------------------------------------------------------+
| 1. OBJECT DETECTION & TRACKING                                                          |
|    - Architecture: Pretrained YOLOv8 (COCO Person Class) + ByteTrack                    |
|    - Function: Localizes each human with bounding box B = (x, y, w, h) & assigns TrackID|
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
| 2. TEMPORAL ACTIVITY CLASSIFIER (THIS DATASET)                                          |
|    - Architecture: Temporal Action Recognition (Pretrained CNN + Temporal Aggregation)  |
|    - Function: Classifies sequence of posture/motion into 5 unified T253 Activity Types |
+-----------------------------------------------------------------------------------------+
                                           |
                                           v
+-----------------------------------------------------------------------------------------+
| 3. SPATIAL & KINEMATIC BEHAVIOUR DETECTOR                                               |
|    - Wrong-Direction Walking: Track vector v_person against Escalator vector v_step     |
|    - Handrail Crossing: Bounding box intersection with polygonal handrail ROI           |
|    - Overcrowding: Track density in designated boarding / egress polygon zones         |
+-----------------------------------------------------------------------------------------+
```

> **CRITICAL RULE:** Bounding box annotations do not exist in the SPHAR dataset. We DO NOT fabricate bounding boxes or train YOLO object detection on video-level action labels.

---

## 2. Unified Target Class Definitions

The 5 unified target classes for Project T253 Activity Classification are defined below:

### Class 0: `NORMAL_RIDING_STANDING`
- **Class ID:** `0`
- **Definition:** Passenger standing stationary on moving escalator step or landing plate.
- **Visual Criteria:** Upright vertical torso; feet stationary; head facing direction of travel; hands resting on handrail or at side.
- **Permissible Variations:** Minor weight shifts, head rotation, glancing at phone while maintaining upright stationary balance.
- **Exclusion Criteria:** Deep squatting, walking steps, sitting, lying down, hand waving or acrobatic gestures.

### Class 1: `WALKING`
- **Class ID:** `1`
- **Definition:** Continuous alternating bipedal locomotion in upright posture.
- **Visual Criteria:** Rhythmic heel-strike and toe-off; standard cadence (~1.0–1.8 m/s); upright spine.
- **Permissible Variations:** Carrying luggage, holding personal bags, dragging rolling suitcase.
- **Directional Note:** General walking kinematics are classified by this model. Whether the walking is *wrong-direction* is determined by the downstream tracking module comparing trajectory vectors against the escalator mechanism.

### Class 2: `RUNNING_RUSHING`
- **Class ID:** `2`
- **Definition:** Rapid, hurried bipedal locomotion across steps or comb plate.
- **Visual Criteria:** Pronounced forward torso lean; extended stride lengths; presence of flight phase (both feet momentarily airborne); stride frequency >2.5 Hz.
- **Safety Hazard:** High probability of trip-and-fall, collision with stationary passengers, handrail over-run.

### Class 3: `SITTING_SQUATTING`
- **Class ID:** `3`
- **Definition:** Passenger seated directly on escalator steps or maintaining a deep squat.
- **Visual Criteria:** Center of mass lowered significantly (<65% of standing height); knees bent >90°; buttocks in contact with or hovering immediately above step surface.
- **Safety Hazard:** Severe entanglement hazard with comb plate, loose clothing entrapment in step gaps, blocking egress.

### Class 4: `FALLEN_PERSON`
- **Class ID:** `4`
- **Definition:** Passenger recumbent, supine, or collapsed on steps/floor following sudden balance loss, medical emergency, or collision.
- **Visual Criteria:** Rapid, uncontrolled transition from vertical to horizontal; body axis prone, supine, or sideways across multiple steps; loss of upright equilibrium.
- **Safety Hazard:** Critical immediate emergency; escalator emergency stop trigger required.

---

## 3. Disallowed Ambiguities & Quarantine Criteria

Any sample meeting the following conditions is quarantined into `reports/ambiguous_samples.csv` and excluded from model training:
1. **Vehicle Ingress / Egress:** Opening car doors or climbing into/out of automobiles (VIRAT dataset noise in SPHAR `sitting`).
2. **Extraneous Upper-Body Gestures:** Clapping, waving, hugging, handshakes (UCF-ARG/UT-Interaction noise in SPHAR `neutral`).
3. **Vehicle Trunk Operations:** Interacting with automotive car trunks.
4. **Transient Bending:** Brief forward torso bending without sitting or falling (creates severe multi-class confusion).
5. **Acrobatic Jumping:** Trampoline or vertical sports jumping (distinct from continuous forward rushing locomotion).

---
*End of Annotation Guidelines.*
