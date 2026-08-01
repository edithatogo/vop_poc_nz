# Specialized estimation and study-efficiency VOI governance

## Overview

C16 is the canonical cross-repository planning contract for the specialized
VOIAGE v1.2.0 family. It governs estimation-focused variance reduction, COSS,
EVSI/EVPI efficiency, utility-equivalent information prices and the VoC
presentation while preserving the VOIAGE runtime repository as implementation
owner.

## Requirements

1. Canonical requirements M14–M17 and this track's MoSCoW requirements agree.
2. VOIAGE tracks `estimation_focused_variance_voi_20260727`,
   `study_design_efficiency_20260727`, and
   `risk_adjusted_information_pricing_20260731` remain the detailed
   implementation plans. The omnibus
   `supported_frontier_method_completion_20260723` remains their parent
   frontier programme and is not completed by one child delivery.
3. GitHub hierarchy #313 > #318 > #571/#595/#619 remains native and
   deduplicated; #595 owns native delivery issues #694–#697.
4. Project 28 exposes MoSCoW `Must`, Contract Version `v1.2.0`, Track ID,
   Record ID, planned/unverified evidence and synchronization state.
5. A versioned public projection can be consumed by any registered repository;
   synchronization preserves human content, detects conflicts and fails closed
   when credentials or repository registration are absent.

## Acceptance criteria

- **AC-01:** Canonical and VOIAGE MoSCoW requirements have one-to-one IDs and
  planned-version mappings.
- **AC-02:** Mermaid designs show estimand, optimizer and synchronization
  boundaries without implying runtime completion.
- **AC-03:** Conductor, GitHub hierarchy and Project 28 fields agree.
- **AC-04:** Other repositories consume the canonical projection rather than
  copying mutable prose or importing runtime source.
- **AC-05:** Automated sync is bounded to managed sections and project fields;
  merge, release, issue closure and risk acceptance remain human governed.

## External gates

Cross-repository write credentials, hosted checks, merge and release remain
separate external gates. Missing credentials produce planned or blocked sync
state, never a false clean result.

## Out of scope

- Implementing the specialized numerical methods in VOP.
- Duplicating VOIAGE issues or Conductor implementation tracks.
- Automatically merging pull requests or closing issues.
