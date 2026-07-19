# Release and supply-chain checklist

This template is intentionally repository-specific at application time.

- Build sdist and wheel from a clean tagged commit.
- Run tests against the built wheel, not only the editable checkout.
- Publish to PyPI through trusted publishing rather than a long-lived token.
- Produce an SBOM for Python and optional native artifacts.
- Attach checksums and provenance/attestations where the release platform supports them.
- Reconcile package version, Git tag, `CITATION.cff`, Zenodo/OSF metadata, docs version, and conda recipe.
- Verify licence metadata and bundled third-party notices.
- Confirm that local-only data, reviewer correspondence, and generated working artifacts are absent.
