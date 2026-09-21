# Publishing open-cricket to PyPI

[Technical index](README.md)

The [GitHub Actions workflow](../../.github/workflows/publish.yml) tests Python
3.10 and 3.12, builds a source distribution and wheel, checks metadata with Twine,
and smoke-tests the installed wheel outside the checkout. The default CI suite
does not install HF or MLX, so their optional runtime tests skip.

Pushes to `main`, pull requests, and manual workflow runs validate artifacts
without publishing. Publishing a GitHub release triggers the same checks followed
by a PyPI upload. The release tag must exactly match `v` plus the version in
`pyproject.toml` (initially `v0.1.0`). Published prereleases also trigger uploads;
use a PEP 440 prerelease version such as `0.2.0rc1` when appropriate.

## Release checklist

1. Update `project.version` in `pyproject.toml` and run `uv lock` to refresh metadata.
2. Run tests and review the artifact checks on GitHub.
3. Commit and push the changes, including the workflow, to GitHub.
4. Create and publish a GitHub release for that commit with tag `v<version>`.
5. Approve the `pypi` environment deployment if reviewers are configured.
6. Verify the workflow and the resulting PyPI release, then install it in a clean
   environment with `python -m pip install 'open-cricket==<version>'`.

For a local packaging check:

```bash
uv run --with build --with twine python -m build
uv run --with twine python -m twine check --strict dist/*
```

PyPI files cannot be overwritten. Correct failures before retrying; if a version
has already been published, prepare a new version. Manual runs validate only and
cannot accidentally publish. Actual publication requires account configuration
and a published release; no package is uploaded merely by adding this workflow.
