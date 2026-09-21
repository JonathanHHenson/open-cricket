# Publishing open-cricket to PyPI

[Technical index](README.md)

## Release notes

- [v0.4.0: image inputs and vision-language models](../releases/v0.4.0.md)
- [v0.3.1: clearer Noul prompts and more CLI examples](../releases/v0.3.1.md)
- [v0.3.0: extensible case-insensitive codes and question batching](../releases/v0.3.0.md)
- [v0.2.0: default answer-code scoring and faster local inference](../releases/v0.2.0.md)

The versioned notes include user-facing changes, compatibility guidance, and
benchmark context, and can be used as the GitHub release description.

## Publishing workflow

The [GitHub Actions workflow](../../.github/workflows/publish.yml) tests Python
3.10 and 3.12, builds a source distribution and wheel, checks metadata with Twine,
and smoke-tests the installed wheel outside the checkout. The default CI suite
does not install HF or MLX, so their optional runtime tests skip.

Pushes to `main`, pull requests, and manual workflow runs validate artifacts
without publishing. Publishing a GitHub release triggers the same checks followed
by a PyPI upload. The release tag must exactly match `v` plus the version in
`pyproject.toml` (for example, `v0.3.0`). Published prereleases also trigger uploads;
use a PEP 440 prerelease version such as `0.3.0rc1` when appropriate.

## Release checklist

1. Update `project.version` in `pyproject.toml` and run `uv lock` to refresh metadata.
2. Run tests and review the artifact checks on GitHub.
3. Commit and push the changes, including the workflow, to GitHub.
4. Create and publish a GitHub release for that commit with tag `v<version>`.
5. Approve the `pypi` environment deployment if reviewers are configured.
6. Verify the workflow and the resulting PyPI release, then install it in a clean
   environment with `python -m pip install 'open-cricket==<version>'`.

For v0.4.0, keep `pyproject.toml` and the root package entry in `uv.lock` aligned.
Check publication status before publishing or changing tags.

For a local packaging check:

```bash
uv run --with build --with twine python -m build
uv run --with twine python -m twine check --strict dist/*
```

PyPI files cannot be overwritten. Correct failures before retrying; if a version
has already been published, prepare a new version. Manual runs validate only and
cannot accidentally publish. Actual publication requires account configuration
and a published release; no package is uploaded merely by adding this workflow.
