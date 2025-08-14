# AdalFlow Release Guide

## Overview

This guide documents the release process for AdalFlow, including both automated and manual release workflows.

## Release Methods

### Method 1: Automated GitHub Actions Release (Recommended)

Push a version tag to trigger the automated release workflow:

```bash
# IMPORTANT: Always create releases from the main branch
# First, ensure you're on main and up to date
git checkout main
git pull origin main

# Tag format: v<MAJOR>.<MINOR>.<PATCH>
git tag v1.1.2 -m "Release v1.1.2"
git push origin v1.1.2
```

**Note**: The GitHub Actions workflow will use the code from whatever branch/commit the tag points to. For production releases, always tag from the `main` branch after all changes have been merged.

The GitHub Actions workflow will automatically:
1. Update version in `__init__.py`
2. Update version in `pyproject.toml`
3. Update release date in `CHANGELOG.md`
4. Build the package
5. Create GitHub release
6. Publish to PyPI

### Method 2: Manual Release with Script

Use the `bump_version.py` script for local version management:

```bash
# Basic version bump (no git operations)
python scripts/bump_version.py 1.1.2

# Version bump with git commit
python scripts/bump_version.py 1.1.2

# Version bump with git commit and tag
python scripts/bump_version.py 1.1.2 --tag

# Version bump without committing (dry run)
python scripts/bump_version.py 1.1.2 --no-commit
```

### Method 3: Manual Workflow Dispatch

Trigger the release workflow manually from GitHub Actions:

1. Go to Actions tab in GitHub
2. Select "Release" workflow
3. Click "Run workflow"
4. Enter version number (e.g., 1.1.2)
5. Click "Run workflow"

## Branch Strategy

### Production Releases
- **Always release from `main` branch**
- Ensure all feature branches are merged to main first
- The main branch should be stable and tested

### Pre-releases (Beta/RC)
For pre-release versions, you may tag from feature branches:
```bash
# Example: Beta release from feature branch
git checkout feature-branch
git tag v1.2.0-beta.1 -m "Beta release v1.2.0-beta.1"
git push origin v1.2.0-beta.1
```

## Pre-Release Checklist

Before creating a release, ensure:

- [ ] You are on the `main` branch (for production releases)
- [ ] Branch is up to date: `git pull origin main`
- [ ] All tests pass locally
- [ ] CHANGELOG.md is updated with new features/fixes
- [ ] Documentation is updated if needed
- [ ] Version bump makes sense (follow [semantic versioning](https://semver.org/))
- [ ] All feature branches are merged

```bash
# Run tests
poetry run pytest

# Check linting
poetry run ruff check adalflow/

# Build and check package
poetry build
poetry check
```

## Version Numbering

Follow semantic versioning (MAJOR.MINOR.PATCH):

- **MAJOR**: Breaking changes
- **MINOR**: New features (backwards compatible)
- **PATCH**: Bug fixes (backwards compatible)

Examples:
- `1.0.0` → `2.0.0`: Breaking API changes
- `1.0.0` → `1.1.0`: New features added
- `1.0.0` → `1.0.1`: Bug fixes only

## Files Updated During Release

The release process updates these files:

1. **adalflow/adalflow/__init__.py**
   ```python
   __version__ = "1.1.2"  # Updated automatically
   ```

2. **adalflow/pyproject.toml**
   ```toml
   [tool.poetry]
   version = "1.1.2"  # Updated automatically
   ```

3. **adalflow/CHANGELOG.md**
   ```markdown
   ## [1.1.2] - 2025-08-11  # Date updated automatically
   ```

## Post-Release Steps

After a successful release:

1. **Verify PyPI Package**:
   ```bash
   pip install adalflow==1.1.2  # Test the new version
   ```

2. **Update Documentation**:
   - Check that docs reflect new version
   - Update any version-specific examples

3. **Announce Release**:
   - Create release notes on GitHub (automated)
   - Announce in Discord/Slack if applicable
   - Update project website if needed

## Troubleshooting

### Failed GitHub Actions Release

If the automated release fails:

1. Check Actions logs for error details
2. Common issues:
   - Missing `PYPI_API_TOKEN` secret
   - Version already exists on PyPI
   - Build failures

### Manual Recovery

If automation fails, manually release:

```bash
# Update versions locally
python scripts/bump_version.py 1.1.2

# Build package
cd adalflow
poetry build

# Upload to PyPI (requires credentials)
poetry publish

# Create GitHub release manually
gh release create v1.1.2 \
  --title "Release v1.1.2" \
  --notes "See CHANGELOG.md for details" \
  adalflow/dist/*
```

### Rollback a Release

If you need to rollback:

```bash
# Delete the tag locally and remotely
git tag -d v1.1.2
git push origin :refs/tags/v1.1.2

# Revert version changes if needed
git revert <commit-hash>
git push
```

## Environment Variables

For local publishing (not recommended for production):

```bash
export POETRY_PYPI_TOKEN_PYPI=<your-token>
```

## CI/CD Secrets Required

Ensure these secrets are configured in GitHub repository settings:

- `PYPI_API_TOKEN`: PyPI API token for publishing packages
- `GITHUB_TOKEN`: Automatically provided by GitHub Actions

## Quick Reference

```bash
# Check current version
python -c "import adalflow; print(adalflow.__version__)"

# Dry run version bump
python scripts/bump_version.py 1.1.2 --no-commit

# Full release with tag
python scripts/bump_version.py 1.1.2 --tag

# Trigger automated release
git tag v1.1.2 && git push origin v1.1.2
```

## Support

For release-related issues:
- Check [GitHub Issues](https://github.com/SylphAI-Inc/AdalFlow/issues)
- Contact maintainers
- Review [GitHub Actions logs](https://github.com/SylphAI-Inc/AdalFlow/actions)