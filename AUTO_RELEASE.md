# Auto Release

Simple automated release process for AdalFlow pip package.

## How to Release

### Option 1: Tag Release (Automatic)
```bash
git tag v1.1.2
git push origin v1.1.2
```

### Option 2: Manual Trigger
1. Go to GitHub Actions → Release workflow
2. Click "Run workflow"
3. Enter version number
4. Click "Run workflow"

## What Happens Automatically

✅ Updates version in all files  
✅ Builds package  
✅ Publishes to PyPI  
✅ Creates GitHub release  

## Requirements

- Tag from `main` branch
- Update `CHANGELOG.md` first
- Ensure tests pass

## Version Format

- `v1.2.3` (patch: bug fixes)
- `v1.3.0` (minor: new features) 
- `v2.0.0` (major: breaking changes)

## Troubleshooting

**Failed release?** Check [Actions logs](https://github.com/SylphAI-Inc/AdalFlow/actions)

**Need details?** See [scripts/release.md](scripts/release.md)