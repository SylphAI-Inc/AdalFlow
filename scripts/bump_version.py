#!/usr/bin/env python3
"""
Version bump script for AdalFlow.

This script updates the version in:
- adalflow/__init__.py
- pyproject.toml
- CHANGELOG.md (updates the date for the version)

Usage:
    python scripts/bump_version.py <version>
    
Example:
    python scripts/bump_version.py 1.1.2
"""

import sys
import re
import subprocess
from pathlib import Path
from datetime import datetime
import argparse


def update_init_version(version: str, root_path: Path) -> bool:
    """Update version in __init__.py file."""
    init_file = root_path / "adalflow" / "adalflow" / "__init__.py"
    
    if not init_file.exists():
        print(f"Error: {init_file} not found")
        return False
    
    content = init_file.read_text()
    
    # Match the version line (with comment preservation)
    pattern = r'__version__ = "[^"]*"(\s*#.*)?'
    replacement = f'__version__ = "{version}"\\1'
    
    new_content = re.sub(pattern, replacement, content)
    
    if new_content == content:
        print(f"Warning: No version found in {init_file}")
        return False
    
    init_file.write_text(new_content)
    print(f"✓ Updated {init_file} to version {version}")
    return True


def update_pyproject_version(version: str, root_path: Path) -> bool:
    """Update version in pyproject.toml using poetry."""
    pyproject_dir = root_path / "adalflow"
    
    try:
        result = subprocess.run(
            ["poetry", "version", version],
            cwd=pyproject_dir,
            capture_output=True,
            text=True,
            check=True
        )
        print(f"✓ Updated pyproject.toml to version {version}")
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error updating pyproject.toml: {e.stderr}")
        return False
    except FileNotFoundError:
        print("Error: Poetry not found. Please install poetry first.")
        return False


def update_changelog_date(version: str, root_path: Path) -> bool:
    """Update the date for the given version in CHANGELOG.md."""
    changelog_file = root_path / "adalflow" / "CHANGELOG.md"
    
    if not changelog_file.exists():
        print(f"Warning: {changelog_file} not found")
        return True  # Not critical
    
    content = changelog_file.read_text()
    today = datetime.now().strftime("%Y-%m-%d")
    
    # Match version header with any date format
    pattern = rf"## \[{re.escape(version)}\] - .*"
    replacement = f"## [{version}] - {today}"
    
    new_content = re.sub(pattern, replacement, content, count=1)
    
    if new_content == content:
        print(f"Info: Version {version} not found in CHANGELOG.md (might be a new version)")
    else:
        changelog_file.write_text(new_content)
        print(f"✓ Updated CHANGELOG.md date to {today}")
    
    return True


def validate_version(version: str) -> bool:
    """Validate version format (semantic versioning)."""
    pattern = r'^\d+\.\d+\.\d+(-[a-zA-Z0-9.-]+)?(\+[a-zA-Z0-9.-]+)?$'
    if not re.match(pattern, version):
        return False
    return True


def create_git_tag(version: str) -> bool:
    """Create and push a git tag for the version."""
    try:
        # Create tag
        subprocess.run(
            ["git", "tag", f"v{version}", "-m", f"Release v{version}"],
            check=True,
            capture_output=True
        )
        print(f"✓ Created git tag v{version}")
        
        # Ask if should push
        response = input("Push tag to remote? (y/n): ").lower()
        if response == 'y':
            subprocess.run(
                ["git", "push", "origin", f"v{version}"],
                check=True,
                capture_output=True
            )
            print(f"✓ Pushed tag v{version} to remote")
        
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error creating git tag: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Bump AdalFlow version")
    parser.add_argument("version", help="New version (e.g., 1.1.2)")
    parser.add_argument("--tag", action="store_true", help="Create and push git tag")
    parser.add_argument("--no-commit", action="store_true", help="Don't commit changes")
    
    args = parser.parse_args()
    
    if not validate_version(args.version):
        print(f"Error: Invalid version format '{args.version}'")
        print("Expected format: MAJOR.MINOR.PATCH (e.g., 1.1.2)")
        sys.exit(1)
    
    # Get root path (parent of adalflow directory)
    root_path = Path(__file__).parent.parent
    
    print(f"Bumping version to {args.version}")
    print("-" * 40)
    
    # Update all version locations
    success = True
    success &= update_init_version(args.version, root_path)
    success &= update_pyproject_version(args.version, root_path)
    success &= update_changelog_date(args.version, root_path)
    
    if not success:
        print("\n❌ Some updates failed")
        sys.exit(1)
    
    print("\n✅ All version updates completed successfully")
    
    # Git operations
    if not args.no_commit:
        try:
            # Add files
            subprocess.run([
                "git", "add",
                "adalflow/adalflow/__init__.py",
                "adalflow/pyproject.toml",
                "adalflow/CHANGELOG.md"
            ], check=True)
            
            # Commit
            subprocess.run([
                "git", "commit", "-m", f"chore: bump version to {args.version}"
            ], check=True)
            
            print(f"✓ Committed version bump to {args.version}")
            
            if args.tag:
                create_git_tag(args.version)
                
        except subprocess.CalledProcessError as e:
            print(f"Error during git operations: {e}")
            sys.exit(1)
    
    print("\n🎉 Version bump complete!")
    
    if not args.tag:
        print(f"\nTo create a release, run:")
        print(f"  git tag v{args.version}")
        print(f"  git push origin v{args.version}")


if __name__ == "__main__":
    main()