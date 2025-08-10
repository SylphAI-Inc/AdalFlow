#!/bin/bash

# Simple script to publish AdalFlow to PyPI
# Usage: ./scripts/publish.sh <version>

VERSION=$1

if [ -z "$VERSION" ]; then
    echo "Usage: $0 <version>"
    echo "Example: $0 1.1.3"
    exit 1
fi

echo "Publishing AdalFlow version $VERSION to PyPI"
echo "==========================================="

# Get the script directory and project root
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"

# Navigate to adalflow package directory
cd "$PROJECT_ROOT/adalflow"

# Update version using poetry
echo "1. Updating version to $VERSION..."
poetry version $VERSION

# Update version in __init__.py
echo "2. Updating __init__.py..."
sed -i '' "s/__version__ = \".*\"/__version__ = \"$VERSION\"/" adalflow/__init__.py

# Build the package
echo "3. Building package..."
poetry build

# Publish to PyPI
echo "4. Publishing to PyPI..."
poetry publish --skip-existing

echo ""
echo "✅ Successfully published AdalFlow $VERSION to PyPI!"
echo ""
echo "Next steps:"
echo "1. Commit the version changes: git add -A && git commit -m 'Release v$VERSION'"
echo "2. Create a git tag: git tag v$VERSION"
echo "3. Push to GitHub: git push origin main --tags"