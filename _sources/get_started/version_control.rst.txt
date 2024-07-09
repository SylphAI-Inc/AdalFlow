.. _release-guide:

Release Version Control Guide
=======================================

Overview
--------

This guide outlines the process for releasing a new version of LightRAG. The workflow pipeline validates the version tag, builds the package, runs tests, publishes to PyPI, and creates a release on GitHub. The workflow is triggered by tags pushed to the **Release** branch. See `GitHub tags <https://docs.github.com/en/desktop/managing-commits/managing-tags-in-github-desktop>`_ for more details on version release tagging.

Steps to Release a New Version
------------------------------

1. Update the **./lightrag/pyproject.toml** version number and the latest dependencies before pushing a new release. Make sure to follow the `PEP 440 rules <https://peps.python.org/pep-0440/>`_ to define the version, otherwise, the workflow will fail. For example:

   .. code-block:: python

      [tool.poetry]
      name = "lightrag"
      
      version = "0.0.0-rc.1"
      description = "The 'PyTorch' library for LLM applications. RAG=Retriever-Agent-Generator."

2. Ensure your updates are the latest and correct. Update the version number following `Semantic Versioning <https://semver.org/>`. Here is a list of sample tags:

   .. code-block:: none

      Stable Release Tags:
      v1.0.0
      v1.2.3
      Pre-release Tags:
      v1.0.0-alpha.1
      v1.0.0-beta.1
      v1.0.0-rc.1
      Custom Pre-release Tags:
      v1.0.0-test.1
      v1.1.0-dev.2
      v1.2.3-pre.3
      v2.0.0-exp.4
      v2.1.0-nightly.5

3. The workflow will be triggered when new releases are pushed to the **release** branch. Push the **./lightrag/pyproject.toml** to the release branch:

   .. code-block:: python

      git add lightrag/pyproject.toml
      git commit -m "new version release"
      git push origin release
   
   Since the workflow only processes **`tags`**, your file submission will not go through the version release workflow.

   Only the tags you pushed will get checked.

To push the new version tag, please run:
   To push the new version tag:

   .. code-block:: python

      git tag -a vx.y.z -m "Release version x.y.z"
      git push origin release

4. To delete redundant local tags:

   .. code-block:: python

      git tags # list the existing tags
      
      git tag -d <tag>
      git push origin --delete <tag>

Important Notes
---------------

- **Do Not Reuse Tags:** If you need to fix a problem after a tag is pushed but before a release is made, you must create a new version number. Never reuse version numbers as this can lead to confusion and potential deployment issues.
- **Monitor the Workflow:** After pushing the tag, monitor the GitHub Actions workflow to ensure that it completes successfully. Check the "Actions" tab in the GitHub repository to see the progress and logs.

Troubleshooting
---------------

- If the workflow fails, review the logs for errors. Common issues might include failing tests or configuration errors in the workflow.
- If you encounter errors related to tagging (e.g., "tag already exists"), check that you're incrementing the version numbers correctly.
