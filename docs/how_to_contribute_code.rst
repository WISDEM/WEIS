.. how_to_contribute_code:

How to contribute code to WEIS
==============================

WEIS is an open-source tool, thus we welcome users to submit additions or fixes to the code to make it better for everybody.

Issues
------
If you have an issue with WEIS, a bug to report, or a feature to request, please submit an issue on the GitHub repository.
This lets other users know about the issue.
If you are comfortable fixing the issue, please do so and submit a pull request.

Documentation
-------------
When you add or modify code, make sure to provide relevant documentation that explains the new code.
This should be done in code via comments and also in the Sphinx documentation if you add a new feature or capability.
Look at the .rst files in the `docs` section of the repo or click on `view source` on any of the doc pages to see some examples.

There is currently very little documentation for WEIS, so you have a lot of flexibility in terms of where you place your new documentation.
Do not stress too much about the outline of the information you create, simply that it exists within the repo.
We will reorganize the documentation content at a later date.

Using Git subtrees
------------------

The WEIS repo contains copies of other codes created by using the `git subtree` commands.
Below are some details about how to add external codes and update them.

To add an external code, using OpenFAST as an example, type:

.. code-block:: bash

  $ git remote add OpenFAST https://github.com/OpenFAST/openfast
  $ git fetch OpenFAST
  $ git subtree add -P OpenFAST OpenFAST/dev --squash


The `--squash` is important so WEIS doesn't get filled up with commits from the subtree repos.

Once a subtree code exists in this repo, we can update it like this.
This first two lines are needed only if you don't have the remote for the particular subtree yet.
If you already have the remote, only the last line is needed.

.. code-block:: bash

  $ git remote add OpenFAST https://github.com/OpenFAST/openfast
  $ git fetch OpenFAST
  $ git subtree pull --prefix OpenFAST https://github.com/OpenFAST/openfast dev --squash --message="Updating to latest OpenFAST develop"

Changes to these subtree codes **should only be made to their original repos**, *not* to this WEIS repo.
Once those individual repos have been updated, use the previous `git subtree pull` command to pull in those updates to the WEIS repo.
Once the upstream repos have your code changes, those changes have been pulled into your branch, you can then submit a PR for WEIS.

If you run into trouble using `git subtree`, specifically if you see `git: 'subtree' is not a git command.`, try using your system git instead of any conda-installed git.
Specifically, try using `/usr/bin/git subtree` for any subtree commands.
If that doesn't work for you, please open an issue on this repo so we can track it.

Testing
-------
When you add code or functionality, add tests that cover the new or modified code.
These may be units tests for individual components or regression tests for entire models that use the new functionality.

Each discipline sub-directory contains tests in the `test` folder.
For example, `weis/multifidelity/test` hosts the tests for multifidelity optimization within WEIS.
Look at `test_simple_models.py` within that folder for a simple unit test that you can mimic when you add new code.

Pull requests
-------------
Once you have added or modified code, submit a pull request via the GitHub interface.
This will automatically go through all of the tests in the repo to make sure everything is functioning properly.
This also automatically does a coverage test to ensure that any added code is covered in a test.
The main developers of WEIS will then merge in the request or provide feedback on how to improve the contribution.