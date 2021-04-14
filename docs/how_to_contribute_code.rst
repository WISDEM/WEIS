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
Look at the .rst files in the :code:`docs` section of the repo or click on :code:`view source` on any of the doc pages to see some examples.

There is currently very little documentation for WEIS, so you have a lot of flexibility in terms of where you place your new documentation.
Do not stress too much about the outline of the information you create, simply that it exists within the repo.
We will reorganize the documentation content at a later date.

Using Git subtrees
------------------

The WEIS repo contains copies of other codes created by using the :code:`git subtree` commands.
Below are some details about how to add external codes and update them.

Adding a subtree for the first time
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To add an external code for the first time, using OpenFAST as an example, type:

.. code-block:: bash

  $ git remote add OpenFAST https://github.com/OpenFAST/openfast
  $ git fetch OpenFAST
  $ git subtree add -P OpenFAST OpenFAST/dev --squash


The :code:`--squash` is important so WEIS doesn't get filled up with commits from the subtree repos.

Updating a subtree
~~~~~~~~~~~~~~~~~~

Once a subtree code exists in this repo, we can update it like this.
This first two lines are needed only if you don't have the remote for the particular subtree yet.
If you already have the remote, only the last line is needed.

.. code-block:: bash

  $ git remote add OpenFAST https://github.com/OpenFAST/openfast
  $ git fetch OpenFAST
  $ git subtree pull --prefix OpenFAST https://github.com/OpenFAST/openfast dev --squash --message="Updating to latest OpenFAST develop"

Changes to these subtree codes **should only be made to their original repos**, *not* to this WEIS repo.
Once those individual repos have been updated, use the previous :code:`git subtree pull` command to pull in those updates to the WEIS repo.
Once the upstream repos have your code changes, those changes have been pulled into your branch, you can then submit a PR for WEIS.

Troubleshooting subtree updates
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

If you run into trouble using :code:`git subtree`, specifically if you see :code:`git: 'subtree' is not a git command.`, try using your system git instead of any conda-installed git.
Specifically, try using :code:`/usr/bin/git subtree` for any subtree commands.
If that doesn't work for you, please open an issue on this repo so we can track it.

Sometimes when updating a subtree, you might get an error that contains :code:`could not rev-parse split hash`, which suggests that the subtree and the more recent branch have diverged in some way.
To fix this, manually remove the entire folder created by the subtree command (e.g. :code:`rm -rf ROSCO`.)
Then, follow the steps to add a repo subtree for the first time.
By deleting then re-adding the subtree repo, you ensure that all the changes are correctly added to the WEIS repo.

Testing
-------
When you add code or functionality, add tests that cover the new or modified code.
These may be units tests for individual code blocks or regression tests for entire models that use the new functionality.
These tests should be a balance between minimizing computational cost and maximizing code coverage.
This ensures continued functionality of WEIS while keeping development time short.

Any Python file with :code:`test` in its name within the :code:`weis` package directory is tested with each commit to WEIS.
This is done through GitHub Actions and you can see the automated testing progress on the GitHub repo under the :code:`Actions` tab, `located here <https://github.com/WISDEM/WEIS/actions>`_.
If any test fails, this information is passed on to GitHub and a red X will be shown next to the commit.
Otherwise, if all tests pass, a green check mark appears to signify the code changes are valid.

Unit tests
~~~~~~~~~~ 

Each discipline sub-directory should contain tests in the :code:`test` folder.
For example, :code:`weis/multifidelity/test` hosts the tests for multifidelity optimization within WEIS.
Look at :code:`test_simple_models.py` within that folder for a simple unit test that you can mimic when you add new code.
Another simple unit test is contained in :code:`weis/aeroelasticse/test` called :code:`test_IECWind.py`.

Unit tests should be short and purposeful, test the smallest reasonable block of code, and quickly point to potential problems in the code.
`This article <https://dzone.com/articles/10-tips-to-writing-good-unit-tests>`_ has some quick tips on how to write good unit tests.

Regression tests
~~~~~~~~~~~~~~~~

Regression tests examine much larger portions of the code by examining top-level input and output relationships.
Specifically, these tests check the values that the code produces against "truth" values and returns an error if they do not match.
As an example, a low-level coding change might alter a default within a subsystem of the model being tested, which might result in a different AEP value for the wind turbine.
The regression test would report that the AEP value differs, and thus the tests fail.
Of course, it would be challenging to completely diagnose a coding change based on only regression tests, so well-made unit tests can help narrow down a problem much more quickly.

Within WEIS, regression tests live in the :code:`weis/test` folder.
Examine :code:`test_aeroelasticse/test_DLC.py` to see an example regression test that checks OpenFAST results obtained through WEIS' wrapper.
Specifically, that test compares all of the channel outputs against truth values contained in :code:`.pkl` files within the same folder.

Like unit tests, regression tests should run quickly.
They can have unrealistic simulation parameters (1 second timeseries) as long as they adequately test the code.


Coveralls
~~~~~~~~~

To understand how WEIS is tested, we use a tool called `Coveralls <https://coveralls.io/github/wisdem/WEIS>`_, which reports the lines of code that are used during testing.
This lets WEIS developers know which functions and methods are tested, as well as where to add tests in the future.

When you push a commit to WEIS, all of the unit and regression tests are ran.
Then, the coverage from those tests is reported to Coveralls automatically.

There may be some files that you specifically do not want Coveralls to report on, such as plotting scripts or helper scripts that aren't part of the main WEIS repo.
To not include certain files, add them to the `omit` section of the `.coverageac` file at the root repo level.
For example, we currently do not report coverage on the `schema2rest.py` file, which is listed in the `omit` section.


Pull requests
-------------
Once you have added or modified code, submit a pull request via the GitHub interface.
This will automatically go through all of the tests in the repo to make sure everything is functioning properly.
This also automatically does a coverage test to ensure that any added code is covered in a test.
The main developers of WEIS will then merge in the request or provide feedback on how to improve the contribution.

In addition to the full unit and regression test suite, on pull requests additional examples are checked using GitHub Actions using the workflow labeled :code:`run_exhaustive_examples`.
These examples are useful for users to adapt, but are computationally expensive, so we do not test them on every commit.
Instead, we test them only when code is about to be added to the main WEIS develop or master branches through pull requests.
The coverage from these examples are not considered in Coveralls.

The examples that are covered are shown in :code:`weis/test/run_examples.py`.
If you add an example to WEIS, make sure to add a call to it in the :code:`run_examples.py` script as well.