.. _known_issues:

Known issues within WEIS
========================

This doc page serves as a non-exhaustive record of any issues relevant to the usage of WISDEM.
Some of these items are features that would be nice to have, but are not necessarily in the pipeline of development.

Running on Eagle
----------------
Depending on the method that send batch scripts to Eagle, they may not run correctly in parallel.
Specifically, when calling ``sbatch submit_job.sh`` and the job involves MPI and WEIS, multiple users have reported issues.
These issues manifest as the script starting correctly, but then returning ``MPI_INIT`` errors.

One user could not successfully run jobs submitted via VS Code, but could via terminal.
For another user, the regular Windows command terminal worked, but not the Ubuntu subsystem.
**If you have any issues regarding running scripts on Eagle, first try a few different terminals to submit sbatch jobs**.

The reason for this error is not known.
