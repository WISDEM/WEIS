WEIS Installation
=================

To install WEIS, please follow the up-to-date instructions contained in the README.md at the root level of this repo, or on the `WEIS GitHub page <https://github.com/WISDEM/WEIS/>`_.

Installing SNOPT for use within WEIS
------------------------------------
SNOPT is available for purchase `here 
<http://www.sbsi-sol-optimize.com/asp/sol_snopt.htm>`_. Upon purchase, you should receive a zip file. Within the zip file, there is a folder called ``src``. To use SNOPT within WEIS, paste all files from ``src`` except snopth.f into ``WEIS/pyoptsparse/pyoptsparse/pySNOPT/source``.
If you do this step before you install WEIS, SNOPT will be automatically compiled within pyOptSparse and should be usable.
Otherwise, you can simply re-install WEIS following the same installation instructions after removing all ``build`` directories from the WEIS, WISDEM, and pyOptSparse directories.