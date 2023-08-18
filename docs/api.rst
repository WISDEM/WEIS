API
===

This will provide a list of all callable functions, classes, and methods.


Conventions
^^^^^^^^^^^

As discussed in :ref:`Model Structure`, Point and Body objects can be either fixed, coupled, or free. 
The object's "type" attribute describes this, with 1=fixed, 0=free, -1=coupled. These properties can
be changed after a mooring system has been created (for example, to simulate a disconnection).



The System Class
^^^^^^^^^^^^^^^^

.. autoclass:: moorpy.system.System
   :members:

