import openmdao.api as om

def load_OMsql(log, verbose=False):
  """ load the openmdao sql file produced by a WEIS run """

  if verbose:
    print(f"loading {log}")

  cr = om.CaseReader(log) # openmdao reader for recorded output data

  rec_data = {} # create a dict for output data that's been recorded
  for case in cr.get_cases("driver"): # loop over the cases
    for key in case.outputs.keys(): # for each key in the outputs
      if key not in rec_data: # if this key isn't present, create a new list
        rec_data[key] = []
      rec_data[key].append(case[key]) # add the data to the list

  return rec_data # return the output