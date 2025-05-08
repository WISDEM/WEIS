import textwrap
import validation
import os
import weis.inputs.validation as sch
import json, copy

mywidth  = 70
myindent = ' '*4
wrapper = textwrap.TextWrapper(initial_indent=myindent, subsequent_indent=myindent, width=mywidth)
rsthdr = ['*','#','=','-','^','~','%']
        
def get_type_string(indict):
    outstr = ''
    if indict['type'] == 'number':
        outstr = 'Float'
        if 'unit' in indict.keys() and indict['unit'].lower() != 'none':
            outstr+=', '+indict['unit']
        elif 'units' in indict.keys() and indict['units'].lower() != 'none':
            outstr+=', '+indict['units']
            
    elif indict['type'] == 'integer':
        outstr = 'Integer'

    elif indict['type'] == 'boolean':
        outstr = 'Boolean'

    elif indict['type'] == 'string':
        if 'enum' in indict.keys():
            outstr = 'String from, '+str(indict['enum'])
        else:
            outstr = 'String'

    elif indict['type'] == 'array':
        outstr = 'Array of '
        if indict['items']['type'] == 'number':
            outstr += 'Floats'
            if 'unit' in indict['items'].keys() and indict['items']['unit'].lower() != 'none':
                outstr+=', '+indict['items']['unit']
            elif 'units' in indict['items'].keys() and indict['items']['units'].lower() != 'none':
                outstr+=', '+indict['items']['units']
            
        elif indict['items']['type'] == 'integer':
            outstr += 'Integers'
        elif indict['items']['type'] == 'string':
            outstr += 'Strings'
        elif indict['items']['type'] == 'boolean':
            outstr += 'Booleans'

    return outstr


def get_description_string(indict):
    outstr = ''
    if 'description' in indict.keys():
        outstr += wrapper.fill(indict['description'])
        
    if 'default' in indict.keys():
        outstr += '\n\n'+myindent + '*Default* = '+str(indict['default'])
        
    if 'minimum' in indict.keys():
        outstr += '\n\n'+myindent + '*Minimum* = '+str(indict['minimum'])
    elif  (indict['type'] == 'array' and 'minimum' in indict['items'].keys()):
        outstr += '\n\n'+myindent + '*Minimum* = '+str(indict['items']['minimum'])
        
    if 'maximum' in indict.keys():
        outstr += myindent + '*Maximum* = '+str(indict['maximum'])+'\n'
    elif  (indict['type'] == 'array' and 'maximum' in indict['items'].keys()):
        outstr += '\n\n'+myindent + '*Maximum* = '+str(indict['items']['maximum'])
        
    outstr += '\n'
    return outstr


class Schema2RST(object):
    def __init__(self, fname):
        self.fname = fname
        self.fout  = fname.replace('.yaml','.rst')
        self.yaml  = validation.load_yaml( fname )
        self.f     = None

    def write_rst(self):
        self.f = open(self.fout, 'w')
        self.write_header()
        self.write_loop(self.yaml['properties'], 0, self.fname.replace('yaml',''))
        self.f.close()
        
    def write_header(self):
        self.f.write('*'*30+'\n')
        self.f.write(self.fname+'\n')
        self.f.write('*'*30+'\n')
        if 'description' in self.yaml.keys():
            self.f.write(self.yaml['description']+'\n')

    def write_loop(self, rv, idepth, name, desc=None):
        self.f.write('\n')
        self.f.write('\n')
        self.f.write(name+'\n')
        print(idepth)
        if idepth > 0: self.f.write(rsthdr[idepth-1]*40+'\n')
        self.f.write('\n')
        if not desc is None: self.f.write(desc+'\n')
        for k in rv.keys():
            print(k)
            try:
                if 'type' in rv[k]:
                    if rv[k]['type'] == 'object' and 'properties' in rv[k].keys():
                        k_desc = None if not 'description' in rv[k] else rv[k]['description']
                        self.write_loop(rv[k]['properties'], idepth+1, k, k_desc)
                    elif rv[k]['type'].lower() in ['number','integer','string','boolean']:
                        self.f.write(':code:`'+k+'` : '+get_type_string( rv[k] )+'\n')
                        self.f.write( get_description_string( rv[k] )+'\n')
                    elif (rv[k]['type'].lower() == 'array' and
                          rv[k]['items']['type'] == 'object' and
                          'properties' in rv[k]['items'].keys() ):
                        k_desc = None if not 'description' in rv[k]['items'] else rv[k]['items']['description']
                        self.write_loop(rv[k]['items']['properties'], idepth+1, k, k_desc)
                    elif (rv[k]['type'].lower() == 'array' and
                          rv[k]['items']['type'] in ['number','integer','string','boolean']):
                        self.f.write(':code:`'+k+'` : '+get_type_string( rv[k] )+'\n')
                        self.f.write( get_description_string( rv[k] )+'\n')
            except:
                print('Error reading,',k,'in',name,', depth',idepth)
                continue
            
if __name__ == '__main__':

    this_dir = os.path.dirname(__file__)
    docs_dir = os.path.realpath(os.path.join(this_dir,'../../docs/inputs'))

    # Merge schemas, write combined schema yamls here
    # Following https://github.com/WISDEM/WISDEM/blob/master/docs/schema/README

    modeling_schema = sch.get_modeling_schema()
    modeling_schema['definitions'] = copy.deepcopy(modeling_schema['properties'])
    modeling_schema.pop('properties')
    with open(os.path.join(docs_dir,'modeling_schema.json'),'w', encoding='utf-8') as f:
        json.dump(modeling_schema,f, ensure_ascii=False, indent=4)

    geometry_schema = sch.get_geometry_schema()
    temp_defs = copy.deepcopy(geometry_schema['definitions'])
    geometry_schema['definitions'] = copy.deepcopy(geometry_schema['properties'])
    geometry_schema['definitions'].update(temp_defs)
    geometry_schema.pop('properties')
    with open(os.path.join(docs_dir,'geometry_schema.json'),'w', encoding='utf-8') as f:
        json.dump(geometry_schema,f, ensure_ascii=False, indent=4)

    analysis_schema = sch.get_analysis_schema()
    analysis_schema['definitions'] = copy.deepcopy(analysis_schema['properties'])
    analysis_schema.pop('properties')
    with open(os.path.join(docs_dir,'analysis_schema.json'),'w', encoding='utf-8') as f:
        json.dump(analysis_schema,f, ensure_ascii=False, indent=4)





        
    
