'''Main Page where we get the input file'''

# Import Packages
import dash
from dash import Dash, dcc, html
import os
import dash_bootstrap_components as dbc
import logging
import argparse
from weis.visualization.utils import checkPort, parse_yaml

this_dir = os.path.dirname( os.path.realpath(__file__) )
weis_dir = os.path.dirname( os.path.dirname( os.path.dirname( os.path.dirname( this_dir ) ) ) )

# Parse necessary arguments for running the app
parser = argparse.ArgumentParser(description='WEIS Visualization App')
parser.add_argument('--port', 
                    type=int, 
                    default=8050, 
                    help='Port number to run the WEIS visualization app'
                    )

parser.add_argument('--host', 
                    type=str, 
                    default="192.168.0.1", 
                    help='Host IP to run the WEIS visualization app'
                    )

parser.add_argument('--debug', 
                    type=bool,
                    default=False, 
                    help='Flag to activate debug mode'
                    )

parser.add_argument('--input', 
                    type=str, 
                    default=os.path.join(weis_dir, 'weis','visualization','appServer','app','tests','input','test.yaml'),    #'tests/input/test.yaml', # From apps (while locally running..) # lets point to an example where viz input could potentially exist.
                    help='Path to the WEIS visualization input yaml file'
                    )

args, unknown = parser.parse_known_args()


# Initialize the app - Internally starts the Flask Server
# Incorporate a Dash Mantine theme
external_stylesheets = [dbc.themes.BOOTSTRAP]
# For Latex
mathjax = 'https://cdnjs.cloudflare.com/ajax/libs/mathjax/2.7.4/MathJax.js?config=TeX-MML-AM_CHTML'
APP_TITLE = "WEIS Visualization APP"
app = Dash(__name__, external_stylesheets=external_stylesheets, external_scripts=[mathjax], suppress_callback_exceptions=True, title=APP_TITLE, use_pages=True)

# Build Navigation Bar
# Each pages are registered on each python script under the pages directory.
navbar = dbc.NavbarSimple(
    children = [
        dbc.NavItem(dbc.NavLink("Home", href='/')),
        dbc.NavItem(dbc.NavLink("OpenFAST", href='/open_fast')),
        dbc.NavItem(dbc.NavLink("Optimization", href='/optimize')),
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem('Blade', href='/wisdem_blade'), dbc.DropdownMenuItem('Cost', href='/wisdem_cost')],
            label="WISDEM",
            nav=True
        ),
        dbc.DropdownMenu(
            [dbc.DropdownMenuItem('3D', href='/windio_3d'), dbc.DropdownMenuItem('Airfoils', href='/windio_airfoils'), dbc.DropdownMenuItem('Blade', href='/windio_blade'), dbc.DropdownMenuItem('Tower', href='/windio_tower')],
            label="WindIO",
            nav=True
        )
    ],
    brand = APP_TITLE,
    color = "darkblue",
    dark = True,
    className = "menu-bar"
)

# Wrap app with loading component
# Whenever it needs some time for loading data, small progress bar would be appear in the middle of the screen.
print(args.input)
app.layout = html.Div(
            [   # Variable Settings to share over pages
                dcc.Store(id='input-dict', data=parse_yaml(args.input)),
                # WindIO Input Files
                dcc.Store(id='file-df', data={'File Path': [], 'Label': [], 'Type': []}),
                # dcc.Store(id='sorted-file-df', data={'model': [], 'analysis': [], 'geometry': []}),
                # Airfoils categorized by 'filelabelname:airfoilname' pairs
                dcc.Store(id='airfoil-by-names', data={}),
                # Geometry components categorized by 'filelabelname:componenttype' pairs
                dcc.Store(id='geometry-components', data={}),
                # Geometry file (whole wind turbine contents) categorized by 'filelabelname'
                dcc.Store(id='wt-options', data={}),
                navbar,
                dash.page_container
            ]
        )


def main():
    # test the port availability, flask calls the main function twice in debug mode
    if not checkPort(args.port, args.host) and not args.debug:
        print(f"Port {args.port} is already in use. Please change the port number and try again.")
        print(f"To change the port number, pass the port number with the '--port' flag. eg: python mainApp.py --port {args.port+1}")
        print("Exiting the app.")
        exit()

    logging.basicConfig(level=logging.DEBUG)        # For debugging
    app.run(debug=args.debug, host=args.host, port=args.port)



# Run the app
if __name__ == "__main__":
    main()
