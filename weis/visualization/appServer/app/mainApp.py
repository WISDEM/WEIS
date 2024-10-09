'''Main Page where we get the input file'''

# Import Packages
import dash
from dash import Dash, dcc, html
import dash_bootstrap_components as dbc
import logging
import argparse
from weis.visualization.utils import checkPort, parse_yaml


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
                    default='test.yaml', # lets point to an example where viz input could potentially exist.
                    help='Path to the WEIS visualization input yaml file'
                    )

args = parser.parse_args()


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
            [dbc.DropdownMenuItem('Import', href='/import'), dbc.DropdownMenuItem('3D', href='/windio_3d'), dbc.DropdownMenuItem('Airfoils', href='/windio_airfoils')],
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
file_indices = ['file1', 'file2', 'file3', 'file4', 'file5']       # Need to define as the way defined in .yaml file

app.layout = dcc.Loading(
    id = 'loading_page_content',
    children = [
        html.Div(
            [   # Variable Settings to share over pages
                dcc.Store(id='input-dict', data=parse_yaml(args.input)),
                # OpenFAST related Data fetched from input-dict
                dcc.Store(id='var-openfast', data={}),
                dcc.Store(id='var-openfast-graph', data={}),
                # Dataframe to share over functions - openfast .out file
                 html.Div(
                    [dcc.Store(id=f'df-{idx}', data={}) for idx in file_indices]      # dcc.Store(id='df-file1', data={}),          # {file1, df1}
                ),
                # Optimization related Data fetched from input-dict
                dcc.Store(id='var-opt', data={}),
                navbar,
                dash.page_container
            ]
        )
    ],
    color = 'primary',
    fullscreen = True
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