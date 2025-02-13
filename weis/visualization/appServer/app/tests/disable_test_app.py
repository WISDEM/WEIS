from contextvars import copy_context
from dash._callback_context import context_value
from dash._utils import AttributeDict
import dash

# Import all of the names of callback functions to tests
from weis.visualization.appServer.app.mainApp import app

def test_app_creation():
    assert isinstance(app, dash.Dash)

def test_layout_components():
    assert app.layout is not None
