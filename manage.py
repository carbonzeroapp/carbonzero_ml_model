from flask.cli import FlaskGroup
from project.app import app_ins

cli = FlaskGroup(app_ins)

if __name__ == "__main__":
    cli()
    