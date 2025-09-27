import os
import click
import requests

# Fetch the API endpoint from environment variables
API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:8000")

@click.group()
def cli():
    """A CLI to interact with the Gemba AI object detection service."""
    pass

@cli.command()
def healthcheck():
    """Checks the health of the object detection API."""
    click.echo(f"Pinging the API at {API_ENDPOINT}...")
    try:
        response = requests.get(API_ENDPOINT, timeout=5)
        response.raise_for_status()  # Raise an exception for bad status codes
        data = response.json()
        click.echo(click.style("API is healthy!", fg="green"))
        click.echo(f"Response: {data}")
    except requests.exceptions.RequestException as e:
        click.echo(click.style(f"API is unreachable: {e}", fg="red"))

if __name__ == "__main__":
    cli()
