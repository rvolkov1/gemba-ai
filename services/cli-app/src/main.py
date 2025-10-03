import os
import sys
import click
import requests
import shlex
from minio import Minio
from minio.error import S3Error

# Fetch the API endpoint from environment variables
API_ENDPOINT = os.environ.get("API_ENDPOINT", "http://localhost:8000")

# MinIO Configuration
MINIO_ENDPOINT = os.environ.get("MINIO_ENDPOINT", "localhost:9000")
MINIO_ACCESS_KEY = os.environ.get("MINIO_ACCESS_KEY", "minioadmin")
MINIO_SECRET_KEY = os.environ.get("MINIO_SECRET_KEY", "miniodevpassword")

# Initialize MinIO client
try:
    minio_client = Minio(
        MINIO_ENDPOINT,
        access_key=MINIO_ACCESS_KEY,
        secret_key=MINIO_SECRET_KEY,
        secure=False
    )
except (S3Error) as e:
    click.echo(click.style(f"Error connecting to MinIO: {e}", fg="red"))
    minio_client = None

@click.group(invoke_without_command=True)
@click.pass_context
def cli(ctx):
    """A CLI to interact with the Gemba AI object detection service."""
    if ctx.invoked_subcommand is None:
        # No subcommand was provided, so enter interactive mode.
        click.echo("Entering interactive mode. Type 'exit' or 'quit' to leave.")
        
        while True:
            try:
                command = input("(gemba-cli) > ")
                if command.lower() in ["exit", "quit"]:
                    break
                
                # Use shlex to handle quoted arguments
                args = shlex.split(command)
                
                # Find and invoke the command
                cmd_name, cmd, args = cli.resolve_command(ctx, args)
                ctx.invoke(cmd, *args)

            except click.exceptions.UsageError as e:
                click.echo(e)
            except EOFError:
                # Handle Ctrl+D
                break
            except Exception as e:
                click.echo(f"An error occurred: {e}")

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
    # Pass arguments from sys.argv to the CLI
    cli(sys.argv[1:])
