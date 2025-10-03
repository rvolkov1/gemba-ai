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

@cli.group()
def minio():
    """Commands for interacting with MinIO storage."""
    pass

@minio.command()
@click.argument('bucket_name')
def list(bucket_name):
    """Lists objects in a specified MinIO bucket."""
    if not minio_client:
        click.echo(click.style("MinIO client not initialized. Please check your connection settings.", fg="red"))
        return

    try:
        click.echo(f"Listing objects in bucket '{bucket_name}':")
        found = False
        for obj in minio_client.list_objects(bucket_name, recursive=True):
            click.echo(f"- {obj.object_name}")
            found = True
        if not found:
            click.echo(click.style(f"No objects found in bucket '{bucket_name}'.", fg="yellow"))
    except S3Error as e:
        click.echo(click.style(f"Error listing objects from MinIO: {e}", fg="red"))
    except Exception as e:
        click.echo(click.style(f"An unexpected error occurred: {e}", fg="red"))

@minio.command()
@click.argument('bucket_name')
@click.argument('object_name')
def process(bucket_name, object_name):
    """Triggers object detection for a video file in MinIO."""
    click.echo(f"Triggering object detection for '{object_name}' in bucket '{bucket_name}'...")
    try:
        response = requests.post(
            f"{API_ENDPOINT}/detect/video_file",
            params={
                "bucket_name": bucket_name,
                "object_name": object_name
            },
            timeout=300 # Increased timeout for video processing
        )
        response.raise_for_status()
        data = response.json()
        click.echo(click.style("Object detection triggered successfully!", fg="green"))
        click.echo(f"Response: {data}")
    except requests.exceptions.RequestException as e:
        click.echo(click.style(f"Error triggering object detection: {e}", fg="red"))
    except Exception as e:
        click.echo(click.style(f"An unexpected error occurred: {e}", fg="red"))

