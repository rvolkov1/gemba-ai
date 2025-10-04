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
        click.echo("Entering interactive mode. Type 'exit' or 'quit' to leave.")
        
        while True:
            try:
                command = input("(gemba-cli) > ")
                if command.lower() in ["exit", "quit"]:
                    break

                if command.strip().lower() in ["help", "--help"]:
                    click.echo(ctx.get_help())
                    continue
                
                args = shlex.split(command)
                
                # Convert 'help' as the last argument to '--help' for subcommands
                if args and args[-1].lower() == 'help':
                    args[-1] = '--help'

                try:
                    cli.main(args=args, standalone_mode=False)
                except SystemExit:
                    pass

            except click.exceptions.UsageError as e:
                click.echo(e)
            except EOFError:
                break
            except Exception as e:
                click.echo(f"An error occurred: {e}")

@cli.command()
def healthcheck():
    """Checks the health of the object detection API.

    This command pings the root (/) endpoint of the object detection service.
    A successful check means the API is running and reachable from the CLI.
    """
    click.echo(f"Pinging the API at {API_ENDPOINT}...")
    try:
        response = requests.get(API_ENDPOINT, timeout=5)
        response.raise_for_status()
        data = response.json()
        click.echo(click.style("API is healthy!", fg="green"))
        click.echo(f"Response: {data}")
    except requests.exceptions.RequestException as e:
        click.echo(click.style(f"API is unreachable: {e}", fg="red"))


@cli.group(invoke_without_command=True)
@click.pass_context
def minio(ctx):
    """Commands for interacting with MinIO storage."""
    if ctx.invoked_subcommand is None:
        click.echo(ctx.get_help())

@minio.command(name='list')
@click.argument('bucket_name', required=False)
def list_objects(bucket_name):
    """Lists all buckets, or objects in a specified bucket."""
    if not minio_client:
        click.echo(click.style("MinIO client not initialized. Please check your connection settings.", fg="red"))
        return

    try:
        if bucket_name is None:
            click.echo("Available MinIO buckets:")
            buckets = minio_client.list_buckets()
            if not buckets:
                click.echo(click.style("No buckets found.", fg="yellow"))
            for bucket in buckets:
                click.echo(f"- {bucket.name}")
        else:
            click.echo(f"Listing objects in bucket '{bucket_name}':")
            found = False
            for obj in minio_client.list_objects(bucket_name, recursive=True):
                click.echo(f"- {obj.object_name}")
                found = True
            if not found:
                click.echo(click.style(f"No objects found in bucket '{bucket_name}'.", fg="yellow"))
    except S3Error as e:
        click.echo(click.style(f"Error communicating with MinIO: {e}", fg="red"))
    except Exception as e:
        click.echo(click.style(f"An unexpected error occurred: {e}", fg="red"))

@minio.command()
def process():
    """Triggers the batch object detection and visualization process.

    This command starts a background task on the object detection service to
    process all new videos. It identifies "new" videos by checking for
    video files in the 'user-data' bucket that do not have a corresponding
    .json result file in the 'user-out' bucket.

    For each new video, two output files are generated:
    1. A .json file with the raw detection data, placed in 'user-out'.
    2. A .mp4 video file with detections drawn on it, placed in 'user-out-visual'.

    The command returns immediately after starting the background task.
    """
    click.echo("Triggering batch object detection...")
    try:
        response = requests.post(
            f"{API_ENDPOINT}/detect/batch",
            timeout=300
        )
        response.raise_for_status()
        data = response.json()
        click.echo(click.style("Batch detection process started in the background.", fg="green"))
        click.echo(f"API Response: {data['message']}")
    except requests.exceptions.RequestException as e:
        click.echo(click.style(f"Error triggering object detection: {e}", fg="red"))
    except Exception as e:
        click.echo(click.style(f"An unexpected error occurred: {e}", fg="red"))

if __name__ == "__main__":
    cli(sys.argv[1:])