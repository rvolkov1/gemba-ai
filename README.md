# To run:
- Install [Docker Desktop](https://docs.docker.com/desktop/) on [MacOS](https://docs.docker.com/desktop/setup/install/mac-install/), [Windows](https://docs.docker.com/desktop/setup/install/windows-install/), or [Linux](https://docs.docker.com/desktop/setup/install/linux/)
- Clone this git repository
- Inside this repository, copy the environment variables with `cp .env.example .env`
- Run `./gemba-cli up`
- To interact with the CLI, run `./gemba-cli`
- If needed to tear down, run `./gemba-cli down`

## Running Object Detection
- Navigate the the MinIO in your web browser at `https://localhost:9001` and upload your video files in the `user-data` bucket
    - Use the username and password from the `.env`
    - You can use [these example video files](https://github.com/intel-iot-devkit/sample-videos?tab=readme-ov-file)
- In the CLI, run `minio process`
- After some time, JSON output will be available in `user-out` bucket, and an annotated video file will be available in `user-out-visual` bucket