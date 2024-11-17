# Use the base image we just created
FROM asia-northeast1-docker.pkg.dev/tikr-mvp/tikr-mvp-repo/tikr-mvp-base-image:20241117-amd64

# Copy your application code
COPY . .

# Command to run your application
CMD ["python", "tikr_mvp_main.py"]