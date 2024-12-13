# Use the base image we just created
# FROM asia-northeast1-docker.pkg.dev/tikr-mvp/tikr-mvp-repo/tikr-mvp-base-image-20241117-amd64
FROM docker.io/library/tikr-mvp-baseimage:20241129-amd64

# Copy the requirements file
COPY requirements_mvp.txt .

# Install Python dependencies. These should already be installed in
# the base image. This is just to double-check if any dependencies
# were added later
RUN pip install --no-cache-dir -r requirements_mvp.txt

# Copy your application code
COPY . .

# Command to run your application
CMD ["python", "tikr_mvp_main.py"]