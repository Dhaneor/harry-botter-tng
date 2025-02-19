# Use the base image we just created
# FROM asia-northeast1-docker.pkg.dev/tikr-mvp/tikr-mvp-repo/tikr-mvp-base-image-20241117-amd64
# FROM docker.io/library/tikr-mvp-baseimage:20241129-amd64
FROM tikr-mvp-baseimage:20241129-amd64

ENV PYTHONPATH=/app:/app/src

COPY requirements_mvp.txt .

# Install Python dependencies. These should already be installed in
# the base image. This is just to double-check if any dependencies
# were added later
RUN pip install --no-cache-dir -r requirements_mvp.txt

COPY . .

RUN python setup.py build_ext --inplace

# Command to run your application
CMD ["python", "tikr_mvp_main.py"]