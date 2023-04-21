# Use the latest Ray master as base.
FROM rayproject/ray:nightly-py37
# Invalidate the cache so that fresh code is pulled in the next step.
ARG BUILD_DATE
# Retrieve your development code.
RUN git clone -b global-logging-2 https://github.com/peytondmurray/ray
# Install symlinks to your modified Python code.
RUN python ray/python/ray/setup-dev.py -y
