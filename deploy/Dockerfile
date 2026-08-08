# Hugging Face Space, Docker SDK.
#
# One container serves both halves: uvicorn answers /predict and /explain, and
# the same process serves the page from frontend/ through CXR_SERVE_FRONTEND.
# Locally those are two servers on two ports, which is better while editing;
# here there is nowhere to put a second one.

FROM python:3.11-slim

# libgomp is torch's OpenMP runtime and is not in -slim. Without it the import
# fails at container start with a missing libgomp.so.1, which reads like a
# broken build rather than a missing apt package.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Spaces run the container as uid 1000. Writing as root here would leave files
# that user cannot read, and the app would start and then 500 on every request.
RUN useradd -m -u 1000 user
USER user
ENV PATH="/home/user/.local/bin:$PATH"
WORKDIR /home/user/app

COPY --chown=user:user requirements.txt .

# CPU wheels, explicitly. The default index resolves the CUDA build: several GB
# of kernels for a machine that has no GPU, and Spaces images have a size limit
# this would put us well past. --extra-index-url keeps everything else coming
# from PyPI as usual.
RUN pip install --no-cache-dir --user \
        --index-url https://download.pytorch.org/whl/cpu \
        --extra-index-url https://pypi.org/simple \
        -r requirements.txt

COPY --chown=user:user . .

ENV CXR_SERVE_FRONTEND=1 \
    CXR_CHECKPOINT=checkpoints/best.pt \
    CXR_OOD_STATS=checkpoints/ood.pt \
    PYTHONUNBUFFERED=1

# Spaces route to 7860 and ignore EXPOSE, but it documents the contract.
EXPOSE 7860

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "7860"]
