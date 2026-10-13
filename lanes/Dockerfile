ARG INFERENCE_IMAGE=ghcr.io/ggml-org/llama.cpp@sha256:ca61722be4814ebd0b36f2ad02643565a53198e4c752b06609d5666571b56409
FROM ${INFERENCE_IMAGE}
ARG WORKER_ARTIFACT=.build/worker
USER root
RUN command -v bash && command -v curl && test -s /etc/ssl/certs/ca-certificates.crt
RUN curl -fsSL https://www.apache.org/licenses/LICENSE-2.0.txt -o /opt/model-license.txt && curl -fsSL https://raw.githubusercontent.com/ggml-org/llama.cpp/e937b17a2/LICENSE -o /opt/llama-license.txt
COPY --chmod=0555 ${WORKER_ARTIFACT} /app/worker
COPY --chmod=0444 .build/model.gguf /opt/model.gguf
COPY --chmod=0444 .build/model-lock.json /opt/model-lock.json
COPY --chmod=0555 entrypoint.sh /app/entrypoint.sh
USER 10001:10001
EXPOSE 8090
ENTRYPOINT ["/app/entrypoint.sh"]
CMD ["serve"]
