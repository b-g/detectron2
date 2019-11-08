#!/bin/bash
DOCKER_VOLUMES+="-v $(pwd)/:/detectron2_repo "
shift
docker run --runtime=nvidia -it $DOCKER_VOLUMES detectron2:v0
