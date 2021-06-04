FROM gcc:latest

LABEL author="scitools" \
      version="0.1" \
      description="Docker image with gcc and enchant"

#
# install system-level dependencies
#
RUN apt-get update -y \
  && apt-get upgrade -y \
  && apt-get install -y enchant-2
