FROM gcc:latest

LABEL author="scitools" \
      version="0.1" \
      description="Docker image with gcc and enchant"

#
# install system-level dependencies
#
RUN apt update -y \
  && apt upgrade -y \
  && apt install -y enchant
