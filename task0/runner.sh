docker build --tag task0 .
docker run --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task0