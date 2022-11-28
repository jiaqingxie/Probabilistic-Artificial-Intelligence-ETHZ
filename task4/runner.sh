docker build --tag task4 .
docker run --init --rm -v "$( cd "$( dirname "$0" )" && pwd )":/results task4
