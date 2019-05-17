#!/bin/bash

source env.sh

pushd ./ray
for server in ${servers}
do
  scp -r rllib tune ${server}:\${HOME}/.pyenv/versions/anaconda3-5.2.0/lib/python3.6/site-packages/ray
done
popd
