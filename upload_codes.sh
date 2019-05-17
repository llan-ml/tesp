#!/bin/bash

set -e

source env.sh

for server in ${servers}
do
  scp -r tesp/* ${server}:\${HOME}/Workspaces/tesp
done
