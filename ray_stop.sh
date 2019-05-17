#!/bin/bash

set -e

source env.sh

host_string=""
for node in ${nodes}
do
  host_string+="${node} "
done

parallel-ssh -i -H "${host_string}" -x "-t -t" " /bin/bash -i -c '
  source \${HOME}/.bashrc
  export PYTHONPATH=${PYTHONPATH}
  echo \$(which ray)
  ray stop
  sleep 0.5
'"

echo "stopping on ${head_node}"
ssh -t ${head_node} " /bin/bash -i -c '
  source \${HOME}/.bashrc
  export PYTHONPATH=${PYTHONPATH}
  which ray
  ray stop
  sleep 2
'"

