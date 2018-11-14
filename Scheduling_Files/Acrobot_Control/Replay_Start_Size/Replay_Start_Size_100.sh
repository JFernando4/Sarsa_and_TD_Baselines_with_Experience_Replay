#!/usr/bin/env bash

echo Enter the number of agents you want to train:
read number_of_agents
echo Enter the number of the first agent:
read first_agent_number
last_agent_number=$(($first_agent_number + $number_of_agents - 1))

export PYTHONPATH=.
for (( i=$first_agent_number; i <= $last_agent_number; i++ ))
do
    echo "Training Agent $i"
    python3 ./sarsa_zero_acrobot_control.py -episodes 500 \
    -name Replay_Start_Size/Replay_Start_Size_100/agent_$i \
    -quiet -tnetwork_update_freq 100 -alpha 0.00025 -onpolicy -hidden_units 2000 -max_steps 500 \
    -replay_start 100
done