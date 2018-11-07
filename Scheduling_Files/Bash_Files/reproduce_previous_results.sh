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
    python3 ./sarsa_zero_mountain_car_control.py -episodes 500 -name Reproduce_Previous_Results/agent_$i \
    -quiet -tnetwork_update_freq 1000 -alpha 0.00025 -onpolicy -hidden_units 1000 -xavier_initialization \
    -max_steps 5000
done