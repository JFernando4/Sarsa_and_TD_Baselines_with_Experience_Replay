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
    python3 ./sarsa_zero_mountain_car_control.py -episodes 500 \
    -name Learning_Rate_Results/Learning_Rate_00000008/agent_$i \
    -quiet -tnetwork_update_freq 10 -alpha 0.0000008 -onpolicy -hidden_units 800 -max_steps 1000 \
    -replay_start 50
done