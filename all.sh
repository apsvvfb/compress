#!/bin/bash

for ((i=1;i<6;i++))
do
	echo $i
	echo "temp2"
	th temp2.lua $i
	echo "test.lua"
	th test.lua
done
