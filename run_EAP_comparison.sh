#!/bin/bash
Nwave='10 20 30 50 100 200 300 500 1000 2000 3000 5000 10000 20000 30000 50000'
for N in $Nwave; do
    python EAP_comparison.py $N
done
