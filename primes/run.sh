#!/bin/bash

# This runs the program while avoiding the printing of the prime numbers
# Useful for benchmarking

time ./primes $@ > /dev/null
