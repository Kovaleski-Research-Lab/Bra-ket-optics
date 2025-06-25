#!/bin/bash
#
# This script runs all tests in the tests/ directory and reports the results.

pytest -vv --order-group-scope=module
