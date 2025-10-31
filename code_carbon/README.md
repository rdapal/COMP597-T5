# Code Carbon

This directory is for exploration of the CodeCarbon library.

## `experimental_prometheus.py`

This file contains a modified output handler for Prometheus that resolves the bug in the CodeCarbon prometheus output handler, where labels are not constructed properly when pushing data to the API gateway.
