# Top-level Makefile under src/
# Convenience targets to build CPU and GPU pricers together.

.PHONY: all cpu gpu clean

all: gpu cpu

cpu:
	$(MAKE) -C cpu

gpu:
	$(MAKE) -C gpu

clean:
	$(MAKE) -C cpu clean || true
	$(MAKE) -C gpu clean || true
