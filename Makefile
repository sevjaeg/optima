.PHONY: all
all: fit sim

.PHONY: fit
fit:
	make -C discharge-model fit

.PHONY: sim
sim:
	make -C system-model sim

.PHONY: clean
clean:
	make -C discharge-model clean
	make -C system-model clean
