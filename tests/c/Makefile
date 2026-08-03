# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 lituus-lab
# Shell-free: `pwd` and `rm` are unavailable under mingw32-make on Windows.
ifeq ($(origin CC),default)
  ifeq ($(OS),Windows_NT)
    CC := gcc
  else
    CC := cc
  endif
endif

ROOT   := $(CURDIR)/../..
CFLAGS := -I$(ROOT)/include -O2 -Wall -Wextra -std=c11
LIB    := $(ROOT)/libUniLinalg.a

ifeq ($(OS),Windows_NT)
  BIN  := test_unilinalg.exe
  RUN  := $(BIN)
  RM_F := del /q
else
  BIN  := test_unilinalg
  RUN  := ./$(BIN)
  RM_F := rm -f
endif

.PHONY: all run clean
all: run
run: $(BIN)
	$(RUN)
$(BIN): test_unilinalg.c $(ROOT)/include/UniLinalg.h $(LIB)
	$(CC) $(CFLAGS) -o $(BIN) test_unilinalg.c $(LIB)
# File target, not phony: `nimble ctest` already built the lib, don't redo it.
$(LIB):
	cd $(ROOT) && nimble clibStatic
clean:
	$(RM_F) $(BIN)
