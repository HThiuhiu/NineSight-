# Build: `mingw32-make` (Windows/MinGW) hoặc `make` (Linux/macOS)
CXX      ?= g++
CXXFLAGS := -std=c++17 -O2 -Wall -Wextra -pthread
BUILD    := build

ifeq ($(OS),Windows_NT)
  EXE     := .exe
  LDLIBS  := -lws2_32
else
  EXE     :=
  LDLIBS  :=
endif

HEADERS := $(wildcard common/*.hpp)
BINS    := $(BUILD)/dummy_server$(EXE) $(BUILD)/closed_loop$(EXE) $(BUILD)/open_loop$(EXE)

all: dirs $(BINS)

# `mkdir <dir>` chạy được trên cả cmd.exe lẫn sh; make tự kiểm tra thư mục đã tồn tại chưa.
dirs:
	$(if $(wildcard $(BUILD)),,mkdir $(BUILD))
	$(if $(wildcard results),,mkdir results)

$(BUILD)/dummy_server$(EXE): target/dummy_server.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/closed_loop$(EXE): harness/closed_loop.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/open_loop$(EXE): harness/open_loop.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@ $(LDLIBS)

$(BUILD)/test_hdr$(EXE): tests/test_hdr.cpp $(HEADERS)
	$(CXX) $(CXXFLAGS) $< -o $@

test: dirs $(BUILD)/test_hdr$(EXE)
	$(BUILD)/test_hdr$(EXE)
	python -m unittest discover -s tests

.PHONY: all dirs test
