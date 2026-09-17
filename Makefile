PROTOC ?= protoc
PROTO_DIR := src/voyager_compiler/codegen
PROTO_NAMES := param tiling
PROTO_SOURCES := $(addprefix $(PROTO_DIR)/,$(addsuffix .proto,$(PROTO_NAMES)))
PROTO_BINDINGS := $(addprefix $(PROTO_DIR)/,$(addsuffix _pb2.py,$(PROTO_NAMES)))

.PHONY: generate-protos check-generated-protos

# Regenerate the checked-in Python bindings from their canonical schemas
generate-protos: $(PROTO_SOURCES)
	$(PROTOC) -I=$(PROTO_DIR) --python_out=$(PROTO_DIR) $(PROTO_SOURCES)

# Reject stale checked-in bindings without modifying the working tree
check-generated-protos: $(PROTO_SOURCES) $(PROTO_BINDINGS)
	@temporary=$$(mktemp -d); \
	trap 'rm -rf "$$temporary"' EXIT; \
	$(PROTOC) -I=$(PROTO_DIR) --python_out="$$temporary" $(PROTO_SOURCES); \
	status=0; \
	for binding in $(notdir $(PROTO_BINDINGS)); do \
		diff -u "$(PROTO_DIR)/$$binding" "$$temporary/$$binding" || status=$$?; \
	done; \
	exit $$status
