# Track 2A — Docker only (host needs just docker + make).
#
#   make run                 M1 (main system, Apertus-based) -> outputs/predictions
#                            B0 (model-free baseline)        -> outputs/baseline_b0
#                            offline: replay cache, --network none, no API key; format-validated; NO scoring
#   make run METHOD=M2       an extra method (M0, M2, M3 or B0 alone) -> outputs/extra/<METHOD>, also offline
#   make live                M1 against the Apertus endpoint (needs LLM_NAME, LLM_BASE_URL, LLM_API_KEY)
#   make eval PRED_PREFIX=<prefix> [SPLIT=dev]
#                            clones github.com/ZurichNLP/SwissGov-RSD at the pinned commit inside the image and
#                            runs their evaluate_predictions_admin.py (needs network at build time). Dev scoring.
#   make test                unit tests (those that need no organizers data)
#   make validate            format-validate outputs on the host (python3 only)
#
# Prediction format (docs/prediction_format.md): one <method>_admin_{de,fr,it}.jsonl.jsonl per language, rows in input
# order, {id,text_a,text_b,labels_a,labels_b}, one float in [0,1] per whitespace token, -1 on ASCII punctuation only.
.PHONY: run live eval test validate build

IMAGE       ?= polyjudge-2a:local
METHOD      ?= M1
INPUT_DIR   ?= data/test_input
SPLIT       ?= dev
ORG_COMMIT  ?= 1807a42100e742ed03d337c54c4b9ea86995f565
DOCKER_USER := $(shell id -u 2>/dev/null || echo 1000):$(shell id -g 2>/dev/null || echo 1000)
MOUNT_OUT   := -v "$(CURDIR)/outputs:/app/outputs"
RUN_OFFLINE := docker run --rm --network none --user $(DOCKER_USER) $(MOUNT_OUT) $(IMAGE)

ifeq ($(METHOD),M1)
OUT_DIR := outputs/predictions
else ifeq ($(METHOD),B0)
OUT_DIR := outputs/baseline_b0
else
OUT_DIR := outputs/extra/$(METHOD)
endif
MODE := $(if $(filter B0,$(METHOD)),offline,replay)

build:
	docker build --target runtime -t $(IMAGE) .

run: build
	mkdir -p outputs
	$(RUN_OFFLINE) python -m polyjudge.predict --method $(METHOD) --mode $(MODE) --input test --out $(OUT_DIR)
	$(RUN_OFFLINE) python scripts/validate_predictions.py --input-dir data/test_input --pred-dir $(OUT_DIR) --prefix $(METHOD)_admin_
ifeq ($(METHOD),M1)
	$(RUN_OFFLINE) python -m polyjudge.predict --method B0 --mode offline --input test --out outputs/baseline_b0
	$(RUN_OFFLINE) python scripts/validate_predictions.py --input-dir data/test_input --pred-dir outputs/baseline_b0 --prefix B0_admin_
endif

# Live mode: env vars are forwarded by name only (values are never echoed). New responses are added to the replay cache.
live: build
	@test -n "$$LLM_NAME" -a -n "$$LLM_BASE_URL" -a -n "$$LLM_API_KEY" || \
	  { echo "make live needs LLM_NAME, LLM_BASE_URL and LLM_API_KEY in the environment"; exit 1; }
	mkdir -p outputs
	docker run --rm --user $(DOCKER_USER) $(MOUNT_OUT) -e LLM_NAME -e LLM_BASE_URL -e LLM_API_KEY $(IMAGE) \
	  python -m polyjudge.predict --method $(METHOD) --mode live --input test --out $(OUT_DIR)
	$(RUN_OFFLINE) python scripts/validate_predictions.py --input-dir data/test_input --pred-dir $(OUT_DIR) --prefix $(METHOD)_admin_

# Scoring with the organizers' own script (not vendored here: their repo has no licence).
eval:
	@test -n "$(PRED_PREFIX)" || { echo "usage: make eval PRED_PREFIX=outputs/<dir>/<method>_admin_ [SPLIT=dev]"; exit 1; }
	docker build --target eval --build-arg ORGANIZERS_COMMIT=$(ORG_COMMIT) -t $(IMAGE)-eval .
	docker run --rm -v "$(CURDIR)/outputs:/outputs:ro" $(IMAGE)-eval \
	  python scripts/evaluate_predictions_admin.py /$(PRED_PREFIX) --split $(SPLIT)

test:
	docker build --target test -t $(IMAGE)-test .
	docker run --rm --network none $(IMAGE)-test

validate:
	python3 scripts/validate_predictions.py --input-dir data/test_input --pred-dir $(OUT_DIR) --prefix $(METHOD)_admin_
