SAMPLE_DATE=2014-11-22
SAMPLE_SYMBOL=XBTUSD

prefect-check:
	prefect deployment run "bitmex-etl-flow" --params '{"start": "$(SAMPLE_DATE)", "end": "$(SAMPLE_DATE)", "symbols": ["$(SAMPLE_SYMBOL)"]}'
	prefect deployment run "bitmex-qc-flow"

.PHONY: prefect-check
